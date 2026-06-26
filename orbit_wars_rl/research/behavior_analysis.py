"""Behavioral profiler — characterize HOW an agent plays, not just its win rate.

Win-rate panels can't tell us whether an agent "knows when to expand vs defend vs
reinforce" — the actual goal. This runs N games of agent A vs agent B (both .py
agent files) and reports A's behavioral profile, split by wins vs losses (to catch
bimodal passive lock-in), across game phases.

Metrics (for agent A):
  EXPANSION   planets owned at step 25/50/75/100 — are we grabbing the board?
  FORCE       avg ships/launch by phase — decisive commitment vs 1-ship probes
  AGGRESSION  fire_rate, multi-source rate by phase
  DEFENSE     reinforce_rate = frac of launches sent to OUR OWN planets (reinforce/
              defend) vs attacks on neutral/enemy; and response to incoming threats
  OUTCOME     win/loss, game length

Usage:
  python orbit_wars_rl/behavior_analysis.py --agent opponents/rev12_6M_peak.py \
      --opponent opponents/candidate_landgrab.py --games 12
"""
from __future__ import annotations
import argparse, math, statistics as st
from collections import defaultdict


def _classify_launch(obs, me, src_id, angle):
    """Resolve which planet a launch [src_id, angle, ships] is aimed at, and return
    'reinforce' (target is our planet), 'attack' (enemy/neutral), or 'none'.
    Target = planet best matching the launch direction from the source."""
    src = next((p for p in obs['planets'] if p[0] == src_id), None)
    if src is None:
        return 'none'
    sx, sy = src[2], src[3]
    best, best_d = None, 0.6  # radians tolerance
    for p in obs['planets']:
        if p[0] == src_id:
            continue
        pa = math.atan2(p[3] - sy, p[2] - sx)
        d = abs((pa - angle + math.pi) % (2 * math.pi) - math.pi)
        if d < best_d:
            best_d, best = d, p
    if best is None:
        return 'attack'  # aimed at open space (still aggressive intent)
    return 'reinforce' if best[1] == me else 'attack'


def analyze_game(steps, a_slot, a_won):
    """Per-game behavioral summary for agent in a_slot."""
    me = steps[1][a_slot]['observation'].get('player', a_slot)
    own_ids_by_step = []
    phase_force = defaultdict(list)   # phase -> ship sizes
    phase_fire = defaultdict(lambda: [0, 0])  # phase -> [fire_steps, total]
    reinforce = attack = 0
    planets_at = {}
    n = len(steps)
    for t, s in enumerate(steps[1:], start=1):
        if a_slot >= len(s):
            continue
        obs = s[a_slot].get('observation', {})
        if not obs:
            continue
        own_ids = {p[0] for p in obs['planets'] if p[1] == me}
        if t in (25, 50, 75, 100):
            planets_at[t] = len(own_ids)
        phase = 'open' if t < 60 else ('mid' if t < 150 else 'late')
        # threatened = we have enemy fleets in flight while behind on planets
        threatened = any(f[1] != me for f in obs.get('fleets', []))
        acts = s[a_slot].get('action') or []
        phase_fire[phase][1] += 1
        if acts:
            phase_fire[phase][0] += 1
            for a in acts:
                if len(a) >= 3:
                    phase_force[phase].append(a[2])
                    kind = _classify_launch(obs, me, a[0], a[1])
                    if kind == 'reinforce':
                        reinforce += 1
                    else:
                        attack += 1
    return dict(won=a_won, n_steps=n, planets_at=planets_at,
                force=phase_force, fire=phase_fire,
                reinforce=reinforce, attack=attack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', required=True, help='Agent A .py (the one we profile)')
    ap.add_argument('--opponent', required=True, help='Agent B .py')
    ap.add_argument('--games', type=int, default=12)
    args = ap.parse_args()

    import kaggle_environments as ke

    def load(path):
        g = {}; exec(open(path).read(), g); return g['agent']
    A, B = load(args.agent), load(args.opponent)

    games = []
    for i in range(args.games):
        a_slot = i % 2  # alternate seats
        agents = [A, B] if a_slot == 0 else [B, A]
        env = ke.make('orbit_wars'); env.run(agents)
        steps = env.steps
        rew = [steps[-1][s]['reward'] for s in range(2)]
        a_won = rew[a_slot] > rew[1 - a_slot]
        games.append(analyze_game(steps, a_slot, a_won))

    wins = [g for g in games if g['won']]
    losses = [g for g in games if not g['won']]
    print(f"\n=== Behavioral profile: {args.agent.split('/')[-1]} vs {args.opponent.split('/')[-1]} ===")
    print(f"record: {len(wins)}W / {len(losses)}L  ({len(wins)/max(len(games),1):.0%})  "
          f"avg game len: {st.mean(g['n_steps'] for g in games):.0f}")

    def agg(group, label):
        if not group:
            print(f"\n{label}: (none)"); return
        tot_r = sum(g['reinforce'] for g in group)
        tot_a = sum(g['attack'] for g in group)
        rr = tot_r / max(tot_r + tot_a, 1)
        print(f"\n{label} (n={len(group)}):  reinforce_rate {rr:.2f}  (reinforce={tot_r} attack={tot_a})")
        for t in (25, 50, 75, 100):
            vals = [g['planets_at'].get(t) for g in group if t in g['planets_at']]
            if vals: print(f"  planets@{t:<3d} {st.mean(vals):.1f}")
        for ph in ('open', 'mid', 'late'):
            forces = [x for g in group for x in g['force'].get(ph, [])]
            fires = [g['fire'].get(ph, [0, 0]) for g in group]
            fs = sum(f[0] for f in fires); tot = sum(f[1] for f in fires)
            fr = fs / max(tot, 1)
            if forces:
                print(f"  {ph:<5s} fire_rate {fr:.2f}  avg_force {st.mean(forces):5.1f}  "
                      f"med_force {st.median(forces):4.0f}  launches {len(forces)}")
            else:
                print(f"  {ph:<5s} fire_rate {fr:.2f}  (no launches)")

    agg(wins, 'WINS')
    agg(losses, 'LOSSES')


if __name__ == '__main__':
    main()
