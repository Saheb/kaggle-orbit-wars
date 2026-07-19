"""Coordination / overkill probe (ground-truth). Do OUR launches waste ships attacking a target
another of our fleets already captures? Measures the force-concentration wall (experiments.md #4b)
directly so we know the behavior exists BEFORE building proposal-context to fix it.

REDUNDANCY IS GROUND-TRUTH, not a garrison guess: an ATTACK launch (target not ours at launch) is
"redundant" iff the target is ACTUALLY owned by us at the fleet's arrival turn (from the replay).
That inherently accounts for enemy inbound + production + arrival timing — if we genuinely needed
two sources to crack a contested target, it isn't ours until both land, so neither is flagged.
(Only approximation: arrival turn ≈ launch + dist/ship_speed; ownership persists across turns so
small ETA error is harmless.)

Two cuts:
  - CROSS-TURN redundant-on-arrival: any attack landing on an already-ours target (the broad wall).
  - SAME-TURN multi-source coincidence: 2+ own sources hitting one target the same turn — the slice
    proposal-context (#4b) addresses (within-turn parallel decode can't coordinate it); of those,
    how many are ground-truth redundant.

    CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/coord_overkill_probe.py --seeds 6 \
        --agent-checkpoint <ckpt.pt> [--opponent opp.py | --self-play]
"""
import argparse, math, os, sys
from collections import defaultdict
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))
import eval as ev                                                   # noqa: E402
import features                                                     # noqa: E402
from ender_sizing import _checkpoint_agent, AJAY                    # noqa: E402
from features import _ship_speed_np                                 # noqa: E402
from kaggle_environments import make                                # noqa: E402


def _eta(dist, ships):
    return max(1, int(round(dist / float(_ship_speed_np(np.array([ships], dtype=np.float32))[0]))))


def _owner_at(steps, seat, turn, pid):
    turn = min(turn, len(steps) - 1)
    for pp in (steps[turn][seat].observation.get("planets") or []):
        if pp[0] == pid:
            return int(pp[1])
    return None


def collect_attacks(steps, seat):
    """Own ATTACK launches (target not ours at launch): (t0, tgt_id, ships, arrival_turn)."""
    out = []
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        p0 = steps[t - 1][seat].observation.get("planets")
        acts = steps[t][seat].action or []
        if not p0:
            continue
        byid = {p[0]: p for p in p0}
        for mv in acts:
            if not mv or len(mv) < 3:
                continue
            src = byid.get(int(mv[0]))
            if src is None:
                continue
            sent, ssh = int(mv[2]), float(src[5])
            if not (ssh > 0 and sent <= ssh):
                continue
            tgt = ev._resolve_launch_target(p0, src, float(mv[1]), sent)
            if tgt is None or int(tgt[1]) == seat:          # skip reinforcements
                continue
            dist = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
            out.append((t, int(tgt[0]), sent, t + _eta(dist, sent)))
    return out


def run_side(steps, seat, acc):
    A = collect_attacks(steps, seat)
    for (t0, tgt_id, ships, arr) in A:
        acc["atk"] += 1
        # redundant = already ours the turn BEFORE we land (captured by an EARLIER fleet, not this
        # one — excludes self-capture false positives).
        if arr - 1 > t0 and _owner_at(steps, seat, arr - 1, tgt_id) == seat:
            acc["redundant"] += 1
            acc["redundant_ships"] += ships
        acc["ships"] += ships
    # same-turn multi-source coincidence
    by_tt = defaultdict(list)
    for (t0, tgt_id, ships, arr) in A:
        by_tt[(t0, tgt_id)].append((ships, arr, tgt_id))
    for (t0, tgt_id), grp in by_tt.items():
        if len(grp) >= 2:
            acc["ms_groups"] += 1
            acc["ms_launches"] += len(grp)
            # ground-truth: of this same-turn group, later-arriving fleets landing on already-ours
            for (ships, arr, tid) in grp:
                if arr - 1 > t0 and _owner_at(steps, seat, arr - 1, tid) == seat:
                    acc["ms_redundant"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--agent-checkpoint", required=True)
    ap.add_argument("--opponent", default=AJAY)
    ap.add_argument("--self-play", action="store_true", help="opponent = the same checkpoint")
    ap.add_argument("--ablate-friendly-contest", action="store_true",
                    help="zero our friendly_contest signal (no roi-deflation for already-inbound "
                         "targets) — A/B whether the deflation reduces redundant re-launch")
    ap.add_argument("--ablate-candidate-delta", action="store_true",
                    help="zero the candidate marginal-value deltas (pairwise 30/31/34/35) — A/B "
                         "whether target choice uses the marginal signal or overkills regardless")
    args = ap.parse_args()
    features._ABLATE_FRIENDLY_CONTEST = bool(args.ablate_friendly_contest)
    features._ABLATE_CANDIDATE_DELTA = bool(args.ablate_candidate_delta)
    print(f"  friendly_contest: {'ABLATED' if args.ablate_friendly_contest else 'ON'} | "
          f"candidate_delta: {'ABLATED' if args.ablate_candidate_delta else 'ON'}")
    agent = _checkpoint_agent(args.agent_checkpoint)
    opp = agent if args.self_play else args.opponent
    opp_name = "SELF" if args.self_play else args.opponent

    acc = defaultdict(int)
    for seed in range(args.seeds):
        for my_seat in (0, 1):
            agents = [agent, opp] if my_seat == 0 else [opp, agent]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            run_side(env.steps, my_seat, acc)
            print(f"  seed={seed} seat={my_seat} atk={acc['atk']} redundant={acc['redundant']}",
                  flush=True)

    atk = acc["atk"] or 1
    print(f"\n=== coordination / overkill (ground-truth) — "
          f"{os.path.basename(args.agent_checkpoint)} vs {opp_name} ({2*args.seeds} games) ===")
    print(f"  total ATTACK launches                 {acc['atk']}")
    print(f"  ⭐ redundant-on-arrival (target already ours when we land)")
    print(f"       {acc['redundant']}  ({100*acc['redundant']/atk:.1f}% of attacks)  "
          f"[cross-turn + same-turn]")
    if acc["ships"]:
        print(f"       wasted ships {acc['redundant_ships']} / {acc['ships']} "
              f"({100*acc['redundant_ships']/acc['ships']:.1f}% of attack ships)")
    print(f"  same-turn multi-source coincidence     {acc['ms_launches']} launches "
          f"({100*acc['ms_launches']/atk:.1f}% of attacks), {acc['ms_groups']} groups")
    if acc["ms_launches"]:
        print(f"       of those, ground-truth redundant {acc['ms_redundant']} "
              f"({100*acc['ms_redundant']/acc['ms_launches']:.1f}%)")
    print("\n  read: high redundant-on-arrival = real coordination waste. If it's mostly SAME-TURN,")
    print("  proposal-context (#4b) is the fix; if mostly CROSS-TURN, the friendly-inbound signal")
    print("  (friendly_contest) isn't landing → decode/feature-usage question, not #4b.")


if __name__ == "__main__":
    main()
