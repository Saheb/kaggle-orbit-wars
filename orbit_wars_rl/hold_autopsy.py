"""Lost-capture autopsy (2026-06-15): WHY can't we hold captures vs a strong planner?

For every planet we CAPTURE then LOSE, classify the loss mode using the state at the step of
loss (decision-time obs t-1, same geometry eval uses):
  ABANDONED   garrison at loss <= 2  -> we captured and moved the army on, left it undefended
  OUT-MASSED  garrison > 2 but enemy inbound fleet > our garrison -> under-garrisoned vs the threat
  TOO-LATE    we had reinforcement inbound but it didn't arrive in time (reactive/too-slow)
  OTHER       residual (e.g. production/combat edge cases)
Also reports garrison_at_capture (did we capture with surplus, or just-enough -> can't hold) and
median hold-steps. Reuses eval.game_conversion's cap/loss detection + eval._friendly_inbound.

Run from repo ROOT:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/hold_autopsy.py --checkpoint <ck> \
      --opponent opponents/candidate_debatreya_1300.py --games 16 --gate 2
"""
import argparse
import statistics
from collections import Counter

from kaggle_environments import make

from eval import _friendly_inbound
from expansion_probe import load_ckpt_agent


def autopsy_game(steps, seat, enemy):
    """Classify every captured-then-lost planet in one game. Returns (events, total_caps)."""
    prev = {}
    cap_step, cap_garr = {}, {}
    events = []          # (mode, hold, garr_at_cap, garr_at_loss, enemy_in, our_reinf)
    total_caps = 0
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            continue
        p0 = steps[t - 1][seat].observation.get("planets")
        p1 = steps[t][seat].observation.get("planets")
        if not p1:
            continue
        byid0 = {p[0]: p for p in (p0 or [])}
        f0 = steps[t - 1][seat].observation.get("fleets") or []
        for p in p1:
            pid, own = p[0], int(p[1])
            was = prev.get(pid)
            if was is not None and was != seat and own == seat:
                total_caps += 1
                cap_step[pid] = t
                cap_garr[pid] = p[5]                       # garrison right after capture
            elif was == seat and own != seat and pid in cap_step:
                tgt0 = byid0.get(pid)                       # the planet as it was just before loss
                garr_loss = tgt0[5] if tgt0 else 0
                enemy_in = _friendly_inbound(f0, tgt0, enemy) if tgt0 else 0.0
                our_reinf = _friendly_inbound(f0, tgt0, seat) if tgt0 else 0.0
                if garr_loss <= 2:
                    mode = "ABANDONED"
                elif enemy_in > garr_loss:
                    mode = "OUT-MASSED"
                elif our_reinf > 0:
                    mode = "TOO-LATE"
                else:
                    mode = "OTHER"
                events.append((mode, t - cap_step[pid], cap_garr.get(pid, 0),
                               garr_loss, enemy_in, our_reinf))
                del cap_step[pid]
            prev[pid] = own
    return events, total_caps


_MODES = ("ABANDONED", "OUT-MASSED", "TOO-LATE", "OTHER")


def _report(tag, label, events, total_caps):
    """Loss-mode breakdown for ONE seat's captured-then-lost planets.
    Also emits a machine-readable `RAW <tag> ...` line so parallel seed-shards aggregate."""
    n = len(events)
    modes = Counter(e[0] for e in events)
    print(f"\n    --- {label}: captures {total_caps} | lost-captures {n} | peel-rate {n/max(total_caps,1):.2f}")
    if n:
        parts = [f"{m} {100*modes.get(m,0)/n:.0f}%" for m in _MODES]
        print("        loss-mode:  " + "   ".join(parts))
        gloss = [e[3] for e in events]; ein = [e[4] for e in events]
        print(f"        garrison@loss {statistics.median(gloss):.0f}  vs  enemy-inbound@loss {statistics.median(ein):.0f}")
    rc = " ".join(f"{m}={modes.get(m,0)}" for m in _MODES)
    print(f"    RAW {tag} caps={total_caps} lost={n} {rc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_debatreya_1300.py")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    agent = load_ckpt_agent(args.checkpoint, args.gate)
    our_ev, our_caps, opp_ev, opp_caps, wins = [], 0, [], 0, 0
    for s in range(args.seed0, args.seed0 + args.games):
        env = make("orbit_wars", configuration={"seed": s}, debug=False)
        env.run([agent, args.opponent])
        steps = env.steps
        last = steps[-1][0].observation.get("planets") or []
        us = sum(1 for p in last if int(p[1]) == 0)
        opp = sum(1 for p in last if int(p[1]) == 1)
        wins += int(us > opp)
        ev0, c0 = autopsy_game(steps, seat=0, enemy=1)   # OUR lost captures
        ev1, c1 = autopsy_game(steps, seat=1, enemy=0)   # OPPONENT's lost captures
        our_ev += ev0; our_caps += c0
        opp_ev += ev1; opp_caps += c1

    print(f"\n=== HOLD AUTOPSY (both seats): {args.checkpoint}")
    print(f"    vs {args.opponent}  |  {args.games} games, gate{args.gate}  |  seat-0 wins {wins}/{args.games}")
    print("    [OUT-MASSED% = of a seat's OWN lost captures, fraction lost to enemy-inbound > garrison]")
    print("    [GHOST test: if OPPONENT's OUT-MASSED% ~ OURS, out-massing is just how captures get lost. Asymmetric = real wall.]")
    _report("US", "US (seat 0) lost captures", our_ev, our_caps)
    _report("OPP", "OPPONENT (seat 1) lost captures", opp_ev, opp_caps)


if __name__ == "__main__":
    main()
