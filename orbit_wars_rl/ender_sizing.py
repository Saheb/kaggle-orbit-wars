"""What does a top-10 agent actually SEND? — Ender's launch-fraction distribution.

The binary experiment replaced the ship head with a resolver that sends ALL available ships at a
non-owned target. docs/training.md rejected a deterministic middle (projected-hold) as the
execution contract, but that rejection did not establish that all-in is right — only that a
no-new-launch projection is the wrong way to pick the middle. This measures the question
directly: per launch, ships_sent / source_garrison_at_launch, from real Ender play.

Read it as: mass at 1.0 = all-in; mass strictly between 0 and 1 = the middle commitment our
action space cannot currently express. Split by attack vs reinforce, because reinforce sizing is
a different decision, and by phase.

    CUDA_VISIBLE_DEVICES="" python orbit_wars_rl/ender_sizing.py --seeds 8
"""
import argparse
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
os.chdir(_REPO)

import eval as ev                        # noqa: E402
from kaggle_environments import make     # noqa: E402

ENDER = "opponents/candidate_ender.py"
AJAY = "opponents/candidate_ajay_1200.py"
BINS = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0001]
LABELS = ["<5%", "5-25%", "25-50%", "50-75%", "75-95%", ">=95% (all-in)"]


def _bin(frac):
    for i in range(len(BINS) - 1):
        if BINS[i] <= frac < BINS[i + 1]:
            return i
    return len(LABELS) - 1


def collect(steps, seat, atk_hist, reinf_hist, phase_hist, raw):
    ev._CONV_ANGVEL = 0.0
    for s in steps:
        if seat < len(s):
            av = s[seat].observation.get("angular_velocity")
            if av is not None:
                ev._CONV_ANGVEL = float(av)
                break
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
            frac = sent / ssh
            raw.append(frac)
            tgt = ev._resolve_launch_target(p0, src, float(mv[1]), sent)
            b = _bin(frac)
            if tgt is not None and int(tgt[1]) == seat:
                reinf_hist[b] += 1
            else:
                atk_hist[b] += 1
                phase_hist[0 if t < 50 else (1 if t < 100 else 2)][b] += 1


def _show(name, hist):
    tot = sum(hist.values())
    if not tot:
        print(f"  {name}: no launches")
        return
    print(f"  {name} (n={tot})")
    for i, lab in enumerate(LABELS):
        n = hist[i]
        bar = "#" * int(round(40 * n / tot))
        print(f"    {lab:>16}  {100*n/tot:5.1f}%  {bar}")


def _checkpoint_agent(path, gate_min_planets=2, garrison_floor=0.0):
    """Load one of OUR checkpoints as an agent_fn, so its sizing is directly comparable to a
    path-agent's on the same histogram.

    ⚠ build_agent_fn reads allow_reinforce and the discipline masks OFF THE MODEL OBJECT
    (eval.py:391 / :1560), NOT from its own kwargs. Forgetting to set them silently disables
    reinforcement and makes any reinforce-related measurement garbage. Defaults mirror the
    eval/watcher masks (gate2/floor0).
    """
    import torch
    from config import Config
    from model import EntityTransformer
    cfg = Config(); cfg.device = "cpu"
    device = torch.device("cpu")
    sd, action_decode = ev.load_checkpoint(path, cfg)
    model = EntityTransformer(cfg.model).to(device)
    model.load_state_dict(sd)
    model.eval()
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(gate_min_planets)
    model.reinforce_forward_only = bool(getattr(cfg.model, "reinforce_forward_only", False))
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.reinforce_garrison_floor = float(garrison_floor)
    model.sufficient_commit_factor = float(getattr(cfg.model, "sufficient_commit_factor", 0.0))
    print(f"  [checkpoint agent] allow_reinforce={model.allow_reinforce} "
          f"gate>={model.reinforce_gate_min_planets} floor={model.reinforce_garrison_floor} "
          f"cooldown={model.reverse_edge_cooldown} mode={cfg.model.ship_bin_mode}")
    return ev.build_agent_fn(model, device, fire_threshold=0.5,
                             ship_bin_mode=cfg.model.ship_bin_mode,
                             target_decode=(action_decode == "target"), num_players=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--agent", default=ENDER)
    ap.add_argument("--opponent", default=AJAY)
    ap.add_argument("--agent-checkpoint", default=None,
                    help="measure one of OUR checkpoints instead of a path agent")
    args = ap.parse_args()
    agent = _checkpoint_agent(args.agent_checkpoint) if args.agent_checkpoint else args.agent

    atk, reinf = Counter(), Counter()
    phase = [Counter(), Counter(), Counter()]
    raw = []
    for seed in range(args.seeds):
        for my_seat in (0, 1):
            agents = ([agent, args.opponent] if my_seat == 0
                      else [args.opponent, agent])
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            collect(env.steps, my_seat, atk, reinf, phase, raw)
            print(f"  seed={seed} seat={my_seat} launches={len(raw)}", flush=True)

    who = args.agent_checkpoint or args.agent
    print(f"\n=== {who} launch sizing vs {args.opponent} "
          f"({2*args.seeds} games, {len(raw)} launches) ===")
    _show("ATTACK launches (target not owned)", atk)
    _show("REINFORCE launches (own target)", reinf)
    for i, nm in enumerate(("attack, opening <50", "attack, mid 50-100", "attack, late >=100")):
        _show(nm, phase[i])
    if raw:
        allin = sum(1 for f in raw if f >= 0.95) / len(raw)
        middle = sum(1 for f in raw if 0.05 <= f < 0.95) / len(raw)
        print(f"\n  all-in (>=95%): {100*allin:.1f}%   middle (5-95%): {100*middle:.1f}%")
        print("  → middle mass is the commitment our binary all-in resolver cannot express.")


if __name__ == "__main__":
    main()
