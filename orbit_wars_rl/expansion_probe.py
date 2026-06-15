"""Expansion diagnostic (2026-06-14): WHY is planets@50 stuck at 6?

Two modes, both built on eval.py's own game_conversion() so planets@N are
apples-to-apples with the panel/replay reference (winner planets@16/32/50/100 = 2/6/9/10).

  --mode bc        Run a checkpoint (default: the snowball BC warmstart) vs zach over N
                   seeds; report mean planets@N (all games + won subset). Answers: does the
                   anchor we warmstart+teacher-KL to ALREADY cap expansion at ~6?

  --mode compare   Pick a seed our checkpoint WINS (scan, vs a common opponent), then replay
                   that SAME seed with Ajay and debatreya as the seat-0 player vs the same
                   opponent. Tabulate the planets-owned trajectory so you can SEE how a strong
                   planner expands on the identical board where we plateau.

Run from repo ROOT so `opponents/...` paths resolve and orbit_wars_rl is sys.path[0]:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/expansion_probe.py --mode bc   ...
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/expansion_probe.py --mode compare ...
"""
import argparse
import torch
from kaggle_environments import make

from config import Config
from model import EntityTransformer
from eval import load_checkpoint, build_agent_fn, game_conversion, _CONV_MILESTONES


def load_ckpt_agent(ckpt_path, gate, device="cpu", fire_threshold=0.5):
    """Load a checkpoint as a seat agent_fn, matching eval's mask setup (gate/floor0)."""
    cfg = Config()
    cfg.device = device
    sd, ckpt_action_decode = load_checkpoint(ckpt_path, cfg)
    target_decode = (ckpt_action_decode == "target")
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(gate)
    model.reinforce_forward_only = False
    model.reinforce_garrison_floor = 0.0
    model.sufficient_commit_factor = 0.0
    # BC checkpoints are policy-only (no value head); tolerate missing value_*/target_head.
    model.load_state_dict(sd, strict=False)
    model.eval()
    return build_agent_fn(model, torch.device(device), fire_threshold=fire_threshold, sample=False,
                          ship_bin_mode=cfg.model.ship_bin_mode, target_decode=target_decode)


def owned_traj(steps, seat):
    """Owned-planet count per step for `seat` (for the trajectory table)."""
    traj = []
    for s in steps:
        if seat >= len(s):
            traj.append(None); continue
        ps = s[seat].observation.get("planets")
        traj.append(sum(1 for p in ps if int(p[1]) == seat) if ps else 0)
    return traj


def run_game(agent0, agent1, seed):
    """Run one game; return (env, is_win_seat0, conversion_seat0)."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent0, agent1])
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    r0 = rewards[0] if rewards[0] is not None else 0.0
    best_opp = max((r for r in rewards[1:] if r is not None), default=0.0)
    return env, (r0 > best_opp), game_conversion(env.steps, 0)


def fmt_pl(conv):
    return "/".join(str(conv[f"p{ms}"]) for ms in _CONV_MILESTONES)


def mode_bc(args):
    agent = load_ckpt_agent(args.checkpoint, args.gate, fire_threshold=args.fire_threshold)
    sums = {ms: 0 for ms in _CONV_MILESTONES}
    won_sums = {ms: 0 for ms in _CONV_MILESTONES}
    wins = 0
    print(f"BC-baseline expansion: {args.checkpoint}")
    print(f"  vs {args.opponent}, {args.games} seeds, gate={args.gate}  [winner ref planets@16/32/50/100 = 2/6/9/10]\n")
    for seed in range(args.games):
        _, win, conv = run_game(agent, args.opponent, seed)
        wins += int(win)
        for ms in _CONV_MILESTONES:
            v = conv[f"p{ms}"] or 0
            sums[ms] += v
            if win:
                won_sums[ms] += v
        print(f"  seed {seed:4d}  win={int(win)}  planets@16/32/50/100 {fmt_pl(conv)}  end {conv['end_planets']}  len {conv['glen']}")
    n, nw = args.games, max(wins, 1)
    print(f"\n  WR {wins}/{args.games}")
    print(f"  MEAN planets@16/32/50/100  ALL  {'/'.join(f'{sums[ms]/n:.1f}' for ms in _CONV_MILESTONES)}")
    print(f"  MEAN planets@16/32/50/100  WON  {'/'.join(f'{won_sums[ms]/nw:.1f}' for ms in _CONV_MILESTONES)}  (winner ref 2/6/9/10)")


def mode_compare(args):
    our = load_ckpt_agent(args.checkpoint, args.gate)
    opp = args.opponent

    # 1) find a seed our checkpoint wins vs the common opponent
    seed = args.seed
    if seed is None:
        print(f"Scanning seeds 0..{args.scan_seeds-1} for an our-WIN vs {opp} ...")
        for s in range(args.scan_seeds):
            _, win, conv = run_game(our, opp, s)
            print(f"  seed {s:4d}  our win={int(win)}  planets@.. {fmt_pl(conv)}")
            if win:
                seed = s
                break
        if seed is None:
            print("No win found in scan range; pass --seed explicitly."); return
    print(f"\n=== SAME-SEED expansion comparison on seed {seed} vs {opp} ===")
    print(f"    (winner ref planets@16/32/50/100 = 2/6/9/10)\n")

    players = [("OURS", our), ("AJAY", args.ajay), ("DEB", args.deb)]
    trajs = {}
    for name, ag in players:
        env, win, conv = run_game(ag, opp, seed)
        trajs[name] = owned_traj(env.steps, 0)
        print(f"  {name:5s}  win={int(win)}  planets@16/32/50/100 {fmt_pl(conv)}  "
              f"end {conv['end_planets']}  caps {conv['captures']}  len {conv['glen']}  "
              f"g50 {conv['g50']}  inflight50 {conv['if50']}")

    # trajectory table at coarse steps
    marks = [0, 10, 16, 25, 32, 40, 50, 65, 80, 100, 130, 160, 200]
    print(f"\n  step      " + "  ".join(f"{m:>4d}" for m in marks))
    for name, _ in players:
        tr = trajs[name]
        row = []
        for m in marks:
            row.append(f"{tr[m]:>4d}" if m < len(tr) and tr[m] is not None else "   .")
        print(f"  {name:5s} owned " + "  ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bc", "compare"], required=True)
    ap.add_argument("--checkpoint", default="seed_checkpoints/bc_snowball_15global.pt",
                    help="our agent / BC ckpt (default = snowball BC warmstart)")
    ap.add_argument("--opponent", default="opponents/candidate_zach_public.py")
    ap.add_argument("--ajay", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--deb", default="opponents/candidate_debatreya_1300.py")
    ap.add_argument("--gate", type=int, default=2)
    ap.add_argument("--fire-threshold", type=float, default=0.5)
    ap.add_argument("--games", type=int, default=16, help="bc mode: number of seeds")
    ap.add_argument("--seed", type=int, default=None, help="compare mode: fixed seed (else scan for a win)")
    ap.add_argument("--scan-seeds", type=int, default=20, help="compare mode: how many seeds to scan for a win")
    args = ap.parse_args()
    (mode_bc if args.mode == "bc" else mode_compare)(args)


if __name__ == "__main__":
    main()
