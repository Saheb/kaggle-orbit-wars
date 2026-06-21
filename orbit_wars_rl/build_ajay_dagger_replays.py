"""Generate Kaggle-format replays of our_agent vs Ajay, keeping only Ajay-wins.

DAgger round-0 seed dataset for Ajay distillation (see docs/ajay_distillation_spec.md).
We imitate the concentrator's WINNING moves, not our under-concentrator's, so only
games Ajay wins are saved. build_replay_action_bc.py --mode winner then labels Ajay's
actual projected actions from these replays.

Reuses compare_tempo_checkpoints.build_agent (checkpoint -> kaggle agent_fn) and the
kaggle_environments game loop (same as eval.py). Ajay alternates seats for diversity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "orbit_wars_rl"
for _path in (ROOT, PKG):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from kaggle_environments import make  # noqa: E402
from orbit_wars_rl.compare_tempo_checkpoints import build_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="our_agent checkpoint (policy whose states we cover).")
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--num-games", type=int, default=200, help="total games to play (not Ajay-wins).")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--out-dir", required=True, help="dir for saved Ajay-win replay JSONs.")
    ap.add_argument("--player-name", default="Saheb")
    ap.add_argument("--opponent-name", default="Ajay")
    ap.add_argument("--target-decode", action=argparse.BooleanOptionalAction, default=True,
                    help="Phase 1 checkpoints must use target decode (default on).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_fn = build_agent(args.checkpoint, target_decode=args.target_decode)
    opponent = args.opponent

    ajay_wins = 0
    games = 0
    t0 = time.time()
    while games < args.num_games:
        seed = args.seed_start + games
        ajay_seat = games % 2  # alternate seats for positional diversity
        agents = [agent_fn, opponent] if ajay_seat == 1 else [opponent, agent_fn]
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        games += 1

        final = env.steps[-1]
        rewards = [s.reward for s in final]
        ajay_r = rewards[ajay_seat] if rewards[ajay_seat] is not None else 0.0
        other_r = rewards[1 - ajay_seat] if rewards[1 - ajay_seat] is not None else 0.0
        ajay_won = ajay_r > other_r
        if ajay_won:
            ajay_wins += 1
            replay = env.toJSON()
            replay.setdefault("info", {})
            if ajay_seat == 1:
                names = [args.player_name, args.opponent_name]
            else:
                names = [args.opponent_name, args.player_name]
            replay["info"]["TeamNames"] = names
            replay["info"]["Agents"] = [{"Name": n} for n in names]
            replay["info"]["EpisodeId"] = seed
            (out_dir / f"ajay_win_seed{seed}_seat{ajay_seat}.json").write_text(json.dumps(replay))

        elapsed = time.time() - t0
        mark = "AJAY_WIN" if ajay_won else "our_win/draw"
        print(f"[{games}/{args.num_games}] seed={seed} seat={ajay_seat} {mark} "
              f"ajay_wins={ajay_wins} ({ajay_wins / games:.0%}) {elapsed / games:.1f}s/game", flush=True)

    elapsed = time.time() - t0
    print(f"\nDONE games={games} ajay_wins={ajay_wins} ({ajay_wins / games:.0%}) time={elapsed:.0f}s")
    print("\n# Build BC samples (labels Ajay's actual actions; flags mirror the proven")
    print("# Jake-unfiltered build: NO move filters so Ajay's reinforces are kept):")
    print(f"python3 orbit_wars_rl/build_replay_action_bc.py {out_dir} --mode winner "
          f"--no-save-quality-filter --reinforce-gate-min-planets 0 --reverse-edge-cooldown 0 "
          f"--samples-out {out_dir}/dagger_round0.pkl --summary-out {out_dir}/dagger_round0.json")


if __name__ == "__main__":
    main()
