"""Compare opening target-tempo metrics across checkpoints on fixed seeds.

This script:
  1. runs each checkpoint against a fixed opponent on fixed seeds
  2. saves replay JSONs locally
  3. audits the opening launches with audit_submission_targets helpers
  4. prints and saves aggregate tempo metrics per checkpoint

Primary use: compare PPO checkpoints on the same Ajay regression slice without
having to manually generate replays and compute audit aggregates each time.

Example:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/compare_tempo_checkpoints.py \
    --checkpoints \
      gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_1572864_rev33_20260604_144227.pt \
      gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_3145728_rev33_20260604_144227.pt \
      gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_4194304_rev33_20260604_144227.pt \
      gpu_run_artifacts/jarvis_rev33/checkpoints/torch_step_5242880_rev33_20260604_144227.pt \
    --opponent opponents/candidate_ajay_1200.py \
    --target-decode \
    --step-limit 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_environments import make

from orbit_wars_rl.audit_submission_targets import (
    audit_episode,
    build_report,
    load_model,
    render_markdown,
)
from orbit_wars_rl.config import Config
from orbit_wars_rl.eval import build_agent_fn, load_checkpoint
from orbit_wars_rl.model import EntityTransformer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--opponent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 3, 5, 9])
    ap.add_argument("--player-slot", type=int, default=0)
    ap.add_argument("--player-name", default="Saheb")
    ap.add_argument("--opponent-name", default="Ajay")
    ap.add_argument("--step-limit", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--aim-gap-deg", type=float, default=15.0)
    ap.add_argument("--target-decode", action="store_true")
    ap.add_argument("--output-dir", default="/tmp/tempo_checkpoint_compare")
    ap.add_argument("--device", default="")
    return ap.parse_args()


def sanitize_label(checkpoint: str) -> str:
    return Path(checkpoint).stem


def build_agent(checkpoint: str, target_decode: bool):
    cfg = Config()
    sd, _ = load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device).eval()
    agent_fn = build_agent_fn(
        model,
        device,
        ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=target_decode,
    )
    return agent_fn


def generate_replays(
    checkpoint: str,
    opponent: str,
    seeds: list[int],
    player_slot: int,
    target_decode: bool,
    replay_dir: Path,
    player_name: str,
    opponent_name: str,
) -> list[Path]:
    replay_dir.mkdir(parents=True, exist_ok=True)
    agent_fn = build_agent(checkpoint, target_decode=target_decode)
    replay_paths: list[Path] = []
    for seed in seeds:
        agents = [agent_fn, opponent] if player_slot == 0 else [opponent, agent_fn]
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        replay = env.toJSON()
        replay.setdefault("info", {})
        if player_slot == 0:
            replay["info"]["TeamNames"] = [player_name, opponent_name]
            replay["info"]["Agents"] = [{"Name": player_name}, {"Name": opponent_name}]
        else:
            replay["info"]["TeamNames"] = [opponent_name, player_name]
            replay["info"]["Agents"] = [{"Name": opponent_name}, {"Name": player_name}]
        replay["info"]["EpisodeId"] = seed
        out_path = replay_dir / f"{seed}.json"
        out_path.write_text(json.dumps(replay))
        replay_paths.append(out_path)
    return replay_paths


def initial_planet_count(replay: dict[str, Any], player_slot: int) -> int:
    for step in replay.get("steps", []):
        agent = step[player_slot]
        obs = agent.get("observation")
        if not obs:
            continue
        return sum(1 for p in obs["planets"] if int(p[1]) == player_slot)
    return 0


def first_capture_step(replay: dict[str, Any], player_slot: int) -> int | None:
    baseline = initial_planet_count(replay, player_slot)
    for t, step in enumerate(replay.get("steps", [])):
        agent = step[player_slot]
        obs = agent.get("observation")
        if not obs:
            continue
        owned = sum(1 for p in obs["planets"] if int(p[1]) == player_slot)
        if owned > baseline:
            return t
    return None


def first_launch_step(replay: dict[str, Any], player_slot: int) -> int | None:
    for t, step in enumerate(replay.get("steps", [])):
        actions = step[player_slot].get("action") or []
        if actions:
            return t
    return None


def launches_before_step(replay: dict[str, Any], player_slot: int, limit_step: int | None) -> tuple[int, int]:
    launches = 0
    ships = 0
    cutoff = limit_step if limit_step is not None else len(replay.get("steps", []))
    for t, step in enumerate(replay.get("steps", [])):
        if t >= cutoff:
            break
        actions = step[player_slot].get("action") or []
        launches += len(actions)
        ships += sum(int(a[2]) for a in actions if len(a) >= 3)
    return launches, ships


def opening_conversion_metrics(replay: dict[str, Any], player_slot: int) -> dict[str, Any]:
    first_cap = first_capture_step(replay, player_slot)
    first_launch = first_launch_step(replay, player_slot)
    launches_pre_cap, ships_pre_cap = launches_before_step(replay, player_slot, first_cap)
    return {
        "first_launch_step": first_launch,
        "first_capture_step": first_cap,
        "launches_before_first_capture": launches_pre_cap,
        "ships_before_first_capture": ships_pre_cap,
        "mean_ships_per_launch_before_first_capture":
            (ships_pre_cap / launches_pre_cap if launches_pre_cap else None),
    }


def summarize_report(report: dict[str, Any], replay_dir: Path) -> dict[str, Any]:
    launches = 0
    farther = 0
    eta_gaps: list[float] = []
    dist_gaps: list[float] = []
    tempo_match = 0
    nearest_match = 0
    first_caps: list[int] = []
    first_launches: list[int] = []
    us_ships_pre_cap: list[int] = []
    opp_caps: list[int] = []
    opp_launches: list[int] = []
    opp_ships_pre_cap: list[int] = []
    per_seed: list[dict[str, Any]] = []

    for ep in report["episodes"]:
        replay = json.loads(Path(ep["replay_path"]).read_text())
        us_slot = int(ep["player_slot"])
        opp_slot = 1 - us_slot
        us_conv = opening_conversion_metrics(replay, us_slot)
        opp_conv = opening_conversion_metrics(replay, opp_slot)
        first_caps.append(999 if us_conv["first_capture_step"] is None else us_conv["first_capture_step"])
        first_launches.append(999 if us_conv["first_launch_step"] is None else us_conv["first_launch_step"])
        us_ships_pre_cap.append(int(us_conv["ships_before_first_capture"]))
        opp_caps.append(999 if opp_conv["first_capture_step"] is None else opp_conv["first_capture_step"])
        opp_launches.append(999 if opp_conv["first_launch_step"] is None else opp_conv["first_launch_step"])
        opp_ships_pre_cap.append(int(opp_conv["ships_before_first_capture"]))
        per_seed.append(
            {
                "episode_id": ep["episode_id"],
                "us": us_conv,
                "opp": opp_conv,
                "launches": ep["summary"]["launches"],
                "farther_than_nearest": ep["summary"]["farther_than_nearest"],
            }
        )
        for action in ep["actions"]:
            launches += 1
            chosen = action.get("decoded_target")
            nearest = action.get("nearest_target")
            tempo = action.get("tempo_target")
            if nearest and chosen and int(chosen["planet_id"]) != int(nearest["planet_id"]):
                farther += 1
            if action.get("eta_gap_vs_nearest") is not None:
                eta_gaps.append(float(action["eta_gap_vs_nearest"]))
            if action.get("distance_gap_vs_nearest") is not None:
                dist_gaps.append(float(action["distance_gap_vs_nearest"]))
            if chosen and tempo and int(chosen["planet_id"]) == int(tempo["planet_id"]):
                tempo_match += 1
            if chosen and nearest and int(chosen["planet_id"]) == int(nearest["planet_id"]):
                nearest_match += 1

    return {
        "replay_dir": str(replay_dir),
        "launches": launches,
        "farther_than_nearest_rate": farther / launches if launches else None,
        "mean_eta_gap_vs_nearest": statistics.mean(eta_gaps) if eta_gaps else None,
        "mean_distance_gap_vs_nearest": statistics.mean(dist_gaps) if dist_gaps else None,
        "tempo_match_rate": tempo_match / launches if launches else None,
        "nearest_match_rate": nearest_match / launches if launches else None,
        "mean_first_launch_step": statistics.mean(first_launches) if first_launches else None,
        "mean_first_capture_step": statistics.mean(first_caps) if first_caps else None,
        "mean_ships_before_first_capture": statistics.mean(us_ships_pre_cap) if us_ships_pre_cap else None,
        "mean_opp_first_launch_step": statistics.mean(opp_launches) if opp_launches else None,
        "mean_opp_first_capture_step": statistics.mean(opp_caps) if opp_caps else None,
        "mean_opp_ships_before_first_capture": statistics.mean(opp_ships_pre_cap) if opp_ships_pre_cap else None,
        "first_capture_steps": per_seed,
        "invalid_raw_argmax": report["aggregate"]["invalid_raw_argmax"],
        "dropped_slot_actions": report["aggregate"]["dropped_slot_actions"],
    }


def render_summary_md(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Tempo Checkpoint Comparison",
        "",
        "| checkpoint | launches | farther_rate | mean_eta_gap | tempo_match | us_first_cap | opp_first_cap | us_precap_ships | opp_precap_ships | invalid_raw_argmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {launches} | {farther:.3f} | {eta:.3f} | {tempo:.3f} | {first_cap:.2f} | {opp_cap:.2f} | {us_ships:.1f} | {opp_ships:.1f} | {invalid} |".format(
                label=row["label"],
                launches=row["summary"]["launches"],
                farther=row["summary"]["farther_than_nearest_rate"] or 0.0,
                eta=row["summary"]["mean_eta_gap_vs_nearest"] or 0.0,
                tempo=row["summary"]["tempo_match_rate"] or 0.0,
                first_cap=row["summary"]["mean_first_capture_step"] or 0.0,
                opp_cap=row["summary"]["mean_opp_first_capture_step"] or 0.0,
                us_ships=row["summary"]["mean_ships_before_first_capture"] or 0.0,
                opp_ships=row["summary"]["mean_opp_ships_before_first_capture"] or 0.0,
                invalid=row["summary"]["invalid_raw_argmax"],
            )
        )
    lines.append("")
    for row in rows:
        lines.append(f"## {row['label']}")
        lines.append("")
        lines.append(f"- checkpoint: `{row['checkpoint']}`")
        lines.append(f"- replay dir: `{row['summary']['replay_dir']}`")
        lines.append(f"- mean distance gap vs nearest: `{row['summary']['mean_distance_gap_vs_nearest']:.3f}`")
        lines.append(f"- nearest match rate: `{row['summary']['nearest_match_rate']:.3f}`")
        lines.append(f"- mean first launch step (us): `{row['summary']['mean_first_launch_step']:.2f}`")
        lines.append(f"- mean first launch step (opp): `{row['summary']['mean_opp_first_launch_step']:.2f}`")
        lines.append(f"- mean ships before first capture (us): `{row['summary']['mean_ships_before_first_capture']:.1f}`")
        lines.append(f"- mean ships before first capture (opp): `{row['summary']['mean_opp_ships_before_first_capture']:.1f}`")
        lines.append("- first capture steps:")
        for per_seed in row["summary"]["first_capture_steps"]:
            lines.append(
                f"  - seed {per_seed['episode_id']}: "
                f"us_launch={per_seed['us']['first_launch_step']} us_cap={per_seed['us']['first_capture_step']} "
                f"us_precap_ships={per_seed['us']['ships_before_first_capture']} "
                f"opp_launch={per_seed['opp']['first_launch_step']} opp_cap={per_seed['opp']['first_capture_step']} "
                f"opp_precap_ships={per_seed['opp']['ships_before_first_capture']} "
                f"launches={per_seed['launches']} farther={per_seed['farther_than_nearest']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )

    rows: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        label = sanitize_label(checkpoint)
        ckpt_dir = output_dir / label
        replay_dir = ckpt_dir / "replays"
        replay_paths = generate_replays(
            checkpoint=checkpoint,
            opponent=args.opponent,
            seeds=list(args.seeds),
            player_slot=args.player_slot,
            target_decode=args.target_decode,
            replay_dir=replay_dir,
            player_name=args.player_name,
            opponent_name=args.opponent_name,
        )
        model, _, action_decode = load_model(checkpoint, device)
        episode_reports = []
        for replay_path in replay_paths:
            replay = json.loads(replay_path.read_text())
            episode_reports.append(
                audit_episode(
                    replay_path,
                    replay,
                    model,
                    device,
                    player_slot=args.player_slot,
                    top_k=args.top_k,
                    aim_gap_deg=args.aim_gap_deg,
                    step_limit=args.step_limit,
                )
            )
        report = build_report(checkpoint, action_decode, args.player_name, episode_reports)
        (ckpt_dir / "audit.json").write_text(json.dumps(report, indent=2))
        (ckpt_dir / "audit.md").write_text(render_markdown(report))
        summary = summarize_report(report, replay_dir)
        rows.append(
            {
                "label": label,
                "checkpoint": checkpoint,
                "summary": summary,
            }
        )

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    summary_json.write_text(json.dumps(rows, indent=2))
    summary_md.write_text(render_summary_md(rows))

    for row in rows:
        s = row["summary"]
        print(
            f"{row['label']}: launches={s['launches']} "
            f"farther_rate={s['farther_than_nearest_rate']:.3f} "
            f"eta_gap={s['mean_eta_gap_vs_nearest']:.3f} "
            f"tempo_match={s['tempo_match_rate']:.3f} "
            f"us_first_cap={s['mean_first_capture_step']:.2f} "
            f"opp_first_cap={s['mean_opp_first_capture_step']:.2f} "
            f"us_precap_ships={s['mean_ships_before_first_capture']:.1f} "
            f"opp_precap_ships={s['mean_opp_ships_before_first_capture']:.1f} "
            f"invalid_raw_argmax={s['invalid_raw_argmax']}"
        )
    print(f"saved summary -> {summary_json}")
    print(f"saved markdown -> {summary_md}")


if __name__ == "__main__":
    main()
