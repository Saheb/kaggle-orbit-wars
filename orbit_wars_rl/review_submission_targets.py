"""Fetch a submission's loss replays and run the target audit in one command.

This wraps the verified daily-review workflow:
  1. fetch Kaggle episode metadata for a submission
  2. select loss episodes for that submission
  3. download replay JSONs with retry/backoff
  4. run audit_submission_targets on the downloaded replays

Example:
  orbit_wars_rl/.venv/bin/python orbit_wars_rl/review_submission_targets.py \
    --submission-id 53359633 \
    --checkpoint seed_checkpoints/rev31_31M_resume.pt \
    --player-name Saheb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.audit_submission_targets import (
    audit_episode,
    build_report,
    choose_player_slot,
    load_model,
    render_markdown,
)


LIST_EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def load_kaggle_creds() -> tuple[dict[str, str], str]:
    creds_path = Path("/Users/saheb/.kaggle/kaggle.json")
    token_path = Path("/Users/saheb/.kaggle/access_token")
    creds = json.loads(creds_path.read_text())
    token = token_path.read_text().strip()
    return creds, token


def fetch_submission_episodes(submission_id: int, creds: dict[str, str]) -> dict[str, Any]:
    resp = requests.post(
        LIST_EPISODES_URL,
        auth=(creds["username"], creds["key"]),
        json={"submissionId": submission_id},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "episodes" not in data:
        raise RuntimeError(f"Unexpected Kaggle response: keys={sorted(data.keys())}")
    return data


def episode_agent_for_submission(ep: dict[str, Any], submission_id: int) -> dict[str, Any] | None:
    for agent in ep.get("agents", []):
        if int(agent.get("submissionId", -1)) == submission_id:
            return agent
    return None


def select_loss_episodes(
    manifest: dict[str, Any],
    submission_id: int,
    only_two_player: bool,
    max_episodes: int | None,
) -> list[dict[str, Any]]:
    teams = {int(t["id"]): t for t in manifest.get("teams", []) if "id" in t}
    submissions = {int(s["id"]): s for s in manifest.get("submissions", []) if "id" in s}
    selected: list[dict[str, Any]] = []
    for ep in manifest.get("episodes", []):
        agents = ep.get("agents", [])
        if only_two_player and len(agents) != 2:
            continue
        ours = episode_agent_for_submission(ep, submission_id)
        if not ours:
            continue
        our_reward = float(ours.get("reward", 0.0))
        max_reward = max(float(a.get("reward", 0.0)) for a in agents) if agents else 0.0
        if our_reward >= max_reward:
            continue

        enriched = dict(ep)
        enriched["our_agent"] = ours
        opponents = []
        for agent in agents:
            if int(agent.get("submissionId", -1)) == submission_id:
                continue
            sub = submissions.get(int(agent.get("submissionId", -1)), {})
            team = teams.get(int(agent.get("teamId", -1)), {})
            opponents.append(
                {
                    "submissionId": agent.get("submissionId"),
                    "teamId": agent.get("teamId"),
                    "reward": agent.get("reward"),
                    "initialScore": agent.get("initialScore"),
                    "updatedScore": agent.get("updatedScore"),
                    "submissionName": sub.get("description") or sub.get("fileName"),
                    "teamName": team.get("teamName"),
                }
            )
        enriched["opponents"] = opponents
        selected.append(enriched)

    selected.sort(
        key=lambda ep: max((float(o.get("initialScore", -1e9)) for o in ep["opponents"]), default=-1e9),
        reverse=True,
    )
    if max_episodes is not None:
        selected = selected[:max_episodes]
    return selected


def replay_url(episode_id: int) -> str:
    return f"https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay"


def download_replay(
    episode_id: int,
    bearer_token: str,
    out_path: Path,
    attempts: int,
    backoff_sec: float,
) -> None:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(
                replay_url(episode_id),
                headers={"Authorization": f"Bearer {bearer_token}"},
                timeout=90,
            )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            return
        except Exception as exc:
            last_err = exc
            if attempt == attempts:
                break
            time.sleep(backoff_sec * attempt)
    raise RuntimeError(f"Replay download failed for episode {episode_id}: {last_err}") from last_err


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-id", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--player-name", default="Saheb")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--only-two-player", action="store_true")
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--aim-gap-deg", type=float, default=15.0)
    ap.add_argument("--step-limit", type=int)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--backoff-sec", type=float, default=1.0)
    ap.add_argument("--device", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir or f"/tmp/sub{args.submission_id}_review")
    replay_dir = out_dir / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)

    creds, token = load_kaggle_creds()
    manifest = fetch_submission_episodes(args.submission_id, creds)
    manifest_path = out_dir / "episodes.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    selected = select_loss_episodes(
        manifest,
        submission_id=args.submission_id,
        only_two_player=args.only_two_player,
        max_episodes=args.max_episodes,
    )
    selected_path = out_dir / "selected_losses.json"
    selected_path.write_text(json.dumps(selected, indent=2))
    if not selected:
        raise SystemExit("No loss episodes matched the current filters.")

    for ep in selected:
        episode_id = int(ep["id"])
        out_path = replay_dir / f"{episode_id}.json"
        if not out_path.exists():
            download_replay(
                episode_id,
                bearer_token=token,
                out_path=out_path,
                attempts=args.attempts,
                backoff_sec=args.backoff_sec,
            )

    device = torch.device(
        args.device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    model, _, action_decode = load_model(args.checkpoint, device)

    episode_reports = []
    effective_player_name = args.player_name
    replay_paths = sorted(replay_dir.glob("*.json"))
    for replay_path in replay_paths:
        replay = json.loads(replay_path.read_text())
        slot = choose_player_slot(replay, args.player_name, None)
        team_names = replay.get("info", {}).get("TeamNames") or []
        if slot < len(team_names):
            effective_player_name = str(team_names[slot])
        episode_reports.append(
            audit_episode(
                replay_path,
                replay,
                model,
                device,
                player_slot=slot,
                top_k=args.top_k,
                aim_gap_deg=args.aim_gap_deg,
                step_limit=args.step_limit,
            )
        )

    report = build_report(args.checkpoint, action_decode, effective_player_name, episode_reports)
    json_path = out_dir / "target_audit.json"
    md_path = out_dir / "target_audit.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))

    print(f"submission_id={args.submission_id}")
    print(f"checkpoint={args.checkpoint}")
    print(f"manifest={manifest_path}")
    print(f"selected_losses={selected_path}")
    print(f"replay_dir={replay_dir}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")
    agg = report["aggregate"]
    print(
        "summary: "
        f"episodes={agg['episodes']} "
        f"launches={agg['launches']} "
        f"open_space={agg['open_space_or_decode_fail']} "
        f"aim_gap={agg['aim_gap']} "
        f"policy_mismatch={agg['policy_mismatch']} "
        f"farther_than_nearest={agg['farther_than_nearest']}"
    )


if __name__ == "__main__":
    main()
