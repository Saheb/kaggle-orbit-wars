"""Download selected replays from an official daily Orbit Wars episode dataset.

The official index points at daily datasets like:
    kaggle/orbit-wars-episodes-2026-05-25

Each daily dataset includes a manifest.csv with episode_id, score metadata,
agent_count, and size. This script downloads a score-sorted slice without
pulling the full ~20 GiB day.

Example:
    python fetch_daily_replays.py \
        --dataset kaggle/orbit-wars-episodes-2026-05-25 \
        --manifest data/kaggle/orbit-wars-episodes-2026-05-25/manifest.csv \
        --n-episodes 200 \
        --agent-count 2 \
        --replay-dir data/kaggle/orbit-wars-episodes-2026-05-25/replays_top200_2p
"""

from __future__ import annotations

import argparse
import csv
import time
import zipfile
from pathlib import Path


def load_episode_ids(
    manifest_path: str,
    n_episodes: int,
    agent_count: int | None,
) -> list[int]:
    """Return top episode ids from daily manifest, preserving score order."""
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: float(r["avg_score"]), reverse=True)
    if agent_count is not None:
        rows = [r for r in rows if int(r["agent_count"]) == agent_count]

    return [int(r["episode_id"]) for r in rows[:n_episodes]]


def download_episodes(
    dataset: str,
    episode_ids: list[int],
    replay_dir: str,
    delay: float,
) -> list[Path]:
    """Download episode JSON files using Kaggle's dataset file API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    out = Path(replay_dir)
    out.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    total = len(episode_ids)
    for i, episode_id in enumerate(episode_ids, 1):
        file_name = f"{episode_id}.json"
        dest = out / file_name
        if dest.exists():
            downloaded.append(dest)
            print(f"[{i:4d}/{total}] exists      {file_name}")
            continue

        try:
            api.dataset_download_file(
                dataset,
                file_name=file_name,
                path=str(out),
                quiet=True,
            )
            zip_path = out / f"{file_name}.zip"
            if zip_path.exists() and not dest.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(out)
                zip_path.unlink()

            if dest.exists():
                downloaded.append(dest)
                print(f"[{i:4d}/{total}] downloaded  {file_name}")
            else:
                print(f"[{i:4d}/{total}] missing     {file_name}")
        except Exception as exc:
            print(f"[{i:4d}/{total}] error       {file_name}: {exc}")

        if delay > 0:
            time.sleep(delay)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Kaggle dataset slug, e.g. kaggle/orbit-wars-episodes-2026-05-25")
    parser.add_argument("--manifest", required=True,
                        help="Daily manifest.csv path")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--agent-count", type=int, default=2,
                        help="Filter by player count. Use 0 to disable.")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    agent_count = args.agent_count if args.agent_count > 0 else None
    episode_ids = load_episode_ids(args.manifest, args.n_episodes, agent_count)
    print(f"Selected {len(episode_ids)} episodes from {args.manifest}")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.replay_dir}")

    paths = download_episodes(args.dataset, episode_ids, args.replay_dir, args.delay)
    total_mb = sum(p.stat().st_size for p in paths if p.exists()) / 1e6
    print()
    print(f"Downloaded/available: {len(paths)}/{len(episode_ids)} files ({total_mb:.1f} MB)")


if __name__ == "__main__":
    main()
