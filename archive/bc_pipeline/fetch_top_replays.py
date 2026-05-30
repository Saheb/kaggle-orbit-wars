"""Download raw replay JSONs for top-agent wins and build BC training data.

Uses the Kaggle API to selectively download specific episode files from
the daily episode datasets, guided by the parquet metadata to target only
games where top agents won.

Usage:
    # Download 100 top-agent winner games and build BC pkl
    python fetch_top_replays.py --n-episodes 100 --output bc_top_agents.pkl

    # Just download JSONs (skip BC extraction)
    python fetch_top_replays.py --n-episodes 100 --download-only

    # Re-process already-downloaded JSONs
    python fetch_top_replays.py --process-only --replay-dir top_agent_replays/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset → episode ID range mapping (probed empirically)
# Each tuple: (dataset_slug, min_episode_id, max_episode_id)
# ---------------------------------------------------------------------------
DAILY_DATASET_RANGES = [
    ("kaggle/orbit-wars-episodes-2026-05-20", 77_135_463, 77_249_936),
    ("kaggle/orbit-wars-episodes-2026-05-19", 77_020_133, 77_135_462),
    ("kaggle/orbit-wars-episodes-2026-05-18", 76_906_186, 77_020_132),
    ("kaggle/orbit-wars-episodes-2026-05-17", 76_792_800, 76_906_185),
    ("kaggle/orbit-wars-episodes-2026-05-16", 76_680_000, 76_792_799),
]

TOP_AGENTS = [
    "Vadasz",
    "bowwowforeach",
    "typeIIIfairy",
    "Isaiah @ Tufa Labs",
    "Jake Will",
    "3Comets",
    "213tubo",
    "Shun_PI",
    "kovi",
]


def load_target_episodes(parquet_path: str, n: int, agents: list[str]) -> dict[int, str]:
    """Return {episode_id: dataset_slug} for the n most-recent top-agent wins."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas + pyarrow required: pip install pandas pyarrow")

    pe = pd.read_parquet(parquet_path)
    winners = pe[(pe["name"].isin(agents)) & (pe["is_winner"] == 1)].copy()
    winners = winners.sort_values("episode_id", ascending=False)

    mapping: dict[int, str] = {}
    for _, row in winners.iterrows():
        eid = int(row["episode_id"])
        slug = _slug_for_episode(eid)
        if slug:
            mapping[eid] = slug
        if len(mapping) >= n:
            break

    print(f"Found {len(mapping)} target episodes across {len(set(mapping.values()))} datasets")
    return mapping


def _slug_for_episode(episode_id: int) -> str | None:
    for slug, lo, hi in DAILY_DATASET_RANGES:
        if lo <= episode_id <= hi:
            return slug
    return None  # outside known range


def download_episodes(
    episode_to_dataset: dict[int, str],
    output_dir: str,
    delay: float = 0.5,
) -> list[Path]:
    """Download specific episode JSON files using the Kaggle Python API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    total = len(episode_to_dataset)

    for i, (eid, slug) in enumerate(episode_to_dataset.items()):
        fname = f"{eid}.json"
        dest = output_path / fname
        if dest.exists():
            downloaded.append(dest)
            print(f"  [{i+1}/{total}] {fname} already exists, skipping")
            continue

        owner_slug, dataset_slug = slug.split("/", 1)
        try:
            api.dataset_download_file(
                owner_slug + "/" + dataset_slug,
                file_name=fname,
                path=str(output_path),
                quiet=True,
            )
            # Kaggle downloads as <fname>.zip if over threshold; unzip if needed
            zip_path = dest.with_suffix(".json.zip")
            if zip_path.exists() and not dest.exists():
                import zipfile
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(output_path)
                zip_path.unlink()

            if dest.exists():
                downloaded.append(dest)
                print(f"  [{i+1}/{total}] Downloaded {fname}")
            else:
                print(f"  [{i+1}/{total}] WARNING: {fname} not found after download")

        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR downloading {fname}: {e}")

        if delay > 0:
            time.sleep(delay)

    print(f"\nDownloaded {len(downloaded)}/{total} files to {output_dir}")
    return downloaded


def build_bc_data(replay_files: list[Path], output_pkl: str) -> None:
    """Process replay JSONs → BC training samples and save as pickle."""
    from replay_to_bc import extract_bc_samples_from_replay

    all_samples = []
    for i, path in enumerate(replay_files):
        print(f"  Processing {path.name} ({i+1}/{len(replay_files)})...", end="", flush=True)
        try:
            samples = extract_bc_samples_from_replay(str(path))
            all_samples.extend(samples)
            print(f" {len(samples)} samples (total: {len(all_samples)})")
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\nTotal samples: {len(all_samples)}")
    with open(output_pkl, "wb") as f:
        pickle.dump(all_samples, f)
    size_mb = os.path.getsize(output_pkl) / 1e6
    print(f"Saved → {output_pkl}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of top-agent winner games to download")
    parser.add_argument("--agents", type=str, default=",".join(TOP_AGENTS),
                        help="Comma-separated agent names to include")
    parser.add_argument("--parquet", type=str,
                        default="/tmp/orbit_parquet/player_episodes.parquet",
                        help="Path to player_episodes.parquet for episode metadata")
    parser.add_argument("--replay-dir", type=str, default="top_agent_replays",
                        help="Directory to store downloaded JSON files")
    parser.add_argument("--output", type=str, default="bc_top_agents.pkl",
                        help="Output pickle file for BC training samples")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download JSONs, skip BC extraction")
    parser.add_argument("--process-only", action="store_true",
                        help="Only process already-downloaded JSONs, skip download")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Seconds between API calls (avoid rate-limiting)")
    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(",")]

    if not args.process_only:
        if not os.path.exists(args.parquet):
            print(f"ERROR: parquet file not found: {args.parquet}")
            print("Download it first:")
            print("  kaggle datasets download nbridelancetb/orbit-wars-replay-parquet "
                  "-f player_episodes.parquet -p /tmp/orbit_parquet")
            exit(1)

        episode_to_dataset = load_target_episodes(args.parquet, args.n_episodes, agents)
        replay_files = download_episodes(episode_to_dataset, args.replay_dir, delay=args.delay)
    else:
        replay_dir = Path(args.replay_dir)
        replay_files = sorted(replay_dir.glob("*.json"))
        print(f"Found {len(replay_files)} JSON files in {args.replay_dir}")

    if not args.download_only:
        build_bc_data(replay_files, args.output)
