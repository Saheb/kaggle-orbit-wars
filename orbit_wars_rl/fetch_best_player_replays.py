"""Fetch daily Kaggle episode slices and retain only named top-player replays.

Kaggle's Orbit Wars daily episode manifests do not include TeamNames, so this
tool downloads a score-sorted slice from each daily dataset, opens each replay,
and keeps only games containing one of the requested players.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.fetch_analyze_top_replays import (  # noqa: E402
    _api,
    _download_file,
    recent_dates,
    select_episodes,
)


def _matches(name: str, filters: list[str]) -> bool:
    lower = name.lower()
    return any(f.lower() in lower for f in filters)


def _load_replay(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) and "steps" in data else None


def _team_names_from_replay(replay: dict) -> list[str]:
    names = replay.get("info", {}).get("TeamNames")
    if names:
        return names
    agents = replay.get("info", {}).get("Agents", [])
    return [a.get("Name", f"player_{i}") for i, a in enumerate(agents)]


def _manifest_rows(manifest_path: Path) -> dict[str, dict]:
    return {r["episode_id"]: r for r in csv.DictReader(open(manifest_path))}


def _download_file_retry(api, dataset: str, fname: str, dest_dir: Path,
                         attempts: int, sleep_s: float):
    last_exc = None
    for attempt in range(max(1, attempts)):
        try:
            return _download_file(api, dataset, fname, dest_dir)
        except Exception as exc:
            last_exc = exc
            if "403" in str(exc):
                return None
            if attempt + 1 >= attempts:
                raise
            wait = sleep_s * (2 ** attempt)
            print(f"[retry] {fname}: {type(exc).__name__}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    return None


def _load_scan_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    episodes = data.get("episodes", data)
    if not isinstance(episodes, dict):
        return {}
    return {str(k): v for k, v in episodes.items() if isinstance(v, dict)}


def _save_scan_cache(path: Path, episodes: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"episodes": episodes}, indent=2, sort_keys=True))
    tmp.replace(path)


def _cache_record(ep: str, d: str, path: Path | None, row: dict,
                  names: list[str], matched: list[str], status: str) -> dict:
    return {
        "episode_id": ep,
        "date": d,
        "path": str(path) if path else "",
        "team_names": names,
        "matched_players": matched,
        "avg_score": row.get("avg_score"),
        "agent_count": row.get("agent_count"),
        "status": status,
    }


def fetch_best_players(
    dates: list[str],
    player_filters: list[str],
    n_per_day: int,
    out_dir: str,
    cache_dir: str,
    agent_count: int | None = 2,
    sort_by: str = "avg_score",
    max_kept: int = 0,
    keep_nonmatching: bool = False,
    use_scan_cache: bool = True,
    retry_attempts: int = 3,
    retry_sleep: float = 2.0,
    cache_flush_every: int = 100,
) -> dict:
    api = _api()
    out = Path(out_dir)
    cache = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    subjects: Counter = Counter()
    kept: list[dict] = []
    scan_cache_path = out / "fetch_best_player_scan_cache.json"
    scan_cache = _load_scan_cache(scan_cache_path) if use_scan_cache else {}
    cache_updates = 0

    def mark_cache_dirty() -> None:
        nonlocal cache_updates
        if not use_scan_cache:
            return
        cache_updates += 1
        if cache_updates % max(1, cache_flush_every) == 0:
            _save_scan_cache(scan_cache_path, scan_cache)

    for d in dates:
        if max_kept and stats["kept"] >= max_kept:
            break
        dataset = f"kaggle/orbit-wars-episodes-{d}"
        man = _download_file_retry(api, dataset, "manifest.csv", cache / d,
                                   retry_attempts, retry_sleep)
        if man is None:
            print(f"[skip] {d}: dataset unpublished or no manifest (403)")
            stats["missing_manifest"] += 1
            continue
        rows_by_episode = _manifest_rows(man)
        episode_ids = select_episodes(man, n_per_day, agent_count, sort_by)
        print(f"[date] {d}: scanning {len(episode_ids)} episodes sorted by {sort_by}")
        for ep in episode_ids:
            if max_kept and stats["kept"] >= max_kept:
                break
            stats["considered"] += 1
            row = rows_by_episode.get(ep, {})
            cached = scan_cache.get(ep) if use_scan_cache else None
            if cached is not None:
                status = cached.get("status")
                matched = list(cached.get("matched_players") or [])
                if status == "nonmatching" and not matched:
                    stats["cache_nonmatching"] += 1
                    continue
                if matched:
                    path = Path(cached.get("path") or out / f"{ep}.json")
                    if not path.exists():
                        got = _download_file_retry(api, dataset, f"{ep}.json", out,
                                                   retry_attempts, retry_sleep)
                        if got is None:
                            stats["download_failed"] += 1
                            continue
                        path = got
                        stats["downloaded"] += 1
                        cached["path"] = str(path)
                    stats["cache_matching"] += 1
                    stats["kept"] += 1
                    subjects.update(matched)
                    kept.append({
                        "episode_id": ep,
                        "date": cached.get("date", d),
                        "path": str(path),
                        "matched_players": matched,
                        "avg_score": cached.get("avg_score", row.get("avg_score")),
                        "agent_count": cached.get("agent_count", row.get("agent_count")),
                    })
                    continue

            path = out / f"{ep}.json"
            if not path.exists():
                got = _download_file_retry(api, dataset, f"{ep}.json", out,
                                           retry_attempts, retry_sleep)
                if got is None:
                    stats["download_failed"] += 1
                    continue
                path = got
                stats["downloaded"] += 1
            replay = _load_replay(path)
            if replay is None:
                stats["invalid_replay_json"] += 1
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            names = _team_names_from_replay(replay)
            matched = [name for name in names if _matches(name, player_filters)]
            if not matched:
                stats["nonmatching"] += 1
                if use_scan_cache:
                    scan_cache[ep] = _cache_record(ep, d, path, row, names, [], "nonmatching")
                    mark_cache_dirty()
                if not keep_nonmatching:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                continue
            stats["kept"] += 1
            subjects.update(matched)
            if use_scan_cache:
                scan_cache[ep] = _cache_record(ep, d, path, row, names, matched, "matching")
                mark_cache_dirty()
            kept.append({
                "episode_id": ep,
                "date": d,
                "path": str(path),
                "matched_players": matched,
                "avg_score": row.get("avg_score"),
                "agent_count": row.get("agent_count"),
            })
        print(f"[ok]   {d}: kept {stats['kept']} total so far")

    summary = {
        "config": {
            "dates": dates,
            "player_filters": player_filters,
            "n_per_day": n_per_day,
            "agent_count": agent_count,
            "sort_by": sort_by,
            "max_kept": max_kept,
            "keep_nonmatching": keep_nonmatching,
            "use_scan_cache": use_scan_cache,
                "retry_attempts": retry_attempts,
                "retry_sleep": retry_sleep,
                "cache_flush_every": cache_flush_every,
        },
        "stats": dict(stats),
        "subjects": dict(subjects.most_common(20)),
        "kept": kept,
        "scan_cache_path": str(scan_cache_path) if use_scan_cache else "",
    }
    summary_path = out / "fetch_best_player_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if use_scan_cache:
        _save_scan_cache(scan_cache_path, scan_cache)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--player-name", action="append", required=True,
                    help="Substring filter; repeat for Jake/Isaiah/etc.")
    ap.add_argument("--n-per-day", type=int, default=1000,
                    help="Score-sorted episodes to inspect per daily dataset.")
    ap.add_argument("--agent-count", type=int, default=2,
                    help="Filter manifest to N-player games; use 0 for all counts.")
    ap.add_argument("--sort-by", default="avg_score")
    ap.add_argument("--max-kept", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/orbit_top_player_replays")
    ap.add_argument("--cache-dir", default="/tmp/ow_manifests")
    ap.add_argument("--keep-nonmatching", action="store_true")
    ap.add_argument("--no-scan-cache", action="store_true",
                    help="Disable out-dir scan cache for nonmatching/matching episodes.")
    ap.add_argument("--retry-attempts", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0,
                    help="Initial retry sleep in seconds; retries use exponential backoff.")
    ap.add_argument("--cache-flush-every", type=int, default=100,
                    help="Persist scan cache after this many newly scanned episodes.")
    args = ap.parse_args()

    dates = args.dates or recent_dates(args.last_days)
    summary = fetch_best_players(
        dates=dates,
        player_filters=args.player_name,
        n_per_day=args.n_per_day,
        out_dir=args.out_dir,
        cache_dir=args.cache_dir,
        agent_count=None if args.agent_count == 0 else args.agent_count,
        sort_by=args.sort_by,
        max_kept=args.max_kept,
        keep_nonmatching=args.keep_nonmatching,
        use_scan_cache=not args.no_scan_cache,
        retry_attempts=args.retry_attempts,
        retry_sleep=args.retry_sleep,
        cache_flush_every=args.cache_flush_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
