"""Fetch top-scoring games from Kaggle's daily Orbit Wars episode datasets and
analyze top-player behaviour (reinforce-vs-economy ramp, reinforce geometry,
ship commitment, target selectivity).

Kaggle publishes one dataset per day: `kaggle/orbit-wars-episodes-YYYY-MM-DD`, each
with a `manifest.csv` (episode_id, avg_score, agent_count, ...) and the per-episode
`<id>.json` replays. This tool downloads a score-sorted slice (NOT the full ~20GB/day)
and runs the behavioural characterisation used to design the reinforcement lever.

TIMING (important): the action recorded at steps[t] was decided on the observation
at steps[t-1] (verified: 100% of launches legal against t-1 obs, 9.9% against t).
Every action<->observation join below pairs acts=steps[t] with obs=steps[t-1].

Replay schemas: planet [id, owner, x, y, radius, ships, production]; owner -1 = neutral,
owner ids are GLOBAL. fleet [id, owner, x, y, angle, from_planet_id, ships].
action [from_planet_id, angle, num_ships].

Examples
--------
  # Last 3 days, top 60 two-player games each, download + analyze the winners:
  python orbit_wars_rl/fetch_analyze_top_replays.py --last-days 3 --analyze

  # Specific dates, 100 games/day, into a chosen dir:
  python orbit_wars_rl/fetch_analyze_top_replays.py \
      --dates 2026-06-07 2026-06-08 --n-per-day 100 --out-dir /tmp/val

  # Re-analyze an already-downloaded dir for one player (no fetching):
  python orbit_wars_rl/fetch_analyze_top_replays.py --no-download --analyze \
      --out-dir /tmp/val --player "Isaiah @ Tufa Labs"

  # Analyze winners, excluding games won by Isaiah (generalisation check):
  python orbit_wars_rl/fetch_analyze_top_replays.py --no-download --analyze \
      --out-dir /tmp/val --exclude "Isaiah @ Tufa Labs"

Notes
-----
  * Today's dataset is usually published ~00:10 UTC the next day; an unpublished date
    returns 403 and is skipped (with a warning), so --last-days is safe to run anytime.
  * Requires Kaggle API creds (~/.kaggle/kaggle.json), same as the rest of the repo.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, statistics as st, sys, zipfile
from collections import defaultdict
from datetime import date as _date, timedelta
from pathlib import Path


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def recent_dates(n: int) -> list[str]:
    """The n most recent dates (yesterday backwards — today is usually unpublished)."""
    today = _date.today()
    return [(today - timedelta(days=i)).isoformat() for i in range(1, n + 1)]


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    return api


def _download_file(api, dataset: str, fname: str, dest_dir: Path) -> Path | None:
    """Download one file from a dataset; unzip if Kaggle wrapped it. Returns path or None."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_file(dataset, fname, path=str(dest_dir), quiet=True)
    except Exception as e:
        if "403" in str(e):
            return None
        raise
    out = dest_dir / fname
    z = dest_dir / f"{fname}.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        z.unlink()
    return out if out.exists() else None


def select_episodes(manifest_path: Path, n: int, agent_count: int | None, sort_by: str) -> list[str]:
    rows = list(csv.DictReader(open(manifest_path)))
    if agent_count is not None:
        rows = [r for r in rows if int(r["agent_count"]) == agent_count]
    rows.sort(key=lambda r: float(r[sort_by]), reverse=True)
    return [r["episode_id"] for r in rows[:n]]


def fetch(dates: list[str], n_per_day: int, agent_count: int | None, sort_by: str,
          out_dir: Path, cache_dir: Path) -> None:
    api = _api()
    for d in dates:
        ds = f"kaggle/orbit-wars-episodes-{d}"
        man = _download_file(api, ds, "manifest.csv", cache_dir / d)
        if man is None:
            print(f"[skip] {d}: dataset unpublished or no manifest (403)")
            continue
        eps = select_episodes(man, n_per_day, agent_count, sort_by)
        got = 0
        for ep in eps:
            if (out_dir / f"{ep}.json").exists():
                got += 1; continue
            if _download_file(api, ds, f"{ep}.json", out_dir):
                got += 1
        print(f"[ok]   {d}: {got}/{len(eps)} games (sorted by {sort_by})")


# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #
def _resolve_target(planets, src_id, angle):
    """Planet a launch [src_id, angle, ships] is aimed at (direction match)."""
    src = next((p for p in planets if p[0] == src_id), None)
    if src is None:
        return None
    sx, sy = src[2], src[3]
    best, bd = None, 0.6
    for p in planets:
        if p[0] == src_id:
            continue
        pa = math.atan2(p[3] - sy, p[2] - sx)
        dd = abs((pa - angle + math.pi) % (2 * math.pi) - math.pi)
        if dd < bd:
            bd, best = dd, p
    return best


def _dist(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def _centroid(pls):
    if not pls:
        return None
    return (None, None, sum(p[2] for p in pls) / len(pls), sum(p[3] for p in pls) / len(pls))


def _pick_seat(d, names, player, exclude):
    """Return the seat to analyze for this game, or None to skip.
    player: explicit player name (any outcome) or None for winner-mode.
    exclude: in winner-mode, skip games won by this name."""
    rew = d.get("rewards") or []
    if player:
        return names.index(player) if player in names else None
    # winner mode: needs a unique winner
    if not rew or rew.count(max(rew)) != 1:
        return None
    me = rew.index(max(rew))
    if me >= len(names):
        return None
    if exclude and names[me] == exclude:
        return None
    return me


def analyze(out_dir: Path, player: str | None = None, exclude: str | None = None) -> None:
    """Characterise a player's behaviour across downloaded games.
    Default (player=None): the unique WINNER of each game. With `player`, that named
    player regardless of outcome. With `exclude`, winners except that name."""
    by_count = defaultdict(lambda: [0, 0])     # planet-count bin -> [reinforce, attack]
    count_prod = defaultdict(list)             # owned_count -> total owned production
    tgt_prods, avail_means, tgt_top = [], [], []   # production selectivity (attacks)
    atk_dist_rank = []                         # distance rank of chosen attack target (0=nearest)
    reinf_geo = [0, 0]                         # [forward-staging, rearward] reinforce launches
    commit = {"1ship": 0, "2-4": 0, "<25%": 0, "25-75%": 0, "75-99%": 0, "100%": 0}
    n_commit = 0
    first_launch, first_cap, end_planets, glen = [], [], [], []
    owned_at = defaultdict(list)               # step milestone -> planets owned
    n_games = 0; subjects = defaultdict(int)

    def cbin(n): return f"{n:02d}" if n <= 8 else "09-12" if n <= 12 else "13+"

    for f in glob.glob(str(out_dir / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        names = d.get("info", {}).get("TeamNames", [])
        if len(steps) < 5 or not names:
            continue
        me = _pick_seat(d, names, player, exclude)
        if me is None:
            continue
        n_games += 1; subjects[names[me]] += 1
        glen.append(len(steps))
        seen_launch = seen_cap = False
        prev_owned = None
        for t in range(1, len(steps)):
            # action at step t was decided on the obs at step t-1
            if me >= len(steps[t]) or me >= len(steps[t - 1]):
                continue
            obs = steps[t - 1][me].get("observation", {})
            acts = steps[t][me].get("action") or []
            planets = obs.get("planets")
            if not planets:
                continue
            owned = [p for p in planets if int(p[1]) == me]
            no = len(owned)
            count_prod[no].append(sum(p[6] for p in owned))
            enemy = [p for p in planets if int(p[1]) != me and int(p[1]) >= 0]
            avail = [p for p in planets if int(p[1]) != me]
            ecen = _centroid(enemy)
            for ms in (25, 50, 75, 100):
                if t == ms:
                    owned_at[ms].append(no)
            if prev_owned is not None and no > prev_owned and not seen_cap:
                first_cap.append(t); seen_cap = True
            prev_owned = no
            for a in acts:
                if not a or len(a) < 3:
                    continue
                if not seen_launch:
                    first_launch.append(t); seen_launch = True
                src = next((p for p in planets if p[0] == int(a[0])), None)
                tgt = _resolve_target(planets, int(a[0]), float(a[1]))
                if tgt is None or src is None:
                    continue
                is_r = int(tgt[1]) == me
                by_count[cbin(no)][0 if is_r else 1] += 1
                # ship commitment (only count legal launches: sent <= source ships)
                sent, sships = int(a[2]), float(src[5])
                if sships > 0 and sent <= sships:
                    n_commit += 1
                    frac = sent / sships
                    if sent == 1: commit["1ship"] += 1
                    elif sent <= 4: commit["2-4"] += 1
                    elif frac >= 1.0: commit["100%"] += 1
                    elif frac >= 0.75: commit["75-99%"] += 1
                    elif frac >= 0.25: commit["25-75%"] += 1
                    else: commit["<25%"] += 1
                if is_r:
                    if ecen is not None:
                        reinf_geo[0 if _dist(tgt, ecen) < _dist(src, ecen) else 1] += 1
                else:
                    if avail:
                        tgt_prods.append(tgt[6]); avail_means.append(sum(p[6] for p in avail) / len(avail))
                        tgt_top.append(1 if tgt[6] >= max(p[6] for p in avail) else 0)
                    if len(avail) > 1:
                        dd = sorted(_dist(src, p) for p in avail)
                        atk_dist_rank.append(sum(1 for x in dd if x < _dist(src, tgt)) / (len(dd) - 1))
        end_planets.append(prev_owned if prev_owned is not None else 0)

    if n_games == 0:
        print("no matching games"); return
    md = lambda x: f"{st.median(x):.2f}" if x else "n/a"
    subj = player or (f"WINNERS (excl. {exclude})" if exclude else "WINNERS")
    print(f"=== {subj}: {n_games} games ===")
    print("subjects:", dict(sorted(subjects.items(), key=lambda x: -x[1])[:8]))
    print(f"\nOPENING/TEMPO: first launch median {md(first_launch)}, first capture median {md(first_cap)}")
    for ms in (25, 50, 75, 100):
        print(f"  planets @ step {ms:<3}: median {md(owned_at[ms])} (n={len(owned_at[ms])})")
    print(f"  planets @ end: median {md(end_planets)}   game len median {md(glen)}")

    print("\nREINFORCE RAMP by planet count:")
    for k in sorted(by_count):
        r, a = by_count[k]; tot = r + a
        print(f"  {k:>6} | {r / tot if tot else 0:.2f} | n={tot}")
    fwd, rear = reinf_geo; rt = fwd + rear
    print(f"REINFORCE GEOMETRY: forward-staging {100 * fwd / rt:.0f}% (n={rt})" if rt else "REINFORCE GEOMETRY: none")

    print("\nSHIP COMMITMENT (legal launches, sent/source):")
    for k, v in commit.items():
        print(f"  {k:>8}: {100 * v / n_commit:.1f}%" if n_commit else f"  {k}: n/a")

    print("\nprod/planet:")
    for c in sorted(count_prod := count_prod):
        pass  # (count_prod populated below in original; kept compatible)
    # prod/planet computed inline:
    cp = defaultdict(list)
    # (recompute cheaply is unnecessary; report selectivity instead)
    if tgt_prods:
        print(f"TARGET PROD SELECTIVITY: target/avail {st.mean(tgt_prods) / st.mean(avail_means):.2f}x | hits-top {100 * st.mean(tgt_top):.0f}%")
    if atk_dist_rank:
        print(f"TARGET DISTANCE SELECTIVITY: chosen-target distance rank (0=nearest) mean {st.mean(atk_dist_rank):.2f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", help="explicit YYYY-MM-DD dates")
    ap.add_argument("--last-days", type=int, default=3, help="most recent N dates (default 3)")
    ap.add_argument("--n-per-day", type=int, default=60, help="top games per day by sort key")
    ap.add_argument("--agent-count", type=int, default=2, help="filter to N-player games (default 2)")
    ap.add_argument("--sort-by", default="avg_score", help="manifest column to rank by")
    ap.add_argument("--out-dir", default="/tmp/fresh_validate")
    ap.add_argument("--cache-dir", default="/tmp/ow_manifests")
    ap.add_argument("--no-download", action="store_true", help="skip fetching; analyze existing dir")
    ap.add_argument("--analyze", action="store_true", help="run behavioural analysis after fetch")
    ap.add_argument("--player", help="analyze this player (any outcome); default = each game's winner")
    ap.add_argument("--exclude", help="winner-mode: skip games won by this player (generalisation check)")
    args = ap.parse_args()

    out_dir, cache_dir = Path(args.out_dir), Path(args.cache_dir)
    if not args.no_download:
        dates = args.dates or recent_dates(args.last_days)
        out_dir.mkdir(parents=True, exist_ok=True)
        fetch(dates, args.n_per_day, args.agent_count, args.sort_by, out_dir, cache_dir)
    if args.analyze or args.no_download:
        analyze(out_dir, player=args.player, exclude=args.exclude)


if __name__ == "__main__":
    main()
