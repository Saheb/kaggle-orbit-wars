"""Over-extension probe — is the out-mass wall driven by capturing planets we
physically can't defend (positional over-extension), vs failing to route defense?

Hypothesis (this session): captures we LOSE to out-massing are planets sitting
closer to ENEMY support than to OUR support at capture time — we over-extend,
grab forward/isolated planets, and they get peeled regardless of attack-mass.

For each capture (a planet that flips TO us mid-game; home/initial excluded), at
the step of capture we classify it by POSITION:
  near_friend = dist to our nearest OTHER owned planet
  near_enemy  = dist to the nearest enemy planet
  OVER-EXTENDED if near_enemy < near_friend  (enemy can reach it faster than we can)
Then we follow it: of captures we LOSE, what fraction were over-extended? And is
the lost-rate of over-extended captures >> supported ones?

  over-ext lost-rate >> supported  → losses are TARGET SELECTION (grab undefendable)
  similar lost-rates               → losses are routing/under-mass on defendable planets

Reuses the Probe-A replay corpus (kaggle JSON, our seat tagged "OURS").
Usage:  python3 probe_overextension.py <dir-or-glob> [...]  [--player OURS]
"""
import argparse
import glob
import json
import math
from pathlib import Path


def _pid(p):
    return int(p[0])


def _owner(p):
    return int(p[1])


def _seat_names(replay):
    return list(replay.get("info", {}).get("TeamNames") or [])


def _iter_paths(inputs):
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(Path(x) for x in glob.glob(str(p / "*.json")))
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(x) for x in glob.glob(item))
        else:
            out.append(p)
    return sorted(set(out))


def _nearest_support(planets, pid, px, py, seat):
    """(near_friend_dist, near_enemy_dist) from planet pid: nearest OTHER owned
    planet and nearest enemy planet. inf if none of that owner exists."""
    nf = ne = math.inf
    for q in planets:
        if _pid(q) == pid:
            continue
        d = math.hypot(q[2] - px, q[3] - py)
        o = _owner(q)
        if o == seat:
            nf = min(nf, d)
        elif o >= 0:
            ne = min(ne, d)
    return nf, ne


def analyze(paths, player):
    caps = 0
    cap_overext = 0
    lost = [0, 0]          # [over-ext, supported]  captures we LOST
    cap_by_cls = [0, 0]    # [over-ext, supported]  all captures (for lost-rate denominators)
    ratios = []
    games = 0

    for path in paths:
        try:
            replay = json.loads(path.read_text())
        except Exception:
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            continue
        names = _seat_names(replay)
        seats = [i for i, n in enumerate(names) if n == player] if player else [0]
        if not seats:
            continue
        for seat in seats:
            games += 1
            open_overext = {}   # pid -> bool (over-extended at capture), while we hold it
            for t in range(1, len(steps)):
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    continue
                p_now = (steps[t][seat] or {}).get("observation", {}).get("planets") or []
                p_prev = (steps[t - 1][seat] or {}).get("observation", {}).get("planets") or []
                if not p_now or not p_prev:
                    continue
                prev_owner = {_pid(q): _owner(q) for q in p_prev}
                for q in p_now:
                    pid = _pid(q)
                    own = _owner(q)
                    was = prev_owner.get(pid)
                    if was is None:
                        continue
                    if was != seat and own == seat:
                        # CAPTURE — classify by position on the board right after taking it
                        nf, ne = _nearest_support(p_now, pid, q[2], q[3], seat)
                        if not (math.isfinite(nf) and math.isfinite(ne)):
                            continue
                        oe = ne < nf
                        caps += 1
                        cap_overext += int(oe)
                        cap_by_cls[0 if oe else 1] += 1
                        ratios.append(ne / nf if nf > 0 else math.inf)
                        open_overext[pid] = oe
                    elif was == seat and own != seat and pid in open_overext:
                        # LOSS of a tracked capture
                        lost[0 if open_overext[pid] else 1] += 1
                        del open_overext[pid]
    return {"games": games, "caps": caps, "cap_overext": cap_overext,
            "cap_by_cls": cap_by_cls, "lost": lost, "ratios": ratios}


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def _report(label, r):
    caps = r["caps"]
    print(f"\n===== {label} =====")
    print(f"games {r['games']}  captures {caps}")
    if caps == 0:
        return
    rr = sorted(r["ratios"])
    p50 = rr[len(rr) // 2] if rr else float("nan")
    print(f"  at-capture  over-extended {_pct(r['cap_overext'], caps):.0f}%  "
          f"median ratio(enemy_d/friend_d) {p50:.2f}   [<1 = enemy planet closer = over-extended]")
    oe_caps, sup_caps = r["cap_by_cls"]
    oe_lost, sup_lost = r["lost"]
    print(f"  lost-rate   over-extended {_pct(oe_lost, oe_caps):.0f}% ({oe_lost}/{oe_caps})  "
          f"vs supported {_pct(sup_lost, sup_caps):.0f}% ({sup_lost}/{sup_caps})")
    tot_lost = oe_lost + sup_lost
    print(f"  of LOST captures: over-extended-at-capture {_pct(oe_lost, tot_lost):.0f}%  "
          f"supported {_pct(sup_lost, tot_lost):.0f}%")
    gap = _pct(oe_lost, oe_caps) - _pct(sup_lost, sup_caps)
    if oe_caps >= 20 and sup_caps >= 20:
        if gap >= 10:
            print(f"  → over-ext lost-rate +{gap:.0f}pp over supported = losses driven by OVER-EXTENSION "
                  f"(target-selection lever)")
        else:
            print(f"  → over-ext vs supported lost-rate ~equal ({gap:+.0f}pp) = NOT over-extension; "
                  f"losses are routing/under-mass on defendable planets")
    else:
        print(f"  → inconclusive (need ≥20 captures per class; have {oe_caps}/{sup_caps})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--player", default="OURS")
    args = ap.parse_args()
    paths = _iter_paths(args.paths)
    _report(f"paths={len(paths)} player={args.player}", analyze(paths, args.player))


if __name__ == "__main__":
    main()
