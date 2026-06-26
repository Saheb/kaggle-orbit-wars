"""Stage-0 audit for the autoregressive action-list pivot.

This is a read-only premise check. It asks whether top-player native action
lists contain deliberate floor-covering multi-source aggregation beyond what
our current factored policy appears to do.

Timing convention: action at steps[t][seat] was selected from observation at
steps[t-1][seat].

Default corpus:
  gpu_run_artifacts/ar_stage0/replays/top2
  gpu_run_artifacts/ar_stage0/replays/jake
  leader-replays/rank1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "orbit_wars_rl"
for _path in (ROOT, PKG):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from orbit_wars_rl import eval as ev  # noqa: E402
from orbit_wars_rl.features import extract_features  # noqa: E402


PHASES = (("all", 0, 10**9), ("open", 0, 50), ("mid", 50, 100), ("late", 100, 10**9))
DEFAULT_PATHS = (
    "gpu_run_artifacts/ar_stage0/replays/top2",
    "gpu_run_artifacts/ar_stage0/replays/jake",
    "leader-replays/rank1",
)


def _iter_paths(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(Path(x) for x in glob.glob(str(p / "*.json")))
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(x) for x in glob.glob(item))
        else:
            out.append(p)
    return sorted(set(out))


def _phase(step: int) -> str:
    for name, lo, hi in PHASES:
        if name == "all":
            continue
        if lo <= step < hi:
            return name
    return "late"


def _pid(p: list[Any]) -> int:
    return int(p[0])


def _owner(p: list[Any]) -> int:
    return int(p[1])


def _winner_seat(replay: dict[str, Any]) -> int | None:
    rewards = replay.get("rewards") or []
    if len(rewards) != 2:
        return None
    return int(max(range(len(rewards)), key=lambda i: rewards[i]))


def _seat_name(replay: dict[str, Any], seat: int) -> str:
    names = replay.get("info", {}).get("TeamNames") or []
    return names[seat] if seat < len(names) else f"seat{seat}"


def _resolve_launch_target(planets: list[list[Any]], src: list[Any], angle: float) -> list[Any] | None:
    sx, sy = float(src[2]), float(src[3])
    best, best_delta = None, 0.6
    for p in planets:
        if _pid(p) == _pid(src):
            continue
        pa = math.atan2(float(p[3]) - sy, float(p[2]) - sx)
        delta = abs((pa - angle + math.pi) % (2 * math.pi) - math.pi)
        if delta < best_delta:
            best, best_delta = p, delta
    return best


def _fleet_eta(src: list[Any], tgt: list[Any], ships: float) -> float:
    dist = math.hypot(float(tgt[2]) - float(src[2]), float(tgt[3]) - float(src[3]))
    return dist / max(ev._ship_speed_py(max(float(ships), 1.0)), 1e-6)


def _target_floor_at_eta(planets: list[list[Any]], fleets: list[list[Any]], tgt: list[Any],
                         seat: int, eta: float, beta: float) -> float:
    owner = _owner(tgt)
    if owner == seat:
        return 0.0
    if owner < 0:
        return max(float(tgt[5]) + ev._DM_OVERHEAD, 1e-6)
    eta = min(max(1.0, float(eta)), ev._DM_HORIZON)
    inbound = ev._friendly_inbound(fleets, tgt, 1 - seat)
    enemy_mass = 0.0
    for ep in planets:
        eo = _owner(ep)
        if eo < 0 or eo == seat or _pid(ep) == _pid(tgt):
            continue
        reach = max(ev._fleet_speed(int(ep[5])) * ev._DM_HORIZON, 1e-6)
        d = math.hypot(float(ep[2]) - float(tgt[2]), float(ep[3]) - float(tgt[3]))
        enemy_mass += float(ep[5]) * max(1.0 - d / reach, 0.0)
    rho = min(max((eta - ev._DM_ETA_FREE) / ev._DM_ETA_SCALE, 0.0), 1.0)
    return max(float(tgt[5]) + float(tgt[6]) * eta + inbound + beta * rho * enemy_mass + ev._DM_OVERHEAD, 1e-6)


def _friendly_inbound_to_target(planets: list[list[Any]], fleets: list[list[Any]], tgt: list[Any],
                                seat: int, max_eta: float | None = None) -> float:
    """Friendly in-flight mass already aimed at `tgt`, optionally arriving by `max_eta`.

    This is stricter than eval._friendly_inbound: it resolves the fleet's target
    through _dm_fleet_target, then checks remaining travel time. It is used only
    for the temporal co-arrival audit; the legacy floor still uses eval's helper
    to keep the original agg2 definition stable.
    """
    total = 0.0
    for f in fleets or []:
        if int(f[1]) != seat:
            continue
        r = ev._dm_fleet_target(planets, f)
        if r is None or _pid(r) != _pid(tgt):
            continue
        if max_eta is not None:
            dist = math.hypot(float(tgt[2]) - float(f[2]), float(tgt[3]) - float(f[3]))
            eta = dist / max(ev._ship_speed_py(float(f[6])), 1e-6)
            if eta > max_eta:
                continue
        total += float(f[6])
    return total


def _drainable_sources(planets: list[list[Any]], fleets: list[list[Any]], tgt: list[Any],
                       seat: int, window: float, beta: float) -> tuple[float, float, int]:
    """Return (total_spare, max_single_spare, source_count) reachable by `window`.

    Mirrors eval's agg2 idea, but exposes max single spare so Stage 0 can ask
    whether a target genuinely needs multiple reachable/drainable sources.
    """
    enemy_in_src: dict[int, float] = {}
    for f in fleets or []:
        o = int(f[1])
        if o < 0 or o == seat:
            continue
        r = ev._dm_fleet_target(planets, f)
        if r is not None:
            enemy_in_src[_pid(r)] = enemy_in_src.get(_pid(r), 0.0) + float(f[6])

    total = 0.0
    max_single = 0.0
    count = 0
    for src in planets:
        if _owner(src) != seat or _pid(src) == _pid(tgt) or float(src[5]) <= 0:
            continue
        g = float(src[5])
        own_threat = enemy_in_src.get(_pid(src), 0.0)
        if own_threat >= g:
            spare = g
        elif own_threat > 0:
            hold_floor = own_threat + beta * ev._reachable_enemy_mass(planets, src, seat) + ev._DM_OVERHEAD
            spare = max(0.0, g - hold_floor)
        else:
            spare = g
        if spare <= 0:
            continue
        if _fleet_eta(src, tgt, spare) <= window:
            total += spare
            max_single = max(max_single, spare)
            count += 1
    return total, max_single, count


def _pct(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * q))))
    return xs[idx]


def _fmt(v: float | None, digits: int = 2) -> str:
    return "NA" if v is None else f"{v:.{digits}f}"


def _target_value_context(obs: dict[str, Any], seat: int) -> dict[str, Any] | None:
    """Build a cheap lookup from replay planet ids to pairwise target-value rows."""
    planets = obs.get("planets") or []
    if not planets:
        return None
    try:
        feats = extract_features(obs, player=seat, num_players=2, max_planets=48, max_fleets=128)
    except Exception:
        return None

    pairwise = feats.get("pairwise_features")
    owned_indices = feats.get("owned_indices")
    if pairwise is None or owned_indices is None:
        return None
    if hasattr(pairwise, "detach"):
        pairwise = pairwise.detach().cpu().numpy()
    if hasattr(owned_indices, "detach"):
        owned_indices = owned_indices.detach().cpu().numpy()

    pid_to_idx = {_pid(p): i for i, p in enumerate(planets)}
    src_pid_to_slot: dict[int, int] = {}
    for slot, src_idx in enumerate(owned_indices):
        idx = int(src_idx)
        if idx >= len(planets) or slot >= pairwise.shape[0]:
            continue
        p = planets[idx]
        if _owner(p) == seat:
            src_pid_to_slot[_pid(p)] = slot
    return {"pairwise": pairwise, "pid_to_idx": pid_to_idx, "src_pid_to_slot": src_pid_to_slot}


def _target_value_row(ctx: dict[str, Any] | None, src_pid: int, tgt_pid: int) -> tuple[float, float, float, float] | None:
    if ctx is None:
        return None
    slot = ctx["src_pid_to_slot"].get(src_pid)
    tgt_idx = ctx["pid_to_idx"].get(tgt_pid)
    if slot is None or tgt_idx is None:
        return None
    pairwise = ctx["pairwise"]
    if tgt_idx >= pairwise.shape[1] or pairwise.shape[2] < 20:
        return None
    row = pairwise[slot, tgt_idx]
    return float(row[16]), float(row[17]), float(row[18]), float(row[19])


def _record_target_value(stats: Counter, ratios: dict[str, list[float]],
                         row: tuple[float, float, float, float] | None) -> None:
    if row is None:
        stats["value_missing_attack_moves"] += 1
        return
    capture_value, reactive_roi, friendly_reach, keepability = row
    stats["value_audited_attack_moves"] += 1
    stats["low_value_attack_moves"] += int(capture_value < 0.05 or reactive_roi < 0.0)
    stats["negative_keep_attack_moves"] += int(keepability < 0.0)
    ratios["capture_value_40"].append(capture_value)
    ratios["reactive_roi_40"].append(reactive_roi)
    ratios["friendly_reachable_mass"].append(friendly_reach)
    ratios["keepability_margin"].append(keepability)


def _update_list_stats(c: Counter, action_len: int) -> None:
    c["action_turns"] += 1
    c["native_moves"] += action_len
    c["turns_gt16_moves"] += int(action_len > 16)
    c["list_len_max"] = max(c["list_len_max"], action_len)
    c.setdefault("_list_lens", []).append(action_len)  # removed before JSON output


def _analyze_turn(obs: dict[str, Any], action: Any, seat: int, step: int, beta: float,
                  coarrival_window: float) -> tuple[Counter, dict[str, list[float]]]:
    stats = Counter()
    ratios: dict[str, list[float]] = defaultdict(list)
    raw_moves = action if isinstance(action, list) else []
    planets = obs.get("planets") or []
    fleets = obs.get("fleets") or []
    if not planets:
        return stats, ratios

    pmap = {_pid(p): p for p in planets}
    own_pids = {_pid(p) for p in planets if _owner(p) == seat}
    value_ctx = _target_value_context(obs, seat) if raw_moves else None
    if raw_moves:
        _update_list_stats(stats, len(raw_moves))

    source_counts = Counter()
    target_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    owned_moves = 0
    resolved_moves = 0

    for move in raw_moves:
        if not isinstance(move, (list, tuple)) or len(move) < 3:
            stats["bad_move"] += 1
            continue
        try:
            src_pid = int(move[0])
            angle = float(move[1])
            ships = max(0, int(move[2]))
        except Exception:
            stats["bad_move"] += 1
            continue
        src = pmap.get(src_pid)
        if src is None or src_pid not in own_pids:
            stats["not_owned_source"] += 1
            continue
        owned_moves += 1
        stats["owned_ship_mass"] += ships
        tgt = _resolve_launch_target(planets, src, angle)
        if tgt is None:
            stats["unresolved_target"] += 1
            stats["unresolved_ship_mass"] += ships
            continue
        resolved_moves += 1
        source_counts[src_pid] += 1
        if _owner(tgt) == seat:
            stats["save_moves"] += 1
            continue
        stats["attack_moves"] += 1
        _record_target_value(stats, ratios, _target_value_row(value_ctx, src_pid, _pid(tgt)))
        target_groups[_pid(tgt)].append({"src": src, "tgt": tgt, "ships": ships})

    stats["owned_moves"] += owned_moves
    stats["resolved_moves"] += resolved_moves
    stats["same_source_turn"] += int(any(c > 1 for c in source_counts.values()))
    stats["same_source_moves"] += sum(c for c in source_counts.values() if c > 1)

    aggregate_targets = 0
    for rows in target_groups.values():
        source_ids = {_pid(r["src"]) for r in rows}
        if len(source_ids) > 1:
            aggregate_targets += 1
    stats["aggregate_turn"] += int(aggregate_targets > 0)
    stats["aggregate_targets"] += aggregate_targets

    for rows in target_groups.values():
        tgt = rows[0]["tgt"]
        etas = [_fleet_eta(r["src"], tgt, r["ships"]) for r in rows]
        eta = min(max(max(etas, default=1.0), 1.0), ev._DM_HORIZON)
        window = eta + 10.0
        floor = _target_floor_at_eta(planets, fleets, tgt, seat, eta, beta)
        timed_prior_inbound = _friendly_inbound_to_target(
            planets, fleets, tgt, seat, max_eta=eta + coarrival_window
        )
        prior_inbound = ev._friendly_inbound(fleets, tgt, seat)
        need = max(0.0, floor - prior_inbound)
        committed = float(sum(int(r["ships"]) for r in rows))
        committed_sources = len({_pid(r["src"]) for r in rows})
        current_plus_timed = committed + timed_prior_inbound

        stats["attack_targets"] += 1
        stats["attack_target_floor_sum"] += floor
        stats["attack_target_need_sum"] += need
        stats["attack_target_committed_sum"] += committed
        stats["attack_target_timed_prior_sum"] += timed_prior_inbound
        if timed_prior_inbound >= floor:
            stats["attack_target_timed_prior_cross"] += 1
        if current_plus_timed >= floor:
            stats["attack_target_coarrival_cross"] += 1
            ratios["attack_target_coarrival_ratio"].append(current_plus_timed / max(floor, 1e-6))
        if committed >= need:
            stats["attack_target_cross"] += 1
            ratios["attack_cross_ratio"].append(committed / max(need, 1e-6))
        if need <= 0:
            stats["already_covered_targets"] += 1
            continue

        total_spare, max_single_spare, drainable_count = _drainable_sources(planets, fleets, tgt, seat, window, beta)

        floor_needed = max_single_spare < need <= total_spare and drainable_count >= 2
        if not floor_needed:
            continue

        stats["floor_needed_targets"] += 1
        stats["floor_needed_need_sum"] += need
        stats["floor_needed_committed_sum"] += committed
        stats["floor_needed_timed_prior_sum"] += timed_prior_inbound
        stats["floor_needed_drainable_sum"] += total_spare
        stats["floor_needed_max_single_sum"] += max_single_spare
        if timed_prior_inbound >= floor:
            stats["floor_needed_timed_prior_cross"] += 1
        if current_plus_timed >= floor:
            stats["floor_needed_coarrival_cross"] += 1
            ratios["floor_needed_coarrival_ratio"].append(current_plus_timed / max(floor, 1e-6))
        if committed_sources > 1:
            stats["floor_needed_multi_source"] += 1
        if committed >= need:
            stats["floor_needed_cross"] += 1
            ratios["floor_needed_cross_ratio"].append(committed / max(need, 1e-6))
            if committed_sources > 1:
                stats["floor_needed_multi_source_cross"] += 1
                ratios["floor_needed_multi_source_cross_ratio"].append(committed / max(need, 1e-6))
        else:
            stats["floor_needed_stop_short"] += 1
        if aggregate_targets == 0 or committed_sources <= 1:
            stats["floor_needed_no_multi_commit"] += 1

    return stats, ratios


def _merge_stats(dst: Counter, src: Counter) -> None:
    for k, v in src.items():
        if k == "_list_lens":
            dst.setdefault("_list_lens", []).extend(v)
        elif k == "list_len_max":
            dst[k] = max(dst[k], v)
        else:
            dst[k] += v


def _merge_ratios(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for k, vals in src.items():
        dst.setdefault(k, []).extend(vals)


def analyze(paths: list[Path], mode: str, beta: float, require_1v1: bool,
            coarrival_window: float) -> dict[str, Any]:
    totals = Counter()
    phase_totals: dict[str, Counter] = {name: Counter() for name, _, _ in PHASES}
    per_player: dict[str, Counter] = defaultdict(Counter)
    ratios: dict[str, list[float]] = defaultdict(list)
    phase_ratios: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name, _, _ in PHASES}

    for path in paths:
        totals["paths"] += 1
        try:
            replay = json.loads(path.read_text())
        except Exception:
            totals["read_fail"] += 1
            continue
        rewards = replay.get("rewards") or []
        if require_1v1 and len(rewards) != 2:
            totals["excluded_non_1v1"] += 1
            continue
        steps = replay.get("steps") or []
        if len(steps) < 2:
            totals["short_replay"] += 1
            continue
        if mode == "winner":
            seat = _winner_seat(replay)
            seats = [] if seat is None else [seat]
        elif mode == "all":
            seats = list(range(len(steps[0])))
        else:
            raise ValueError(f"unsupported mode: {mode}")
        if not seats:
            totals["no_selected_seat"] += 1
            continue
        totals["replays_used"] += 1

        for seat in seats:
            pname = _seat_name(replay, seat)
            per_player[pname]["replays"] += 1
            for t in range(1, len(steps)):
                if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                    continue
                prev_row = steps[t - 1][seat] or {}
                cur_row = steps[t][seat] or {}
                obs = prev_row.get("observation") or {}
                action = cur_row.get("action") or []
                totals["usable_turns"] += 1
                phase = _phase(t)
                phase_totals[phase]["usable_turns"] += 1
                st, rt = _analyze_turn(obs, action, seat, t, beta, coarrival_window)
                _merge_stats(totals, st)
                _merge_stats(phase_totals["all"], st)
                _merge_stats(phase_totals[phase], st)
                _merge_stats(per_player[pname], st)
                _merge_ratios(ratios, rt)
                _merge_ratios(phase_ratios["all"], rt)
                _merge_ratios(phase_ratios[phase], rt)

    return {
        "totals": totals,
        "phase_totals": phase_totals,
        "per_player": per_player,
        "ratios": ratios,
        "phase_ratios": phase_ratios,
    }


def _summarize(label: str, c: Counter, ratios: dict[str, list[float]]) -> dict[str, Any]:
    lens = c.get("_list_lens", [])
    floor_ratios = ratios.get("floor_needed_multi_source_cross_ratio", [])
    coarrival_ratios = ratios.get("floor_needed_coarrival_ratio", [])
    capture_values = ratios.get("capture_value_40", [])
    reactive_rois = ratios.get("reactive_roi_40", [])
    friendly_reaches = ratios.get("friendly_reachable_mass", [])
    keepability_margins = ratios.get("keepability_margin", [])
    return {
        "label": label,
        "replays_used": int(c["replays_used"]),
        "usable_turns": int(c["usable_turns"]),
        "action_turns": int(c["action_turns"]),
        "native_moves": int(c["native_moves"]),
        "owned_moves": int(c["owned_moves"]),
        "resolved_moves": int(c["resolved_moves"]),
        "list_len_p50": _quantile(lens, 0.50),
        "list_len_p90": _quantile(lens, 0.90),
        "list_len_max": int(c["list_len_max"]),
        "turns_gt16_moves_rate": _pct(c["turns_gt16_moves"], c["action_turns"]),
        "same_source_turn_rate": _pct(c["same_source_turn"], c["action_turns"]),
        "aggregate_turn_rate": _pct(c["aggregate_turn"], c["action_turns"]),
        "attack_targets": int(c["attack_targets"]),
        "attack_target_cross_rate": _pct(c["attack_target_cross"], c["attack_targets"]),
        "attack_target_coarrival_cross_rate": _pct(c["attack_target_coarrival_cross"], c["attack_targets"]),
        "attack_target_timed_prior_cross_rate": _pct(c["attack_target_timed_prior_cross"], c["attack_targets"]),
        "floor_needed_targets": int(c["floor_needed_targets"]),
        "floor_needed_multi_source_rate": _pct(c["floor_needed_multi_source"], c["floor_needed_targets"]),
        "floor_needed_cross_rate": _pct(c["floor_needed_cross"], c["floor_needed_targets"]),
        "floor_needed_multi_source_cross_rate": _pct(c["floor_needed_multi_source_cross"], c["floor_needed_targets"]),
        "floor_needed_coarrival_cross_rate": _pct(c["floor_needed_coarrival_cross"], c["floor_needed_targets"]),
        "floor_needed_timed_prior_cross_rate": _pct(c["floor_needed_timed_prior_cross"], c["floor_needed_targets"]),
        "floor_needed_stop_short_rate": _pct(c["floor_needed_stop_short"], c["floor_needed_targets"]),
        "floor_needed_no_multi_commit_rate": _pct(c["floor_needed_no_multi_commit"], c["floor_needed_targets"]),
        "floor_needed_cross_ratio_p50": _quantile(floor_ratios, 0.50),
        "floor_needed_cross_ratio_p90": _quantile(floor_ratios, 0.90),
        "floor_needed_coarrival_ratio_p50": _quantile(coarrival_ratios, 0.50),
        "floor_needed_coarrival_ratio_p90": _quantile(coarrival_ratios, 0.90),
        "value_audited_attack_moves": int(c["value_audited_attack_moves"]),
        "value_missing_attack_moves": int(c["value_missing_attack_moves"]),
        "low_value_attack_rate": _pct(c["low_value_attack_moves"], c["value_audited_attack_moves"]),
        "negative_keep_attack_rate": _pct(c["negative_keep_attack_moves"], c["value_audited_attack_moves"]),
        "capture_value_p10": _quantile(capture_values, 0.10),
        "capture_value_p50": _quantile(capture_values, 0.50),
        "capture_value_p90": _quantile(capture_values, 0.90),
        "reactive_roi_p10": _quantile(reactive_rois, 0.10),
        "reactive_roi_p50": _quantile(reactive_rois, 0.50),
        "reactive_roi_p90": _quantile(reactive_rois, 0.90),
        "friendly_reach_p50": _quantile(friendly_reaches, 0.50),
        "keepability_p10": _quantile(keepability_margins, 0.10),
        "keepability_p50": _quantile(keepability_margins, 0.50),
        "keepability_p90": _quantile(keepability_margins, 0.90),
    }


def _clean_counter(c: Counter) -> dict[str, Any]:
    return {k: int(v) if isinstance(v, int) else v for k, v in c.items() if k != "_list_lens"}


def _print_summary(row: dict[str, Any]) -> None:
    print(
        f"{row['label']:>8s} turns={row['usable_turns']} action_turns={row['action_turns']} "
        f"moves={row['native_moves']} list_p50/p90/max={_fmt(row['list_len_p50'],0)}/"
        f"{_fmt(row['list_len_p90'],0)}/{row['list_len_max']} "
        f"aggTurn={row['aggregate_turn_rate']:.3f} sameSrc={row['same_source_turn_rate']:.3f} "
        f"floorN={row['floor_needed_targets']} "
        f"msCross={row['floor_needed_multi_source_cross_rate']:.3f} "
        f"coArr={row['floor_needed_coarrival_cross_rate']:.3f} "
        f"stopShort={row['floor_needed_stop_short_rate']:.3f} "
        f"lean_p50/p90={_fmt(row['floor_needed_cross_ratio_p50'])}/{_fmt(row['floor_needed_cross_ratio_p90'])} "
        f"valueN={row['value_audited_attack_moves']} lowVal={row['low_value_attack_rate']:.3f} "
        f"negKeep={row['negative_keep_attack_rate']:.3f} "
        f"val_p50={_fmt(row['capture_value_p50'])} roi_p50={_fmt(row['reactive_roi_p50'])} "
        f"keep_p50={_fmt(row['keepability_p50'])}"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AR Stage-0 action-list audit",
        "",
        f"Paths scanned: {payload['paths_scanned']}",
        f"Mode: `{payload['mode']}`",
        f"Require 1v1: `{payload['require_1v1']}`",
        f"Beta: `{payload['beta']}`",
        f"Co-arrival window: `{payload['coarrival_window']}`",
        "",
        "## Gate",
        "",
        "Pre-registered gate from `docs/autoregressive-head.md`: PASS only if winner floor-needed",
        "multi-source floor-cross exceeds our comparable policy by >=15pp and lean-overkill p50 is",
        ">=20% lower. This script reports the winner side; compare against the policy-side Stage-0",
        "rollout/audit before passing the gate.",
        "",
        "## Summary",
        "",
        "| phase | action turns | moves | list p50/p90/max | aggTurn | sameSrc | floor-needed | multi-source cross | co-arrival cross | stop-short | lean p50/p90 | co-arrival p50/p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summaries"]:
        lines.append(
            f"| {row['label']} | {row['action_turns']} | {row['native_moves']} | "
            f"{_fmt(row['list_len_p50'],0)}/{_fmt(row['list_len_p90'],0)}/{row['list_len_max']} | "
            f"{row['aggregate_turn_rate']:.3f} | {row['same_source_turn_rate']:.3f} | "
            f"{row['floor_needed_targets']} | {row['floor_needed_multi_source_cross_rate']:.3f} | "
            f"{row['floor_needed_coarrival_cross_rate']:.3f} | {row['floor_needed_stop_short_rate']:.3f} | "
            f"{_fmt(row['floor_needed_cross_ratio_p50'])}/{_fmt(row['floor_needed_cross_ratio_p90'])} | "
            f"{_fmt(row['floor_needed_coarrival_ratio_p50'])}/{_fmt(row['floor_needed_coarrival_ratio_p90'])} |"
        )
    lines.extend([
        "",
        "## Target Value Audit",
        "",
        "| phase | audited attacks | missing | low-value rate | negative-keep rate | value p10/p50/p90 | reactive ROI p10/p50/p90 | friendly reach p50 | keepability p10/p50/p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["summaries"]:
        lines.append(
            f"| {row['label']} | {row['value_audited_attack_moves']} | {row['value_missing_attack_moves']} | "
            f"{row['low_value_attack_rate']:.3f} | {row['negative_keep_attack_rate']:.3f} | "
            f"{_fmt(row['capture_value_p10'])}/{_fmt(row['capture_value_p50'])}/{_fmt(row['capture_value_p90'])} | "
            f"{_fmt(row['reactive_roi_p10'])}/{_fmt(row['reactive_roi_p50'])}/{_fmt(row['reactive_roi_p90'])} | "
            f"{_fmt(row['friendly_reach_p50'])} | "
            f"{_fmt(row['keepability_p10'])}/{_fmt(row['keepability_p50'])}/{_fmt(row['keepability_p90'])} |"
        )
    lines.extend([
        "",
        "## Raw Counters",
        "",
        "```json",
        json.dumps(payload["counters"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS),
                    help="Replay JSON files, directories, or globs. Defaults to Stage-0 durable corpus.")
    ap.add_argument("--mode", choices=("winner", "all"), default="winner")
    ap.add_argument("--no-require-1v1", action="store_true")
    ap.add_argument("--beta", type=float, default=ev._DM_BETA_EVAL)
    ap.add_argument("--coarrival-window", type=float, default=10.0,
                    help="Count existing friendly inbound as co-arriving if it arrives by current max ETA + this many steps.")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    paths = _iter_paths(args.paths)
    res = analyze(
        paths,
        mode=args.mode,
        beta=args.beta,
        require_1v1=not args.no_require_1v1,
        coarrival_window=args.coarrival_window,
    )
    summaries = []
    for name, _, _ in PHASES:
        c = res["totals"] if name == "all" else res["phase_totals"][name]
        summaries.append(_summarize(name, c, res["phase_ratios"][name]))

    payload = {
        "paths_scanned": len(paths),
        "mode": args.mode,
        "require_1v1": not args.no_require_1v1,
        "beta": args.beta,
        "coarrival_window": args.coarrival_window,
        "summaries": summaries,
        "counters": {
            "totals": _clean_counter(res["totals"]),
            "phase_totals": {k: _clean_counter(v) for k, v in res["phase_totals"].items()},
        },
    }

    print(
        f"paths={len(paths)} mode={args.mode} require_1v1={not args.no_require_1v1} "
        f"coarrival_window={args.coarrival_window}"
    )
    print(
        f"replays_used={res['totals']['replays_used']} excluded_non_1v1={res['totals']['excluded_non_1v1']} "
        f"read_fail={res['totals']['read_fail']}"
    )
    for row in summaries:
        _print_summary(row)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out, payload)


if __name__ == "__main__":
    main()
