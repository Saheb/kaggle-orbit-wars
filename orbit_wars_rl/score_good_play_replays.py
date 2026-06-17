"""Score replay seats for high-precision supervised BC filtering.

This answers "learn what good play looks like" with an auditable proxy:
selected winner seats must look like elite replay behaviour on conversion,
expansion, retention, and launch waste before their decisions are cloned.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbit_wars_rl.build_supervised_bc import (  # noqa: E402
    _add_sample_stats,
    _clone_sample,
    _matches_any,
    _moves,
    _normalize_obs,
    _reinforce_label_count,
    _strict_winner,
    _team_names,
)
from orbit_wars_rl.bc import trajectory_to_training_sample  # noqa: E402
from orbit_wars_rl.eval import _resolve_launch_target, game_conversion  # noqa: E402
from orbit_wars_rl.action_mask import _target_intercept_angle  # noqa: E402


DEFAULT_STRONG_NAMES = [
    "Isaiah",
    "Jake Will",
    "TonyK",
    "typeIIIfairy",
    "Hober Malloc",
    "Vadasz",
]


@dataclass
class QualityThresholds:
    min_attack_launches: int = 6
    min_cap_attack: float = 0.30
    min_planets16: int = 1
    min_planets50: int = 5
    max_lost_cap: float = 0.85
    max_redundant_early: float = 0.30
    max_underkill_early: float = 0.80
    min_score: float = 7.0


def _attr_steps(steps: list) -> list:
    out = []
    for step in steps:
        row = []
        for agent in step:
            row.append(SimpleNamespace(
                observation=agent.get("observation", {}),
                action=agent.get("action") or [],
            ))
        out.append(row)
    return out


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _garrison_frac(conv: dict, ms: int) -> float | None:
    g = conv.get(f"g{ms}")
    inf = conv.get(f"if{ms}")
    if g is None or inf is None or (g + inf) <= 0:
        return None
    return g / (g + inf)


def _ships_per_planet(conv: dict, ms: int) -> float | None:
    p = conv.get(f"p{ms}")
    g = conv.get(f"g{ms}")
    if p is None or g is None or p <= 0:
        return None
    return g / p


def _metrics_from_conversion(conv: dict) -> dict:
    atk = conv["attack_launches"]
    caps = conv["captures"]
    atk_early = conv["atk_early"]
    hold = conv["hold_durations"]
    return {
        "captures": caps,
        "attack_launches": atk,
        "reinforce_launches": conv["reinforce_launches"],
        "cap_attack": _ratio(caps, atk),
        "ships_cap": _ratio(conv["attack_ships"], caps),
        "lost_cap": _ratio(conv["lost_caps"], caps),
        "median_hold": int(statistics.median(hold)) if hold else 0,
        "redundant_early": _ratio(conv["redundant_early"], atk_early),
        "underkill_early": _ratio(conv["underkill_early"], atk_early),
        "atk_early": atk_early,
        "p16": conv.get("p16"),
        "p32": conv.get("p32"),
        "p50": conv.get("p50"),
        "p100": conv.get("p100"),
        "end_planets": conv["end_planets"],
        "garr50": _garrison_frac(conv, 50),
        "shipspp50": _ships_per_planet(conv, 50),
        "game_len": conv["glen"],
    }


def score_metrics(metrics: dict, name: str, thresholds: QualityThresholds,
                  strong_names: list[str]) -> tuple[float, list[str], list[str]]:
    reasons: list[str] = []
    hard_fails: list[str] = []
    score = 0.0

    atk = metrics["attack_launches"]
    cap_attack = metrics["cap_attack"]
    lost_cap = metrics["lost_cap"]
    red = metrics["redundant_early"]
    under = metrics["underkill_early"]
    p16 = metrics["p16"]
    p50 = metrics["p50"]
    garr50 = metrics["garr50"]

    if _matches_any(name, strong_names):
        score += 1.0
        reasons.append("known_strong_name")

    if atk < thresholds.min_attack_launches:
        hard_fails.append(f"too_few_attack_launches:{atk}")
    else:
        score += min(1.5, atk / max(thresholds.min_attack_launches, 1))
        reasons.append(f"attack_launches:{atk}")

    if cap_attack < thresholds.min_cap_attack:
        hard_fails.append(f"low_cap_attack:{cap_attack:.3f}")
    else:
        score += min(4.0, 4.0 * cap_attack / 0.60)
        reasons.append(f"cap_attack:{cap_attack:.3f}")

    if p16 is not None:
        if p16 < thresholds.min_planets16:
            hard_fails.append(f"low_planets16:{p16}")
        else:
            score += min(1.0, p16 / 3.0)
            reasons.append(f"p16:{p16}")
    if p50 is not None:
        if p50 < thresholds.min_planets50:
            hard_fails.append(f"low_planets50:{p50}")
        else:
            score += min(1.5, p50 / 8.0)
            reasons.append(f"p50:{p50}")

    if lost_cap > thresholds.max_lost_cap:
        hard_fails.append(f"high_lost_cap:{lost_cap:.2f}")
    else:
        score += max(0.0, 1.5 * (1.0 - lost_cap))
        reasons.append(f"lost_cap:{lost_cap:.2f}")

    if red > thresholds.max_redundant_early:
        hard_fails.append(f"high_redundant_early:{red:.2f}")
    else:
        score += max(0.0, 1.0 - red)
        reasons.append(f"redundant_early:{red:.2f}")

    if under > thresholds.max_underkill_early:
        hard_fails.append(f"high_underkill_early:{under:.2f}")
    else:
        score += max(0.0, 1.0 - under)
        reasons.append(f"underkill_early:{under:.2f}")

    if garr50 is not None:
        if 0.35 <= garr50 <= 0.75:
            score += 1.0
            reasons.append(f"garr50_elite_band:{garr50:.2f}")
        elif garr50 > 0.85:
            hard_fails.append(f"hoard_garr50:{garr50:.2f}")

    return round(score, 3), reasons, hard_fails


def score_replay(path: str, thresholds: QualityThresholds, winner_filters: list[str],
                 strong_names: list[str], require_known: bool = False) -> list[dict]:
    try:
        replay = json.loads(Path(path).read_text())
    except Exception as e:
        return [{
            "replay_path": path,
            "accepted": False,
            "score": 0.0,
            "hard_fails": [f"load_failed:{type(e).__name__}"],
        }]

    steps = replay.get("steps") or []
    names = _team_names(replay)
    winner = _strict_winner(replay.get("rewards"))
    if winner is None:
        return [{
            "replay_path": path,
            "accepted": False,
            "score": 0.0,
            "hard_fails": ["no_strict_winner"],
        }]
    name = names[winner] if winner < len(names) else f"player_{winner}"
    if not _matches_any(name, winner_filters):
        return [{
            "replay_path": path,
            "seat": winner,
            "name": name,
            "accepted": False,
            "score": 0.0,
            "hard_fails": ["winner_name_filtered"],
        }]
    if require_known and not _matches_any(name, strong_names):
        return [{
            "replay_path": path,
            "seat": winner,
            "name": name,
            "accepted": False,
            "score": 0.0,
            "hard_fails": ["not_known_strong_name"],
        }]

    conv = game_conversion(_attr_steps(steps), winner)
    metrics = _metrics_from_conversion(conv)
    score, reasons, hard_fails = score_metrics(metrics, name, thresholds, strong_names)
    accepted = score >= thresholds.min_score and not hard_fails
    return [{
        "replay_path": path,
        "replay_id": replay.get("id"),
        "seat": winner,
        "name": name,
        "score": score,
        "accepted": accepted,
        "reasons": reasons,
        "hard_fails": hard_fails,
        "metrics": metrics,
    }]


def score_replay_dirs(replay_dirs: list[str], glob_pattern: str,
                      thresholds: QualityThresholds, winner_filters: list[str],
                      strong_names: list[str], require_known: bool = False) -> list[dict]:
    paths: list[str] = []
    for replay_dir in replay_dirs:
        paths.extend(sorted(glob.glob(os.path.join(replay_dir, glob_pattern))))
    rows: list[dict] = []
    for path in paths:
        rows.extend(score_replay(path, thresholds, winner_filters, strong_names, require_known))
    rows.sort(key=lambda r: (r.get("accepted", False), r.get("score", 0.0)), reverse=True)
    return rows


def select_rows_for_samples(rows: list[dict], max_accepted_per_subject: int = 0) -> tuple[list[dict], dict]:
    """Select accepted rows for sample generation, optionally capping each subject.

    `rows` remains the full audit surface for score JSON. This selector only
    controls which accepted seats contribute supervised labels.
    """
    accepted = [r for r in rows if r.get("accepted")]
    if max_accepted_per_subject <= 0:
        return accepted, {
            "candidate_accepted_replays": len(accepted),
            "selected_accepted_replays": len(accepted),
            "skipped_by_subject_cap": 0,
            "selected_subjects": dict(Counter(r.get("name", "") for r in accepted).most_common(20)),
        }

    selected: list[dict] = []
    skipped = 0
    counts: Counter = Counter()
    for row in accepted:
        name = row.get("name", "")
        if counts[name] >= max_accepted_per_subject:
            skipped += 1
            continue
        counts[name] += 1
        selected.append(row)

    return selected, {
        "candidate_accepted_replays": len(accepted),
        "selected_accepted_replays": len(selected),
        "skipped_by_subject_cap": skipped,
        "max_accepted_per_subject": max_accepted_per_subject,
        "selected_subjects": dict(counts.most_common(20)),
    }


def _capture_steps_by_pid(steps: list, seat: int) -> dict[int, int]:
    prev_owner: dict[int, int] = {}
    captures: dict[int, int] = {}
    if steps and seat < len(steps[0]):
        obs0 = steps[0][seat].get("observation") or {}
        for p in obs0.get("planets") or []:
            prev_owner[int(p[0])] = int(p[1])
    for t in range(1, len(steps)):
        if seat >= len(steps[t]):
            continue
        obs = steps[t][seat].get("observation") or {}
        planets = obs.get("planets") or []
        for p in planets:
            pid, owner = int(p[0]), int(p[1])
            was = prev_owner.get(pid)
            if was is not None and was != seat and owner == seat:
                captures[pid] = t
            prev_owner[pid] = owner
    return captures


def _loss_steps_by_pid(steps: list, seat: int) -> dict[int, list[int]]:
    prev_owner: dict[int, int] = {}
    losses: dict[int, list[int]] = {}
    if steps and seat < len(steps[0]):
        obs0 = steps[0][seat].get("observation") or {}
        for p in obs0.get("planets") or []:
            prev_owner[int(p[0])] = int(p[1])
    for t in range(1, len(steps)):
        if seat >= len(steps[t]):
            continue
        obs = steps[t][seat].get("observation") or {}
        for p in obs.get("planets") or []:
            pid, owner = int(p[0]), int(p[1])
            was = prev_owner.get(pid)
            if was == seat and owner != seat:
                losses.setdefault(pid, []).append(t)
            prev_owner[pid] = owner
    return losses


def _add_threat_labels(sample: dict, obs: dict, seat: int,
                       losses: dict[int, list[int]],
                       horizon: int) -> tuple[int, int]:
    threat = torch.zeros_like(sample["fire_target"], dtype=torch.float32)
    mask = sample["slot_valid"].bool().clone()
    step = int(obs.get("step", 0))
    planets = obs.get("planets") or []
    owned_indices = sample["owned_indices"]
    positives = 0
    total = 0
    for slot in range(min(len(owned_indices), len(mask))):
        if not bool(mask[slot]):
            continue
        pidx = int(owned_indices[slot].item())
        if pidx >= len(planets) or int(planets[pidx][1]) != seat:
            mask[slot] = False
            continue
        total += 1
        pid = int(planets[pidx][0])
        if any(step < loss_step <= step + horizon for loss_step in losses.get(pid, [])):
            threat[slot] = 1.0
            positives += 1
    sample["threat_target"] = threat
    sample["threat_mask"] = mask
    return positives, total


def _fleet_points_at_planet(fleet: list, planet: list) -> bool:
    tx, ty, tr = float(planet[2]), float(planet[3]), float(planet[4])
    fx, fy, angle = float(fleet[2]), float(fleet[3]), float(fleet[4])
    c, s = math.cos(angle), math.sin(angle)
    vx, vy = tx - fx, ty - fy
    along = vx * c + vy * s
    perp = abs(vx * s - vy * c)
    return along > 0 and perp < tr + 1.5


def _fleet_speed(ships: int) -> float:
    if ships <= 0:
        return 1.0
    s = 1.0 + (6.0 - 1.0) * (math.log(max(ships, 1)) / math.log(1000.0)) ** 1.5
    return min(s, 6.0)


def _fleet_eta_to_planet(fleet: list, planet: list) -> int | None:
    tx, ty, tr = float(planet[2]), float(planet[3]), float(planet[4])
    fx, fy, angle = float(fleet[2]), float(fleet[3]), float(fleet[4])
    c, s = math.cos(angle), math.sin(angle)
    vx, vy = tx - fx, ty - fy
    along = vx * c + vy * s
    perp = abs(vx * s - vy * c)
    if along <= 0 or perp >= tr + 1.5:
        return None
    speed = _fleet_speed(int(fleet[6]) if len(fleet) > 6 else 1)
    return max(1, int(math.ceil(max(0.0, along - tr) / speed)))


def _nearest_enemy_dist(planet: list, planets: list, seat: int) -> float:
    enemies = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != seat]
    if not enemies:
        return float("inf")
    px, py = float(planet[2]), float(planet[3])
    return min(math.hypot(px - float(e[2]), py - float(e[3])) for e in enemies)


def _synthetic_defense_moves(obs: dict, moves: list, seat: int, stats: Counter,
                             garrison_floor: int = 10,
                             min_need: int = 5,
                             eligible_target_pids: set[int] | None = None) -> list:
    planets = obs.get("planets") or []
    own_planets = [p for p in planets if int(p[1]) == seat]
    if len(own_planets) < 2:
        return []

    used_sources = {int(m[0]) for m in moves if isinstance(m, list) and len(m) >= 1}
    synthetic: list[list] = []
    for target in own_planets:
        target_pid = int(target[0])
        if eligible_target_pids is not None and target_pid not in eligible_target_pids:
            stats["synthetic_defense_skipped_ineligible_target"] += 1
            continue
        if target_pid in used_sources:
            continue

        inbound_ships = 0
        min_eta: int | None = None
        for fleet in obs.get("fleets") or []:
            if int(fleet[1]) == seat:
                continue
            eta = _fleet_eta_to_planet(fleet, target)
            if eta is None:
                continue
            inbound_ships += int(fleet[6]) if len(fleet) > 6 else 0
            min_eta = eta if min_eta is None else min(min_eta, eta)
        if inbound_ships <= 0 or min_eta is None:
            continue

        projected_garrison = float(target[5]) + float(target[6]) * min_eta
        need = int(math.ceil(inbound_ships + min_need - projected_garrison))
        if need < min_need:
            stats["synthetic_defense_skipped_sufficient_garrison"] += 1
            continue

        target_enemy_dist = _nearest_enemy_dist(target, planets, seat)
        candidates = []
        for src in own_planets:
            src_pid = int(src[0])
            if src_pid == target_pid or src_pid in used_sources:
                continue
            src_ships = int(src[5])
            sendable = src_ships - garrison_floor
            if sendable < min_need:
                continue
            # Match the disciplined eval decode: support should move from rear
            # planets toward a more forward threatened planet.
            if _nearest_enemy_dist(src, planets, seat) <= target_enemy_dist:
                continue
            candidates.append((sendable, src))

        if not candidates:
            stats["synthetic_defense_no_source"] += 1
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        sendable, src = candidates[0]
        ships = min(sendable, max(need, min_need))
        angle = _target_intercept_angle(src, target, ships, obs)
        synthetic.append([int(src[0]), angle, int(ships)])
        used_sources.add(int(src[0]))
        stats["synthetic_defense_moves"] += 1
        stats["synthetic_defense_ships"] += int(ships)

    return synthetic


def _contest_reasons(obs: dict, seat: int, captures: dict[int, int],
                     contest_window: int) -> list[str]:
    planets = obs.get("planets") or []
    step = int(obs.get("step", 0))
    reasons: list[str] = []
    if contest_window > 0:
        for p in planets:
            pid = int(p[0])
            if int(p[1]) == seat and pid in captures:
                age = step - captures[pid]
                if 0 <= age <= contest_window:
                    reasons.append("recent_capture")
                    break

    own_planets = [p for p in planets if int(p[1]) == seat]
    for fleet in obs.get("fleets") or []:
        if int(fleet[1]) == seat:
            continue
        if any(_fleet_points_at_planet(fleet, p) for p in own_planets):
            reasons.append("enemy_inbound")
            break
    return reasons


def _threatened_owned_pids(obs: dict, seat: int) -> set[int]:
    planets = obs.get("planets") or []
    own_planets = [p for p in planets if int(p[1]) == seat]
    threatened: set[int] = set()
    for fleet in obs.get("fleets") or []:
        if int(fleet[1]) == seat:
            continue
        for p in own_planets:
            if _fleet_points_at_planet(fleet, p):
                threatened.add(int(p[0]))
    return threatened


def _held_recent_capture_pids(obs: dict, seat: int, captures: dict[int, int],
                              losses: dict[int, list[int]],
                              recent_window: int, hold_success_horizon: int,
                              stats: Counter) -> set[int]:
    if recent_window <= 0:
        return set()
    step = int(obs.get("step", 0))
    held: set[int] = set()
    for p in obs.get("planets") or []:
        pid = int(p[0])
        if int(p[1]) != seat or pid not in captures:
            continue
        age = step - captures[pid]
        if age < 0 or age > recent_window:
            continue
        stats["held_capture_recent_owned"] += 1
        future_losses = losses.get(pid, [])
        if hold_success_horizon > 0:
            if any(step < loss_step <= step + hold_success_horizon for loss_step in future_losses):
                stats["held_capture_rejected_future_loss"] += 1
                continue
        held.add(pid)
    return held


def _held_capture_moves(obs: dict, moves: list, held_pids: set[int],
                        stats: Counter) -> list:
    if not moves or not held_pids:
        return []
    planets = obs.get("planets") or []
    byid = {int(p[0]): p for p in planets}
    kept = []
    for move in moves:
        src = byid.get(int(move[0]))
        if src is None:
            continue
        source_held = int(src[0]) in held_pids
        target_held = False
        target = _resolve_launch_target(planets, src, float(move[1]))
        if target is not None and int(target[0]) in held_pids:
            target_held = True
        if source_held or target_held:
            kept.append(move)
            if source_held:
                stats["held_capture_source_moves"] += 1
            if target_held:
                stats["held_capture_target_moves"] += 1
    return kept


def _answer_inbound_moves(obs: dict, moves: list, threatened_pids: set[int],
                          stats: Counter) -> list:
    if not moves or not threatened_pids:
        return []
    planets = obs.get("planets") or []
    byid = {int(p[0]): p for p in planets}
    kept = []
    for move in moves:
        src = byid.get(int(move[0]))
        if src is None:
            continue
        source_threatened = int(src[0]) in threatened_pids
        target_threatened = False
        target = _resolve_launch_target(planets, src, float(move[1]))
        if target is not None and int(target[0]) in threatened_pids:
            target_threatened = True
        if source_threatened or target_threatened:
            kept.append(move)
            if source_threatened:
                stats["answer_source_threatened"] += 1
            if target_threatened:
                stats["answer_target_threatened"] += 1
    return kept


def build_samples_from_rows(rows: list[dict], steps_min: int, steps_max: int,
                            noop_keep_prob: float, fire_repeat: int,
                            reinforce_repeat: int, contest_window: int,
                            answer_inbound_only: bool,
                            answer_inbound_repeat: int = 1,
                            synthetic_defense_repeat: int = 1,
                            threat_horizon: int = 0,
                            held_capture_window: int = 0,
                            hold_success_horizon: int = 0,
                            held_capture_repeat: int = 1,
                            held_capture_only: bool = False,
                            max_samples_per_subject: int = 0,
                            seed: int = 0) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    samples = []
    stats: Counter = Counter()
    subjects: Counter = Counter()
    subject_samples: Counter = Counter()
    subject_decision_samples: Counter = Counter()

    for row in rows:
        if not row.get("accepted"):
            continue
        replay = json.loads(Path(row["replay_path"]).read_text())
        steps = replay.get("steps") or []
        seat = int(row["seat"])
        subject_name = row["name"]
        subjects[subject_name] += 1
        captures = _capture_steps_by_pid(steps, seat) if contest_window > 0 or held_capture_window > 0 else {}
        losses = _loss_steps_by_pid(steps, seat) if threat_horizon > 0 or hold_success_horizon > 0 else {}
        t_start = max(1, steps_min)
        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)

        for t in range(t_start, t_end):
            if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
                stats["missing_agent_step"] += 1
                continue
            obs = steps[t - 1][seat].get("observation")
            if not obs or "planets" not in obs:
                stats["missing_obs"] += 1
                continue
            contest_reasons = _contest_reasons(obs, seat, captures, contest_window)
            if contest_window > 0:
                if not contest_reasons:
                    stats["frames_skipped_not_contest"] += 1
                    continue
                stats["contest_frames_seen"] += 1
                for reason in set(contest_reasons):
                    stats[f"contest_{reason}"] += 1
            moves = _moves(steps[t][seat].get("action") or [])
            answer_moves = []
            if answer_inbound_only or answer_inbound_repeat > 1:
                threatened_pids = _threatened_owned_pids(obs, seat)
                if not threatened_pids:
                    if answer_inbound_only:
                        stats["frames_skipped_no_inbound_threat"] += 1
                        continue
                else:
                    stats["answer_inbound_frames_seen"] += 1
                    stats["answer_threatened_planets"] += len(threatened_pids)
                    answer_moves = _answer_inbound_moves(obs, moves, threatened_pids, stats)
                    if answer_inbound_only:
                        moves = answer_moves
                        if not moves:
                            stats["frames_skipped_no_answer_move"] += 1
                            continue
                        stats["answer_frames_kept"] += 1
                    elif answer_moves:
                        stats["answer_frames_weighted"] += 1
                        stats["answer_moves_weighted"] += len(answer_moves)
            held_capture_moves = []
            if held_capture_only or held_capture_repeat > 1:
                held_pids = _held_recent_capture_pids(
                    obs, seat, captures, losses,
                    held_capture_window, hold_success_horizon, stats,
                )
                if not held_pids:
                    if held_capture_only:
                        stats["frames_skipped_no_held_capture"] += 1
                        continue
                else:
                    stats["held_capture_frames_seen"] += 1
                    stats["held_capture_planets"] += len(held_pids)
                    held_capture_moves = _held_capture_moves(obs, moves, held_pids, stats)
                    if held_capture_only:
                        moves = held_capture_moves
                        if not moves:
                            stats["frames_skipped_no_held_capture_move"] += 1
                            continue
                        stats["held_capture_frames_kept"] += 1
                    elif held_capture_moves:
                        stats["held_capture_frames_weighted"] += 1
                        stats["held_capture_moves_weighted"] += len(held_capture_moves)
            synthetic_moves = []
            if synthetic_defense_repeat > 1:
                synthetic_moves = _synthetic_defense_moves(obs, moves, seat, stats)
                if synthetic_moves:
                    moves = moves + synthetic_moves
                    stats["synthetic_defense_frames"] += 1
            if moves:
                stats["decision_frames_seen"] += 1
                repeat = max(1, fire_repeat)
                if answer_moves and not answer_inbound_only:
                    repeat = max(repeat, answer_inbound_repeat)
                if held_capture_moves and not held_capture_only:
                    repeat = max(repeat, held_capture_repeat)
                if synthetic_moves:
                    repeat = max(repeat, synthetic_defense_repeat)
            else:
                stats["noop_frames_seen"] += 1
                if rng.random() >= noop_keep_prob:
                    continue
                stats["noop_frames_kept"] += 1
                repeat = 1

            sample = trajectory_to_training_sample({
                "obs": _normalize_obs(obs, seat, t - 1),
                "action": moves,
            })
            if sample is None:
                stats["sample_build_failed"] += 1
                continue
            if threat_horizon > 0:
                pos, total = _add_threat_labels(sample, obs, seat, losses, threat_horizon)
                stats["threat_slots"] += total
                stats["threat_pos_slots"] += pos
                if pos:
                    stats["threat_positive_frames"] += 1
            reinforce_labels = _reinforce_label_count(sample, obs, seat)
            if reinforce_labels:
                stats["reinforce_frames_seen"] += 1
                stats["reinforce_labels_seen"] += reinforce_labels
                repeat = max(repeat, reinforce_repeat)
            added = 0
            for _ in range(repeat):
                if max_samples_per_subject > 0 and subject_samples[subject_name] >= max_samples_per_subject:
                    stats["samples_skipped_subject_sample_cap"] += 1
                    continue
                samples.append(_clone_sample(sample))
                subject_samples[subject_name] += 1
                added += 1
                _add_sample_stats(stats, sample)
                if reinforce_labels:
                    stats["reinforce_labels"] += reinforce_labels
            if moves:
                stats["decision_samples_added"] += added
                subject_decision_samples[subject_name] += added
                if reinforce_labels:
                    stats["reinforce_samples_added"] += added
            else:
                stats["noop_samples_added"] += added

    summary = {
        "accepted_replays": sum(1 for r in rows if r.get("accepted")),
        "config": {
            "steps_min": steps_min,
            "steps_max": steps_max,
            "noop_keep_prob": noop_keep_prob,
            "fire_repeat": fire_repeat,
            "reinforce_repeat": reinforce_repeat,
            "contest_window": contest_window,
            "answer_inbound_only": answer_inbound_only,
            "answer_inbound_repeat": answer_inbound_repeat,
            "synthetic_defense_repeat": synthetic_defense_repeat,
            "threat_horizon": threat_horizon,
            "held_capture_window": held_capture_window,
            "hold_success_horizon": hold_success_horizon,
            "held_capture_repeat": held_capture_repeat,
            "held_capture_only": held_capture_only,
            "max_samples_per_subject": max_samples_per_subject,
            "seed": seed,
        },
        "subjects": dict(subjects.most_common(20)),
        "subject_samples": dict(subject_samples.most_common(20)),
        "subject_decision_samples": dict(subject_decision_samples.most_common(20)),
        "samples": len(samples),
        "stats": dict(stats),
    }
    if stats["valid_slots"]:
        summary["fire_slot_rate"] = stats["fire_slots"] / stats["valid_slots"]
    if stats["decision_frames_seen"] + stats["noop_frames_seen"]:
        summary["decision_frame_rate_seen"] = stats["decision_frames_seen"] / (
            stats["decision_frames_seen"] + stats["noop_frames_seen"]
        )
    if samples:
        summary["decision_sample_share"] = stats["decision_samples_added"] / len(samples)
    if stats["threat_slots"]:
        summary["threat_pos_rate"] = stats["threat_pos_slots"] / stats["threat_slots"]
    return samples, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-dir", action="append", required=True)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--winner-name", action="append", default=[])
    ap.add_argument("--strong-name", action="append", default=DEFAULT_STRONG_NAMES)
    ap.add_argument("--require-known-winner", action="store_true")
    ap.add_argument("--min-attack-launches", type=int, default=QualityThresholds.min_attack_launches)
    ap.add_argument("--min-cap-attack", type=float, default=QualityThresholds.min_cap_attack)
    ap.add_argument("--min-planets16", type=int, default=QualityThresholds.min_planets16)
    ap.add_argument("--min-planets50", type=int, default=QualityThresholds.min_planets50)
    ap.add_argument("--max-lost-cap", type=float, default=QualityThresholds.max_lost_cap)
    ap.add_argument("--max-redundant-early", type=float, default=QualityThresholds.max_redundant_early)
    ap.add_argument("--max-underkill-early", type=float, default=QualityThresholds.max_underkill_early)
    ap.add_argument("--min-score", type=float, default=QualityThresholds.min_score)
    ap.add_argument("--scores-out", required=True)
    ap.add_argument("--samples-out", default="")
    ap.add_argument("--max-accepted-per-subject", type=int, default=0,
                    help="If >0, sample generation uses only the top N accepted "
                         "replays per exact winner name. Score JSON still keeps "
                         "the full accepted/rejected audit rows.")
    ap.add_argument("--max-samples-per-subject", type=int, default=0,
                    help="If >0, cap final repeated training samples per exact "
                         "winner name after frame weighting/repeats.")
    ap.add_argument("--steps-min", type=int, default=1)
    ap.add_argument("--steps-max", type=int, default=0)
    ap.add_argument("--noop-keep-prob", type=float, default=0.05)
    ap.add_argument("--fire-repeat", type=int, default=2)
    ap.add_argument("--reinforce-repeat", type=int, default=1)
    ap.add_argument("--contest-window", type=int, default=0,
                    help="If >0, keep only frames with a captured planet this many "
                         "steps old or an enemy fleet inbound to an owned planet.")
    ap.add_argument("--answer-inbound-only", action="store_true",
                    help="Keep only enemy-inbound frames where the teacher move starts "
                         "from or targets a threatened owned planet; unrelated moves "
                         "are removed from the BC label.")
    ap.add_argument("--answer-inbound-repeat", type=int, default=1,
                    help="If >1, upweight frames where at least one teacher move starts "
                         "from or targets a threatened owned planet, while preserving "
                         "the full action label. This is the soft alternative to "
                         "--answer-inbound-only.")
    ap.add_argument("--synthetic-defense-repeat", type=int, default=1,
                    help="If >1, append synthetic reinforce labels from rear owned "
                         "planets to threatened owned planets with insufficient "
                         "projected garrison, then repeat those frames.")
    ap.add_argument("--threat-horizon", type=int, default=0,
                    help="If >0, add per-owned-slot label: planet is lost within "
                         "this many future steps.")
    ap.add_argument("--held-capture-window", type=int, default=0,
                    help="If >0, identify owned planets captured within this many "
                         "steps and upweight/filter replay moves involving them.")
    ap.add_argument("--hold-success-horizon", type=int, default=0,
                    help="With --held-capture-window, reject recent captured planets "
                         "that are lost within this many future steps. 0 only checks "
                         "that the planet is a recent owned capture at the current step.")
    ap.add_argument("--held-capture-repeat", type=int, default=1,
                    help="If >1, upweight frames where at least one replay move starts "
                         "from or targets a held recent capture, preserving the full "
                         "action label.")
    ap.add_argument("--held-capture-only", action="store_true",
                    help="Keep only moves that start from or target a held recent "
                         "capture. This is the hard-filter version of "
                         "--held-capture-repeat.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thresholds = QualityThresholds(
        min_attack_launches=args.min_attack_launches,
        min_cap_attack=args.min_cap_attack,
        min_planets16=args.min_planets16,
        min_planets50=args.min_planets50,
        max_lost_cap=args.max_lost_cap,
        max_redundant_early=args.max_redundant_early,
        max_underkill_early=args.max_underkill_early,
        min_score=args.min_score,
    )
    rows = score_replay_dirs(
        args.replay_dir,
        args.glob,
        thresholds,
        args.winner_name,
        args.strong_name,
        args.require_known_winner,
    )
    payload = {
        "config": {
            "replay_dirs": args.replay_dir,
            "glob": args.glob,
            "winner_names": args.winner_name,
            "strong_names": args.strong_name,
            "require_known_winner": args.require_known_winner,
            "max_accepted_per_subject": args.max_accepted_per_subject,
            "max_samples_per_subject": args.max_samples_per_subject,
            "thresholds": asdict(thresholds),
        },
        "total_rows": len(rows),
        "accepted_rows": sum(1 for r in rows if r.get("accepted")),
        "accepted_subjects": dict(Counter(r["name"] for r in rows if r.get("accepted")).most_common(20)),
        "rows": rows,
    }

    if args.samples_out:
        selected_rows, selection_summary = select_rows_for_samples(
            rows,
            max_accepted_per_subject=args.max_accepted_per_subject,
        )
        samples, sample_summary = build_samples_from_rows(
            selected_rows,
            steps_min=args.steps_min,
            steps_max=args.steps_max,
            noop_keep_prob=args.noop_keep_prob,
            fire_repeat=args.fire_repeat,
            reinforce_repeat=args.reinforce_repeat,
            contest_window=args.contest_window,
            answer_inbound_only=args.answer_inbound_only,
            answer_inbound_repeat=args.answer_inbound_repeat,
            synthetic_defense_repeat=args.synthetic_defense_repeat,
            threat_horizon=args.threat_horizon,
            held_capture_window=args.held_capture_window,
            hold_success_horizon=args.hold_success_horizon,
            held_capture_repeat=args.held_capture_repeat,
            held_capture_only=args.held_capture_only,
            max_samples_per_subject=args.max_samples_per_subject,
            seed=args.seed,
        )
        out = Path(args.samples_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(samples, f)
        sample_summary["samples_out"] = str(out)
        sample_summary["selection"] = selection_summary
        payload["sample_summary"] = sample_summary

    scores_out = Path(args.scores_out)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    scores_out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "total_rows": payload["total_rows"],
        "accepted_rows": payload["accepted_rows"],
        "accepted_subjects": payload["accepted_subjects"],
        "scores_out": str(scores_out),
        "sample_summary": payload.get("sample_summary"),
    }, indent=2))


if __name__ == "__main__":
    main()
