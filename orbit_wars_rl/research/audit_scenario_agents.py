"""Run external agents on the tiny scenario-curriculum boards.

This is a diagnostic, not a normal eval panel. It asks whether a heuristic agent
can solve the handcrafted scenario when controlling the advantaged seat.

Example:
  /Users/saheb/home/.venv/bin/python orbit_wars_rl/audit_scenario_agents.py \
    --agent opponents/candidate_ajay_1200.py --games 64 --print-examples 3
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

from orbit_wars_rl.config import Config  # noqa: E402
from orbit_wars_rl.eval import build_agent_fn, load_checkpoint  # noqa: E402
from orbit_wars_rl.model import EntityTransformer  # noqa: E402
from orbit_wars_rl.torch_env import VecTorchEnv, to_legacy_obs  # noqa: E402
from orbit_wars_rl.train_torch import _heuristic_moves_to_action_tensor  # noqa: E402
from orbit_wars_rl.features import extract_features  # noqa: E402
from orbit_wars_rl.action_mask import compute_action_masks, _ship_bin_to_count  # noqa: E402


SCENARIOS = ("agg_attack", "stage_attack", "hold_under_peel")


def _selected_scenarios(name: str) -> tuple[str, ...]:
    if name == "all":
        return SCENARIOS
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}")
    return (name,)


def _load_agent(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import agent from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reset_agent(mod) -> None:
    runtime = getattr(mod, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        runtime.reset()


def _planet_summary(env: VecTorchEnv, env_i: int, target: int, adv: int) -> str:
    p = env.planets[env_i].detach().cpu()
    alive = env.planet_alive[env_i].detach().cpu()
    rows = []
    for i in range(p.shape[0]):
        if not bool(alive[i]):
            continue
        owner = int(p[i, 1].item())
        if owner == adv or i == target:
            rows.append(
                f"{i}:own{owner}:ships{p[i,5].item():.0f}:prod{p[i,6].item():.0f}"
            )
    return " ".join(rows)


def _load_checkpoint_model(path: str, device: torch.device):
    cfg = Config()
    cfg.device = str(device)
    state_dict, action_decode = load_checkpoint(path, cfg)
    target_decode = action_decode == "target"
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(getattr(cfg.model, "reinforce_gate_min_planets", 0))
    model.reinforce_forward_only = bool(getattr(cfg.model, "reinforce_forward_only", False))
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.reinforce_garrison_floor = float(getattr(cfg.model, "reinforce_garrison_floor", 0.0))
    model.sufficient_commit_factor = float(getattr(cfg.model, "sufficient_commit_factor", 0.0))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"target_head.weight", "target_head.bias"}
    bad_missing = [k for k in missing if k not in allowed_missing]
    bad_unexpected = [k for k in unexpected if not k.startswith("value_pp_")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
    model.eval()
    return model, cfg, target_decode


def _load_checkpoint_agent(path: str, device: torch.device, *, sample: bool = False):
    model, cfg, target_decode = _load_checkpoint_model(path, device)
    return build_agent_fn(
        model,
        device,
        fire_threshold=0.5,
        sample=sample,
        ship_bin_mode=cfg.model.ship_bin_mode,
        target_decode=target_decode,
    )


def _make_env(scenario: str, seed: int, deadline: int) -> VecTorchEnv:
    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        action_decode="target",
        allow_reinforce=True,
        scenario_curriculum=scenario,
        scenario_fraction=1.0,
        scenario_deadline=deadline,
    )
    env.reset(seeds=[seed])
    return env


def _run_one(agent_fn, scenario: str, seed: int, deadline: int, print_example: bool) -> tuple[bool, int]:
    env = _make_env(scenario, seed, deadline)
    adv = int(env.scenario_adv_player[0].item())
    target = int(env.scenario_target[0].item())
    initial_board = _planet_summary(env, 0, target, adv)

    first_moves = None
    first_targets = []
    done = torch.tensor([False])
    steps = 0
    while not bool(done[0]) and steps <= deadline + 5:
        obs = to_legacy_obs(env, env_idx=0, player=adv)
        moves = agent_fn(obs) or []
        if first_moves is None:
            first_moves = moves
        act, cont = _heuristic_moves_to_action_tensor([moves], env, adv, env.device)
        state, rewards, done = env.step({adv: act}, angle_overrides={adv: cont})
        if steps == 0:
            tgt_idx = env._fleet_target_idx()[0].detach().cpu()
            alive = env.fleet_alive[0].detach().cpu()
            owners = env.fleets[0, :, 1].detach().cpu()
            ships = env.fleets[0, :, 6].detach().cpu()
            for f in range(alive.shape[0]):
                if bool(alive[f]) and int(owners[f].item()) == adv:
                    first_targets.append((int(tgt_idx[f].item()), int(round(float(ships[f].item())))))
        steps += 1

    success = bool(env._last_scenario_success[0].item()) if bool(done[0]) else False
    if print_example:
        print(f"\n[{scenario} seed={seed}] adv={adv} target={target} success={success} steps={steps}")
        print(f"  board: {initial_board}")
        print(f"  first_moves={first_moves}")
        print(f"  first_fleet_targets={first_targets}")
    return success, steps


def _ship_bin_at_least(ships: int, max_ships: int, mode: str = "absolute") -> int:
    if mode == "fraction":
        best = 0
        for b in range(10):
            if _ship_bin_to_count(b, max_ships, mode=mode) >= ships:
                return b
            best = b
        return best
    best = 0
    for b in range(32):
        best = b
        if _ship_bin_to_count(b, max_ships, mode=mode) >= ships:
            return b
    return best


def _slot_for_pid(env: VecTorchEnv, player: int, pid: int) -> int | None:
    owned_idx, slot_valid = env.owned_indices_for(player)
    for slot in range(owned_idx.shape[1]):
        if not bool(slot_valid[0, slot]):
            continue
        pidx = int(owned_idx[0, slot].item())
        if int(env.planets[0, pidx, 0].item()) == pid:
            return slot
    return None


def _oracle_action(env: VecTorchEnv, player: int, scenario: str) -> torch.Tensor:
    action = torch.zeros((1, 16, 4), dtype=torch.long)
    target = int(env.scenario_target[0].item())
    if scenario == "agg_attack":
        plan = [(0, 50), (1, 50)]
    elif scenario == "stage_attack":
        plan = [(0, 42)]
    elif scenario == "hold_under_peel":
        plan = [(0, 42), (1, 42)]
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    for pid, ships in plan:
        slot = _slot_for_pid(env, player, pid)
        if slot is None:
            raise RuntimeError(f"oracle source pid={pid} not owned by player={player}")
        source_pidx = int(env.owned_indices_for(player)[0][0, slot].item())
        max_ships = int(env.planets[0, source_pidx, 5].item())
        action[0, slot, 0] = 1
        action[0, slot, 1] = 0
        action[0, slot, 2] = _ship_bin_at_least(ships, max_ships)
        action[0, slot, 3] = target
    return action


def _run_oracle_one(scenario: str, seed: int, deadline: int, print_example: bool) -> tuple[bool, int]:
    env = _make_env(scenario, seed, deadline)
    adv = int(env.scenario_adv_player[0].item())
    target = int(env.scenario_target[0].item())
    initial_board = _planet_summary(env, 0, target, adv)
    first_action = _oracle_action(env, adv, scenario)
    fired = []
    owned_idx, _ = env.owned_indices_for(adv)
    for slot in range(first_action.shape[1]):
        if int(first_action[0, slot, 0].item()) == 0:
            continue
        pidx = int(owned_idx[0, slot].item())
        fired.append((
            int(env.planets[0, pidx, 0].item()),
            int(first_action[0, slot, 3].item()),
            int(_ship_bin_to_count(int(first_action[0, slot, 2].item()), int(env.planets[0, pidx, 5].item()))),
        ))
    state, rewards, done = env.step({adv: first_action})
    steps = 1
    while not bool(done[0]) and steps <= deadline + 5:
        noop = torch.zeros((1, 16, 4), dtype=torch.long)
        state, rewards, done = env.step({adv: noop})
        steps += 1
    success = bool(env._last_scenario_success[0].item()) if bool(done[0]) else False
    if print_example:
        print(f"\n[oracle {scenario} seed={seed}] adv={adv} target={target} success={success} steps={steps}")
        print(f"  board: {initial_board}")
        print(f"  fired={fired}")
    return success, steps


def _run_noop_one(scenario: str, seed: int, deadline: int, print_example: bool) -> tuple[bool, int]:
    env = _make_env(scenario, seed, deadline)
    adv = int(env.scenario_adv_player[0].item())
    target = int(env.scenario_target[0].item())
    initial_board = _planet_summary(env, 0, target, adv)
    noop = torch.zeros((1, 16, 4), dtype=torch.long)
    done = torch.tensor([False])
    steps = 0
    while not bool(done[0]) and steps <= deadline + 5:
        _, _, done = env.step({adv: noop})
        steps += 1
    success = bool(env._last_scenario_success[0].item()) if bool(done[0]) else False
    if print_example:
        print(f"\n[noop {scenario} seed={seed}] adv={adv} target={target} success={success} steps={steps}")
        print(f"  board: {initial_board}")
    return success, steps


def _legalized_target_logits(outputs, masks, env: VecTorchEnv, player: int) -> torch.Tensor:
    logits = outputs["target_logits"].detach().cpu().clone()
    planets = env.planets[0].detach().cpu()
    alive = env.planet_alive[0].detach().cpu()
    owned_indices = masks["owned_indices"].detach().cpu()
    owned_count = int(masks["owned_count"])
    for slot in range(min(owned_count, logits.shape[1])):
        pidx = int(owned_indices[slot].item())
        for tidx in range(min(planets.shape[0], logits.shape[-1])):
            if not bool(alive[tidx]):
                logits[0, slot, tidx] = -1e9
                continue
            if tidx == pidx:
                logits[0, slot, tidx] = -1e9
    return logits


def _head_audit_one(model: EntityTransformer, cfg: Config, scenario: str,
                    seed: int, deadline: int, device: torch.device, topk: int) -> None:
    env = _make_env(scenario, seed, deadline)
    adv = int(env.scenario_adv_player[0].item())
    target = int(env.scenario_target[0].item())
    obs = to_legacy_obs(env, env_idx=0, player=adv)
    features = extract_features(obs, adv, num_players=2)
    masks = compute_action_masks(obs, adv)
    with torch.no_grad():
        outputs = model(
            features["planet_features"].unsqueeze(0).to(device),
            features["fleet_features"].unsqueeze(0).to(device),
            features["global_features"].unsqueeze(0).to(device),
            features["planet_mask"].unsqueeze(0).to(device),
            features["fleet_mask"].unsqueeze(0).to(device),
            fire_mask=masks["fire_mask"].to(device),
            angle_mask=masks["angle_mask"].to(device),
            slot_valid=masks["slot_valid"].to(device),
            owned_indices=masks["owned_indices"].to(device),
            owned_count=masks["owned_count"],
            pairwise_features=features["pairwise_features"].unsqueeze(0).to(device)
            if "pairwise_features" in features else None,
        )
    target_logits = _legalized_target_logits(outputs, masks, env, adv)
    fire_p_by_target = torch.sigmoid(outputs["fire_logits"].detach().cpu())[0]
    ship_logits_by_target = outputs["ship_logits"].detach().cpu()[0]
    probs = torch.softmax(target_logits[0], dim=-1)
    owned_indices = masks["owned_indices"].detach().cpu()
    owned_count = int(masks["owned_count"])
    planets = env.planets[0].detach().cpu()
    print(f"\n[head {scenario} seed={seed}] adv={adv} target={target}")
    print(f"  board: {_planet_summary(env, 0, target, adv)}")
    for slot in range(min(owned_count, target_logits.shape[1])):
        pidx = int(owned_indices[slot].item())
        pid = int(planets[pidx, 0].item())
        if pid not in (0, 1, 5):
            continue
        row = target_logits[0, slot, : planets.shape[0]]
        top_vals, top_idx = torch.topk(row, k=min(topk, row.numel()))
        target_order = torch.argsort(row, descending=True)
        target_rank = int((target_order == target).nonzero(as_tuple=False)[0].item()) + 1
        fire_p_target = float(fire_p_by_target[slot, target].item())
        ship_bin = int(torch.argmax(ship_logits_by_target[slot, target]).item())
        max_ships = int(planets[pidx, 5].item())
        decoded = _ship_bin_to_count(ship_bin, max_ships, mode=cfg.model.ship_bin_mode)
        top_desc = ", ".join(
            f"{int(t.item())}:p{float(probs[slot, int(t)].item()):.2f}"
            for t in top_idx
        )
        print(
            f"  src_pid={pid} slot={slot} ships={max_ships} "
            f"fire_p@target={fire_p_target:.3f} "
            f"target_rank={target_rank} target_p={float(probs[slot, target].item()):.3f} "
            f"ship_bin={ship_bin} ship_count={decoded} top{topk}=[{top_desc}]"
        )


def _run_oracle(args) -> None:
    print(f"oracle games={args.games} deadline={args.deadline}")
    for scenario in _selected_scenarios(args.scenario):
        wins = 0
        steps_sum = 0
        examples_left = args.print_examples
        for i in range(args.games):
            seed = args.seed_start + 10000 * SCENARIOS.index(scenario) + i
            success, steps = _run_oracle_one(scenario, seed, args.deadline, examples_left > 0)
            wins += int(success)
            steps_sum += steps
            examples_left -= 1
        print(
            f"{scenario}: oracle success {wins}/{args.games} = {wins / max(args.games, 1):.1%} "
            f"avg_steps={steps_sum / max(args.games, 1):.1f}"
        )


def _run_noop(args) -> None:
    print(f"noop games={args.games} deadline={args.deadline}")
    for scenario in _selected_scenarios(args.scenario):
        wins = 0
        steps_sum = 0
        examples_left = args.print_examples
        for i in range(args.games):
            seed = args.seed_start + 10000 * SCENARIOS.index(scenario) + i
            success, steps = _run_noop_one(scenario, seed, args.deadline, examples_left > 0)
            wins += int(success)
            steps_sum += steps
            examples_left -= 1
        print(
            f"{scenario}: noop success {wins}/{args.games} = {wins / max(args.games, 1):.1%} "
            f"avg_steps={steps_sum / max(args.games, 1):.1f}"
        )


def _run_head(args) -> None:
    if not args.checkpoint:
        raise SystemExit("--mode head requires --checkpoint")
    device = torch.device(args.device)
    model, cfg, _ = _load_checkpoint_model(args.checkpoint, device)
    print(f"head-audit checkpoint={args.checkpoint} games={args.games} deadline={args.deadline}")
    for scenario in _selected_scenarios(args.scenario):
        for i in range(args.games):
            seed = args.seed_start + 10000 * SCENARIOS.index(scenario) + i
            _head_audit_one(model, cfg, scenario, seed, args.deadline, device, args.topk)


def _run_success(args, *, sample: bool = False) -> None:
    if args.checkpoint:
        device = torch.device(args.device)
        agent_fn = _load_checkpoint_agent(args.checkpoint, device, sample=sample)
        label = f"checkpoint={args.checkpoint}"
    else:
        agent_mod = _load_agent(args.agent)
        agent_fn = agent_mod.agent
        _reset_agent(agent_mod)
        label = f"agent={args.agent}"
    mode = "sample" if sample else "deterministic"
    print(f"{label} mode={mode} games={args.games} deadline={args.deadline}")
    for scenario in _selected_scenarios(args.scenario):
        if not args.checkpoint:
            _reset_agent(agent_mod)
        wins = 0
        steps_sum = 0
        examples_left = args.print_examples
        for i in range(args.games):
            seed = args.seed_start + 10000 * SCENARIOS.index(scenario) + i
            if not args.checkpoint:
                _reset_agent(agent_mod)
            success, steps = _run_one(
                agent_fn, scenario, seed, args.deadline, examples_left > 0
            )
            wins += int(success)
            steps_sum += steps
            examples_left -= 1
        print(
            f"{scenario}: success {wins}/{args.games} = {wins / max(args.games, 1):.1%} "
            f"avg_steps={steps_sum / max(args.games, 1):.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="opponents/candidate_ajay_1200.py")
    ap.add_argument("--checkpoint", default="", help="Run a trained torch checkpoint instead of --agent")
    ap.add_argument("--mode", choices=("run", "head", "oracle", "noop", "sample", "all"), default="run")
    ap.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--deadline", type=int, default=20)
    ap.add_argument("--print-examples", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if args.mode in ("oracle", "all"):
        _run_oracle(args)
    if args.mode in ("noop", "all"):
        _run_noop(args)
    if args.mode in ("head", "all"):
        _run_head(args)
    if args.mode == "sample":
        _run_success(args, sample=True)
    elif args.mode == "run":
        _run_success(args, sample=False)
    elif args.mode == "all":
        _run_success(args, sample=True)


if __name__ == "__main__":
    main()
