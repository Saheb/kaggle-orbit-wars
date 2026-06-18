"""Stage-A parity harness for Phase 4 per-target fire/ship heads.

Checks three things against a real checkpoint:
1. `load_state_dict(strict=False)` misses only the new per-target head params.
2. Residual-zero parity: gathered chosen-target fire/ship logits match the legacy
   slot heads on real seeded opening observations.
3. Greedy decoded moves match when the new per-target outputs are replaced with
   legacy slot logits broadcast across targets.

This harness is intentionally opening-state only. Residual=0 gives analytic parity
at any state; the separate 16-game Ajay panel is the trajectory-level sanity check.

Usage:
  PYTHONPATH=/Users/saheb/home/kaggle-orbit-wars/orbit_wars_rl:/Users/saheb/home/kaggle-orbit-wars \
    orbit_wars_rl/.venv/bin/python orbit_wars_rl/check_phase4_parity.py \
      --checkpoint gpu_run_artifacts/revedge1/checkpoints/torch_step_4718592_revedge1_20260616_095906.pt
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

import torch

from action_mask import actions_from_target_policy, compute_action_masks
from config import Config
from eval import load_checkpoint
from features import extract_features
from model import EntityTransformer
from torch_env import VecTorchEnv


EXPECTED_NEW_KEYS = {
    "fire_q.weight", "fire_q.bias",
    "fire_k.weight", "fire_k.bias",
    "fire_scorer.0.weight", "fire_scorer.0.bias",
    "fire_scorer.2.weight", "fire_scorer.2.bias",
    "ship_q.weight", "ship_q.bias",
    "ship_k.weight", "ship_k.bias",
    "ship_scorer.0.weight", "ship_scorer.0.bias",
    "ship_scorer.2.weight", "ship_scorer.2.bias",
}


def _env_obs(env: VecTorchEnv, env_idx: int, player: int) -> dict:
    planets = []
    for i in range(env.planets.shape[1]):
        if bool(env.planet_alive[env_idx, i]):
            planets.append([float(x) for x in env.planets[env_idx, i].tolist()])
    fleets = []
    for i in range(env.fleets.shape[1]):
        if bool(env.fleet_alive[env_idx, i]):
            fleets.append([float(x) for x in env.fleets[env_idx, i].tolist()])
    initial_planets = []
    for i in range(env.init_planets.shape[1]):
        p = env.init_planets[env_idx, i]
        if int(p[0].item()) >= 0:
            initial_planets.append([float(x) for x in p.tolist()])
    return {
        "step": int(env.step_count[env_idx].item()),
        "player": player,
        "angular_velocity": float(env.angular_velocity[env_idx].item()),
        "planets": planets,
        "fleets": fleets,
        "initial_planets": initial_planets,
        "comet_planet_ids": [],
    }


def _build_legacy_broadcast_outputs(model: EntityTransformer, features: dict, masks: dict) -> tuple[torch.Tensor, torch.Tensor]:
    slot_valid = masks["slot_valid"] if masks["slot_valid"].dim() == 2 else masks["slot_valid"].unsqueeze(0)
    fire_mask = masks["fire_mask"] if masks["fire_mask"].dim() == 2 else masks["fire_mask"].unsqueeze(0)
    owned_indices = masks["owned_indices"] if masks["owned_indices"].dim() == 2 else masks["owned_indices"].unsqueeze(0)
    with torch.no_grad():
        encoded = model.encode_state(
            features["planet_features"].unsqueeze(0),
            features["fleet_features"].unsqueeze(0),
            features["global_features"].unsqueeze(0),
            features["planet_mask"].unsqueeze(0),
            features["fleet_mask"].unsqueeze(0),
            slot_valid=slot_valid,
            owned_indices=owned_indices,
            pairwise_features=features["pairwise_features"].unsqueeze(0) if "pairwise_features" in features else None,
        )
        fire_slot = model.fire_head(encoded["owned_enriched"]).squeeze(-1)
        ship_slot = model.ship_head(encoded["owned_enriched"])
        max_planets = model.max_planets
        fire = fire_slot.unsqueeze(-1).expand(-1, -1, max_planets).clone()
        ship = ship_slot.unsqueeze(2).expand(-1, -1, max_planets, -1).clone()
        min_ship_bin = int(getattr(model.cfg, "min_ship_bin", 0))
        if min_ship_bin > 0:
            ship[..., :min_ship_bin] = -100.0
        fire = fire.masked_fill(~fire_mask.unsqueeze(-1), -100.0)
        fire = fire.masked_fill(~slot_valid.unsqueeze(-1), -100.0)
        ship = ship.masked_fill(~slot_valid.unsqueeze(-1).unsqueeze(-1), -100.0)
        return fire, ship


def _gather_chosen(per_target: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    target_idx = target_logits.argmax(dim=-1)
    return torch.gather(per_target, -1, target_idx.unsqueeze(-1)).squeeze(-1)


def _gather_chosen_ship(per_target: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    target_idx = target_logits.argmax(dim=-1)
    return torch.gather(
        per_target,
        2,
        target_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, per_target.shape[-1]),
    ).squeeze(2)


def _moves_for(obs: dict, masks: dict, fire_logits: torch.Tensor, target_logits: torch.Tensor,
               ship_logits: torch.Tensor, model: EntityTransformer) -> list[list[float]]:
    return actions_from_target_policy(
        fire_logits.cpu(),
        target_logits.cpu(),
        ship_logits.cpu(),
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
        obs,
        player=int(obs["player"]),
        fire_threshold=0.5,
        sample=False,
        ship_bin_mode=str(getattr(model.cfg, "ship_bin_mode", "absolute")),
        allow_reinforce=bool(getattr(model.cfg, "allow_reinforce", False)),
        reinforce_gate_min_planets=int(getattr(model.cfg, "reinforce_gate_min_planets", 0)),
        reinforce_forward_only=bool(getattr(model.cfg, "reinforce_forward_only", False)),
        reinforce_garrison_floor=float(getattr(model.cfg, "reinforce_garrison_floor", 0.0)),
        sufficient_commit_factor=0.0,
        reverse_edge_cooldown=int(getattr(model.cfg, "reverse_edge_cooldown", 0)),
        cooldown_last={} if int(getattr(model.cfg, "reverse_edge_cooldown", 0)) > 0 else None,
        cooldown_step=int(obs.get("step", 0)),
    )


def _iter_seed_player_pairs(seeds: Iterable[int], players: Iterable[int]) -> Iterable[tuple[int, int]]:
    for seed in seeds:
        for player in players:
            yield seed, player


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[7, 11, 42, 113, 911, 1337, 2476, 6462])
    args = ap.parse_args()

    cfg = Config()
    state_dict, action_decode = load_checkpoint(args.checkpoint, cfg)
    assert action_decode == "target", f"expected target decode, got {action_decode}"
    model = EntityTransformer(cfg.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    unexpected = set(unexpected)
    print(f"checkpoint: {args.checkpoint}")
    print(f"missing_keys: {sorted(missing)}")
    print(f"unexpected_keys: {sorted(unexpected)}")
    assert missing == EXPECTED_NEW_KEYS, f"unexpected missing keys: {sorted(missing ^ EXPECTED_NEW_KEYS)}"
    assert not unexpected, f"unexpected load keys: {sorted(unexpected)}"
    model.eval()

    env = VecTorchEnv(
        num_envs=1,
        num_players=2,
        device="cpu",
        action_decode="target",
        allow_reinforce=bool(getattr(cfg.model, "allow_reinforce", False)),
        reinforce_gate_min_planets=int(getattr(cfg.model, "reinforce_gate_min_planets", 0)),
        reinforce_forward_only=bool(getattr(cfg.model, "reinforce_forward_only", False)),
        reinforce_garrison_floor=float(getattr(cfg.model, "reinforce_garrison_floor", 0.0)),
        reverse_edge_cooldown=int(getattr(cfg.model, "reverse_edge_cooldown", 0)),
    )

    total = 0
    for seed, player in _iter_seed_player_pairs(args.seeds, [0, 1]):
        env.reset([seed])
        obs = _env_obs(env, 0, player)
        features = extract_features(
            obs,
            player=player,
            num_players=2,
            max_planets=cfg.model.max_planets,
            max_fleets=128,
        )
        masks = compute_action_masks(obs, player=player)
        fire_mask = masks["fire_mask"] if masks["fire_mask"].dim() == 2 else masks["fire_mask"].unsqueeze(0)
        slot_valid = masks["slot_valid"] if masks["slot_valid"].dim() == 2 else masks["slot_valid"].unsqueeze(0)
        owned_indices = masks["owned_indices"] if masks["owned_indices"].dim() == 2 else masks["owned_indices"].unsqueeze(0)
        with torch.no_grad():
            outputs = model(
                features["planet_features"].unsqueeze(0),
                features["fleet_features"].unsqueeze(0),
                features["global_features"].unsqueeze(0),
                features["planet_mask"].unsqueeze(0),
                features["fleet_mask"].unsqueeze(0),
                fire_mask=fire_mask,
                slot_valid=slot_valid,
                owned_indices=owned_indices,
                pairwise_features=features["pairwise_features"].unsqueeze(0) if "pairwise_features" in features else None,
            )
            legacy_fire, legacy_ship = _build_legacy_broadcast_outputs(model, features, masks)

        chosen_fire_new = _gather_chosen(outputs["fire_logits"], outputs["target_logits"])
        chosen_fire_old = _gather_chosen(legacy_fire, outputs["target_logits"])
        chosen_ship_new = _gather_chosen_ship(outputs["ship_logits"], outputs["target_logits"])
        chosen_ship_old = _gather_chosen_ship(legacy_ship, outputs["target_logits"])
        valid_slots = slot_valid

        assert torch.allclose(chosen_fire_new[valid_slots], chosen_fire_old[valid_slots]), (
            f"chosen fire mismatch at seed={seed} player={player}"
        )
        ship_valid = valid_slots.unsqueeze(-1).expand_as(chosen_ship_new)
        assert torch.allclose(chosen_ship_new[ship_valid], chosen_ship_old[ship_valid]), (
            f"chosen ship mismatch at seed={seed} player={player}"
        )

        moves_new = _moves_for(obs, masks, outputs["fire_logits"], outputs["target_logits"], outputs["ship_logits"], model)
        moves_old = _moves_for(obs, masks, legacy_fire, outputs["target_logits"], legacy_ship, model)
        assert moves_new == moves_old, f"decode mismatch at seed={seed} player={player}: {moves_new} != {moves_old}"
        total += 1

    print(f"parity_cases: {total}")
    print("PASS: phase4 Stage-A residual parity on seeded opening states (trajectory parity stays with panel eval)")


if __name__ == "__main__":
    main()
