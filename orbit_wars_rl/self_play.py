"""Self-play rollout collector and opponent pool for Orbit Wars."""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

from env import OrbitWarsEnv
from features import extract_features
from action_mask import (
    compute_action_masks,
    actions_from_policy,
    actions_from_sampled_policy,
)
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS, ANGLE_BIN_WIDTH, SHIP_COUNTS


class Transition:
    __slots__ = [
        'obs', 'features', 'masks', 'actions', 'log_probs', 'reward', 'done',
        'value', 'bc_targets',
    ]

    def __init__(
        self,
        obs,
        features,
        masks,
        actions,
        log_probs,
        reward,
        done,
        value,
        bc_targets=None,
    ):
        self.obs = obs
        self.features = features
        self.masks = masks
        self.actions = actions
        self.log_probs = log_probs
        self.reward = reward
        self.done = done
        self.value = value
        self.bc_targets = bc_targets


class OpponentPool:
    def __init__(self, max_size=8):
        self.max_size = max_size
        self.checkpoints = deque(maxlen=max_size)

    def add(self, state_dict, step):
        self.checkpoints.append({"params": copy.deepcopy(state_dict), "step": step})

    def sample(self):
        if not self.checkpoints:
            return None
        return random.choice(list(self.checkpoints))["params"]

    def __len__(self):
        return len(self.checkpoints)


def _material_from_obs(obs, player):
    total = 0.0
    for p in obs["planets"]:
        if p[1] == player:
            total += p[5]
    for f in obs["fleets"]:
        if f[1] == player:
            total += f[6]
    return total


def _find_angle_bin(angle_rad: float) -> int:
    return int(float(angle_rad) / ANGLE_BIN_WIDTH) % NUM_ANGLE_BINS


def _find_ship_bin(ships: int, max_ships: int = 10000) -> int:
    best_bin, best_diff = 0, float("inf")
    for b in range(NUM_SHIP_BINS):
        count = SHIP_COUNTS[b]
        diff = abs(count - int(ships))
        if diff < best_diff:
            best_bin, best_diff = b, diff
    return best_bin


def teacher_targets_from_action(obs, action, masks, max_owned: int = 10):
    """Convert teacher moves into per-owned-slot BC targets."""
    player = obs.get("player", 0)
    planets = obs["planets"]
    owned_indices = masks["owned_indices"].cpu().numpy()
    max_ships = masks["max_ships"].cpu().numpy().squeeze(0)
    n_owned = masks["owned_count"]

    pid_to_slot = {}
    for slot in range(min(n_owned, max_owned)):
        pidx = int(owned_indices[slot])
        if pidx < len(planets):
            pid_to_slot[int(planets[pidx][0])] = slot

    fire = torch.zeros(max_owned, dtype=torch.long)
    angle = torch.zeros(max_owned, dtype=torch.long)
    ship = torch.zeros(max_owned, dtype=torch.long)

    for move in action or []:
        if len(move) < 3:
            continue
        slot = pid_to_slot.get(int(move[0]))
        if slot is None:
            continue
        fire[slot] = 1
        angle[slot] = _find_angle_bin(float(move[1]))
        ship[slot] = _find_ship_bin(int(move[2]), int(max_ships[slot]))

    return {"fire": fire, "angle": angle, "ship": ship}


def _policy_forward(model, obs, player, num_players, device):
    """Run a single forward pass; return (outputs, features, masks).

    Uses whatever device the model is currently on. The caller is responsible
    for placing the model on CPU before collection (faster for BS=1) and
    moving it back to the training device before PPO updates.
    """
    features = extract_features(obs, player, num_players=num_players)
    masks = compute_action_masks(obs, player)

    infer_dev = next(model.parameters()).device
    planet_feats = features["planet_features"].unsqueeze(0).to(infer_dev)
    fleet_feats = features["fleet_features"].unsqueeze(0).to(infer_dev)
    global_feats = features["global_features"].unsqueeze(0).to(infer_dev)
    planet_mask = features["planet_mask"].unsqueeze(0).to(infer_dev)
    fleet_mask = features["fleet_mask"].unsqueeze(0).to(infer_dev)
    fire_mask = masks["fire_mask"].to(infer_dev)
    angle_mask = masks["angle_mask"].to(infer_dev)
    slot_valid = masks["slot_valid"].to(infer_dev)
    owned_indices = masks["owned_indices"].to(infer_dev)

    with torch.no_grad():
        outputs = model(
            planet_feats, fleet_feats, global_feats,
            planet_mask, fleet_mask,
            fire_mask=fire_mask, angle_mask=angle_mask,
            slot_valid=slot_valid, owned_indices=owned_indices,
            owned_count=masks["owned_count"],
        )
    # Move outputs back to CPU so transitions are always stored on CPU
    return {k: v.cpu() for k, v in outputs.items()}, features, masks


def _sample_action_masked(outputs, fire_mask, angle_mask, device, uniform=False):
    """Sample actions from (optionally uniform) masked distributions.

    uniform=True gives uniform probability over legal actions (for epsilon exploration).
    Returns (fire_action, angle_action, ship_action, fire_dist, angle_dist, ship_dist).
    """
    if uniform:
        # Uniform over legal moves; masks are already embedded in the model output,
        # but here we build fresh uniform logits over valid actions.
        fire_logits_u = torch.zeros_like(outputs["fire_logits"])
        angle_logits_u = torch.zeros_like(outputs["angle_logits"])
        ship_logits_u = torch.zeros_like(outputs["ship_logits"])

        if fire_mask is not None:
            fire_logits_u = fire_logits_u.masked_fill(~fire_mask, -1e9)
        if angle_mask is not None:
            angle_logits_u = angle_logits_u.masked_fill(~angle_mask, -1e9)

        fire_dist = torch.distributions.Bernoulli(logits=fire_logits_u)
        angle_dist = torch.distributions.Categorical(logits=angle_logits_u)
        ship_dist = torch.distributions.Categorical(logits=ship_logits_u)
    else:
        fire_dist = torch.distributions.Bernoulli(logits=outputs["fire_logits"])
        angle_dist = torch.distributions.Categorical(logits=outputs["angle_logits"])
        ship_dist = torch.distributions.Categorical(logits=outputs["ship_logits"])

    fire_action = fire_dist.sample()
    angle_action = angle_dist.sample()
    ship_action = ship_dist.sample()

    return fire_action, angle_action, ship_action, fire_dist, angle_dist, ship_dist


def collect_rollout(
    model,
    env: OrbitWarsEnv,
    device,
    num_steps=500,
    epsilon=0.0,
    shaping_coef=0.0,
    opponent_model=None,
    teacher_agent=None,
):
    """Collect one episode of transitions using the current policy.

    Args:
        model: current policy (PyTorch nn.Module)
        env: OrbitWarsEnv instance
        device: torch device
        epsilon: probability of using uniform-over-legal-actions exploration
        shaping_coef: coefficient for material-delta shaping reward (0 = off)
        opponent_model: if not None, use this model for player 1's actions (self-play)

    Returns list of Transition objects.
    """
    model.eval()
    if opponent_model is not None:
        opponent_model.eval()

    # Inference device: use wherever the model currently is.
    # Callers should move model to CPU before collection (13ms CPU < 21ms MPS for BS=1)
    # and back to training device before PPO updates (MPS/CUDA needed for large batches).
    infer_dev = next(model.parameters()).device

    transitions = []
    obs = env.reset()
    done = False
    step = 0
    prev_material = env.compute_material(0)

    while not done and step < num_steps:
        prev_obs = obs
        player = obs["player"]

        outputs, features, masks = _policy_forward(model, obs, player, env.num_players, device)
        value = outputs["value"].item()

        # outputs are always on CPU (returned by _policy_forward); masks stay on CPU too
        fire_mask = masks["fire_mask"]
        angle_mask = masks["angle_mask"]

        use_uniform = random.random() < epsilon
        fire_action, angle_action, ship_action, fire_dist, angle_dist, ship_dist = \
            _sample_action_masked(outputs, fire_mask, angle_mask, infer_dev, uniform=use_uniform)

        log_prob_fire = fire_dist.log_prob(fire_action.float())
        log_prob_angle = angle_dist.log_prob(angle_action)
        log_prob_ships = ship_dist.log_prob(ship_action)

        # Convert the sampled action to env actions. PPO log-probs must describe
        # the same action that is applied to the environment.
        env_actions = actions_from_sampled_policy(
            fire_action, angle_action, ship_action,
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs, player,
        )
        bc_targets = None
        if teacher_agent is not None:
            bc_targets = teacher_targets_from_action(
                prev_obs,
                teacher_agent(prev_obs),
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            )

        # Compute opponent actions if self-playing
        opponent_actions = None
        if opponent_model is not None and env.num_players > 1:
            opp_obs = env.get_obs_for_player(1)
            opp_out, _, opp_masks = _policy_forward(opponent_model, opp_obs, 1, env.num_players, device)
            opp_actions = actions_from_policy(
                opp_out["fire_logits"], opp_out["angle_logits"], opp_out["ship_logits"],
                {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in opp_masks.items()},
                opp_obs, 1,
            )
            opponent_actions = [opp_actions]

        # Step environment
        obs, reward, done, info = env.step(env_actions, opponent_actions=opponent_actions)

        # Shaping reward: tanh-scaled material delta keeps signal in [-1,1] range
        if shaping_coef > 0.0 and not done:
            curr_material = env.compute_material(0)
            delta = curr_material - prev_material
            step_reward = shaping_coef * math.tanh(delta / 50.0)
            prev_material = curr_material
        else:
            step_reward = 0.0

        transition = Transition(
            obs=prev_obs,
            features={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in features.items()},
            masks={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            actions={
                "fire": fire_action.cpu(),
                "angle": angle_action.cpu(),
                "ship": ship_action.cpu(),
            },
            log_probs={
                "fire": log_prob_fire.cpu(),
                "angle": log_prob_angle.cpu(),
                "ships": log_prob_ships.cpu(),
            },
            reward=step_reward,
            done=done,
            value=value,
            bc_targets=bc_targets,
        )
        transitions.append(transition)
        step += 1

    # Set terminal reward on the last step
    if transitions:
        final_reward = env._compute_reward()
        transitions[-1].reward += final_reward

    return transitions


def collect_rollouts_vec(
    model,
    vec_pool,
    device,
    batch_size: int,
    gamma: float = 0.995,
    lam: float = 0.95,
    epsilon: float = 0.0,
    shaping_coef: float = 0.0,
    teacher_agent=None,
    opponent_model=None,
):
    """Collect batch_size transitions from N environments in parallel.

    Workers handle env.step + feature extraction simultaneously. Main process
    batches all N observations into a single model forward (BS=N on MPS).

    Returns (all_transitions_with_gae, episode_rewards) where
    all_transitions_with_gae is a list of (Transition, advantage, return).
    """
    from vec_env import _to_tensors

    N = vec_pool.num_envs
    model.eval()
    # For batched inference, MPS wins at BS≥4; keep model on training device.
    infer_dev = device

    # Per-env episode accumulators
    ep_transitions = [[] for _ in range(N)]
    completed: list[tuple] = []   # (Transition, advantage, return)
    ep_rewards: list[float] = []
    prev_materials = [0.0] * N

    # Initial reset
    obs_data = vec_pool.reset()
    prev_materials = [_material_from_obs(x["obs"], x["obs"].get("player", 0)) for x in obs_data]

    while len(completed) < batch_size:
        # ---- Batch model inference ----------------------------------------
        feats, masks = _to_tensors(obs_data)

        with torch.no_grad():
            outputs = model(
                feats["planet_features"].to(infer_dev),
                feats["fleet_features"].to(infer_dev),
                feats["global_features"].to(infer_dev),
                feats["planet_mask"].to(infer_dev),
                feats["fleet_mask"].to(infer_dev),
                fire_mask=masks["fire_mask"].to(infer_dev),
                angle_mask=masks["angle_mask"].to(infer_dev),
                slot_valid=masks["slot_valid"].to(infer_dev),
                owned_indices=masks["owned_indices"].to(infer_dev),
            )
        # Move outputs to CPU for action sampling
        outputs_cpu = {k: v.cpu() for k, v in outputs.items()}

        # ---- Per-env action sampling + env step ---------------------------
        all_actions = []
        env_data = []

        for i in range(N):
            eo = {k: v[i:i+1] for k, v in outputs_cpu.items()}
            fire_m  = masks["fire_mask"][i:i+1]
            angle_m = masks["angle_mask"][i:i+1]

            use_uniform = random.random() < epsilon
            fa, aa, sa, fd, ad, sd = _sample_action_masked(
                eo, fire_m, angle_m, torch.device("cpu"), uniform=use_uniform
            )

            env_actions = actions_from_sampled_policy(
                fa, aa, sa,
                {
                    "fire_mask":   fire_m,
                    "angle_mask":  angle_m,
                    "slot_valid":  masks["slot_valid"][i:i+1],
                    "owned_indices": masks["owned_indices"][i],
                    "max_ships":   masks["max_ships"][i:i+1],
                    "owned_count": masks["owned_counts"][i],
                },
                obs_data[i]["obs"], obs_data[i]["obs"].get("player", 0),
            )
            bc_targets = None
            if teacher_agent is not None:
                teacher_obs = obs_data[i]["obs"]
                bc_targets = teacher_targets_from_action(
                    teacher_obs,
                    teacher_agent(teacher_obs),
                    {
                        "fire_mask": fire_m,
                        "angle_mask": angle_m,
                        "slot_valid": masks["slot_valid"][i:i+1],
                        "owned_indices": masks["owned_indices"][i],
                        "max_ships": masks["max_ships"][i:i+1],
                        "owned_count": masks["owned_counts"][i],
                    },
                )
            all_actions.append(env_actions)
            env_data.append({
                "obs": obs_data[i]["obs"],
                "features": {
                    "planet_features": feats["planet_features"][i],
                    "fleet_features":  feats["fleet_features"][i],
                    "global_features": feats["global_features"][i],
                    "planet_mask":     feats["planet_mask"][i],
                    "fleet_mask":      feats["fleet_mask"][i],
                    "owned_indices":   masks["owned_indices"][i],
                    "owned_count":     masks["owned_counts"][i],
                },
                "masks": {
                    "fire_mask":    fire_m,
                    "angle_mask":   angle_m,
                    "slot_valid":   masks["slot_valid"][i:i+1],
                    "owned_indices":masks["owned_indices"][i],
                    "max_ships":    masks["max_ships"][i:i+1],
                    "owned_count":  masks["owned_counts"][i],
                },
                "actions":   {"fire": fa, "angle": aa, "ship": sa},
                "log_probs": {
                    "fire":  fd.log_prob(fa.float()),
                    "angle": ad.log_prob(aa),
                    "ships": sd.log_prob(sa),
                },
                "value": eo["value"].item(),
                "bc_targets": bc_targets,
            })

        # Step all envs (parallel in workers)
        step_results = vec_pool.step(all_actions)

        # ---- Process results per env -------------------------------------
        obs_data = []
        for i, (sr, ed) in enumerate(zip(step_results, env_data)):
            done = sr["done"]
            player = sr["obs"].get("player", 0)
            if shaping_coef > 0.0 and not done:
                curr_material = _material_from_obs(sr["obs"], player)
                delta = curr_material - prev_materials[i]
                step_reward = shaping_coef * math.tanh(delta / 50.0)
                prev_materials[i] = curr_material
            else:
                step_reward = 0.0

            t = Transition(
                obs=ed["obs"],
                features=ed["features"],
                masks=ed["masks"],
                actions=ed["actions"],
                log_probs=ed["log_probs"],
                reward=step_reward,
                done=done,
                value=ed["value"],
                bc_targets=ed["bc_targets"],
            )
            ep_transitions[i].append(t)

            if done:
                # Set terminal reward (worker auto-reset, so terminal_reward is separate)
                term_r = sr.get("terminal_reward") or 0.0
                ep_transitions[i][-1].reward += term_r

                # GAE for completed episode
                ep = ep_transitions[i]
                adv, ret = compute_gae(ep, gamma=gamma, lam=lam)
                for tt, a, r in zip(ep, adv, ret):
                    completed.append((tt, float(a), float(r)))
                ep_rewards.append(term_r)
                ep_transitions[i] = []
                prev_materials[i] = _material_from_obs(sr["obs"], player)

            # Worker already auto-reset on done; obs_data carries new obs
            obs_data.append(sr)

    return completed[:batch_size], ep_rewards


def compute_gae(transitions, gamma=0.995, lam=0.95):
    """Compute GAE advantages and returns from transitions."""
    values = [t.value for t in transitions]
    rewards = [t.reward for t in transitions]
    dones = [float(t.done) for t in transitions]

    advantages = np.zeros(len(transitions), dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(len(transitions))):
        if t == len(transitions) - 1:
            next_value = 0.0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = delta + gamma * lam * next_non_terminal * last_gae
        last_gae = advantages[t]

    returns = advantages + np.array(values)
    return advantages, returns


def make_batch(transitions, advantages, returns, device="cpu"):
    """Convert transitions into a training batch dict."""
    planet_features = torch.stack([t.features["planet_features"] for t in transitions])
    fleet_features = torch.stack([t.features["fleet_features"] for t in transitions])
    global_features = torch.stack([t.features["global_features"] for t in transitions])
    planet_mask = torch.stack([t.features["planet_mask"] for t in transitions])
    fleet_mask = torch.stack([t.features["fleet_mask"] for t in transitions])

    fire_mask = torch.stack([t.masks["fire_mask"][0] for t in transitions])
    angle_mask = torch.stack([t.masks["angle_mask"][0] for t in transitions])
    slot_valid = torch.stack([t.masks["slot_valid"][0] for t in transitions])
    owned_indices = torch.stack([t.masks["owned_indices"] for t in transitions])

    fire_action = torch.stack([t.actions["fire"][0] for t in transitions])
    angle_action = torch.stack([t.actions["angle"][0] for t in transitions])
    ship_action = torch.stack([t.actions["ship"][0] for t in transitions])

    log_prob_fire = torch.stack([t.log_probs["fire"][0] for t in transitions])
    log_prob_angle = torch.stack([t.log_probs["angle"][0] for t in transitions])
    log_prob_ships = torch.stack([t.log_probs["ships"][0] for t in transitions])

    old_values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    ret_tensor = torch.tensor(returns, dtype=torch.float32)

    out = {
        "planet_features": planet_features,
        "fleet_features": fleet_features,
        "global_features": global_features,
        "planet_mask": planet_mask,
        "fleet_mask": fleet_mask,
        "fire_mask": fire_mask,
        "angle_mask": angle_mask,
        "slot_valid": slot_valid,
        "owned_indices": owned_indices,
        "actions": {
            "fire": fire_action,
            "angle": angle_action,
            "ship": ship_action,
        },
        "old_log_probs": {
            "fire": log_prob_fire,
            "angle": log_prob_angle,
            "ships": log_prob_ships,
        },
        "advantages": adv_tensor,
        "returns": ret_tensor,
        "old_values": old_values,
    }

    if transitions and transitions[0].bc_targets is not None:
        out["bc_targets"] = {
            "fire": torch.stack([t.bc_targets["fire"] for t in transitions]),
            "angle": torch.stack([t.bc_targets["angle"] for t in transitions]),
            "ship": torch.stack([t.bc_targets["ship"] for t in transitions]),
        }

    return out


def make_minibatches(batch, num_minibatches, device="cpu"):
    """Split batch into minibatches."""
    batch_size = batch["planet_features"].shape[0]
    mini_size = batch_size // num_minibatches
    indices = torch.randperm(batch_size)
    minibatches = []
    for i in range(num_minibatches):
        idx = indices[i * mini_size:(i + 1) * mini_size]
        mini = {k: (v[idx] if isinstance(v, torch.Tensor) else
                     {k2: v2[idx] for k2, v2 in v.items()} if isinstance(v, dict) else v)
                for k, v in batch.items()}
        minibatches.append(mini)
    return minibatches
