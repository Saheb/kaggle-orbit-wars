"""Self-play rollout collector and opponent pool for Orbit Wars."""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

from env import OrbitWarsEnv
from features import extract_features
from action_mask import compute_action_masks, actions_from_policy
from model import EntityTransformer, NUM_ANGLE_BINS, NUM_SHIP_BINS


class Transition:
    __slots__ = ['obs', 'features', 'masks', 'actions', 'log_probs', 'reward', 'done', 'value']

    def __init__(self, obs, features, masks, actions, log_probs, reward, done, value):
        self.obs = obs
        self.features = features
        self.masks = masks
        self.actions = actions
        self.log_probs = log_probs
        self.reward = reward
        self.done = done
        self.value = value


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


def _policy_forward(model, obs, player, num_players, device):
    """Run a single forward pass; return (outputs, features, masks)."""
    features = extract_features(obs, player, num_players=num_players)
    masks = compute_action_masks(obs, player)

    planet_feats = features["planet_features"].unsqueeze(0).to(device)
    fleet_feats = features["fleet_features"].unsqueeze(0).to(device)
    global_feats = features["global_features"].unsqueeze(0).to(device)
    planet_mask = features["planet_mask"].unsqueeze(0).to(device)
    fleet_mask = features["fleet_mask"].unsqueeze(0).to(device)
    fire_mask = masks["fire_mask"].to(device)
    angle_mask = masks["angle_mask"].to(device)
    slot_valid = masks["slot_valid"].to(device)
    owned_indices = masks["owned_indices"].to(device)

    with torch.no_grad():
        outputs = model(
            planet_feats, fleet_feats, global_feats,
            planet_mask, fleet_mask,
            fire_mask=fire_mask, angle_mask=angle_mask,
            slot_valid=slot_valid, owned_indices=owned_indices,
            owned_count=masks["owned_count"],
        )
    return outputs, features, masks


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

    transitions = []
    obs = env.reset()
    done = False
    step = 0
    prev_material = env.compute_material(0)

    while not done and step < num_steps:
        player = obs["player"]

        outputs, features, masks = _policy_forward(model, obs, player, env.num_players, device)
        value = outputs["value"].item()

        fire_mask = masks["fire_mask"].to(device)
        angle_mask = masks["angle_mask"].to(device)

        use_uniform = random.random() < epsilon
        fire_action, angle_action, ship_action, fire_dist, angle_dist, ship_dist = \
            _sample_action_masked(outputs, fire_mask, angle_mask, device, uniform=use_uniform)

        log_prob_fire = fire_dist.log_prob(fire_action.float())
        log_prob_angle = angle_dist.log_prob(angle_action)
        log_prob_ships = ship_dist.log_prob(ship_action)

        # Convert to env actions
        env_actions = actions_from_policy(
            outputs["fire_logits"], outputs["angle_logits"], outputs["ship_logits"],
            {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in masks.items()},
            obs, player,
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

        # Shaping reward: material delta (small signal, dominated by terminal ±1)
        if shaping_coef > 0.0 and not done:
            curr_material = env.compute_material(0)
            step_reward = shaping_coef * (curr_material - prev_material)
            prev_material = curr_material
        else:
            step_reward = 0.0

        transition = Transition(
            obs=obs,
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
        )
        transitions.append(transition)
        step += 1

    # Set terminal reward on the last step
    if transitions:
        final_reward = env._compute_reward()
        transitions[-1].reward += final_reward

    return transitions


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

    return {
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
