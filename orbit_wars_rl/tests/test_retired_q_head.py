"""Checkpoints saved before the COMA Q-head was removed (2026-10) must still load.

They carry 10 q_* weights (registered last) and list those params last in the single Adam
group. Weights: EntityTransformer.load_state_dict drops them. Optimizer: the resume path trims
them, or the warm-Adam load size-mismatches and silently falls back to a cold optimizer.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from model import EntityTransformer  # noqa: E402
from ppo import PPOLearner  # noqa: E402
from train_torch import _drop_retired_q_head_from_optimizer  # noqa: E402

_Q_KEYS = ["q_fire_embed.weight", "q_ship_embed.weight", "q_tgt_proj.weight", "q_tgt_proj.bias",
           "q_sa_mlp.weight", "q_sa_mlp.bias", "q_fc.weight", "q_fc.bias",
           "q_out.weight", "q_out.bias"]


def _old_style_checkpoint():
    """A current learner's checkpoint, made to look pre-removal: q_* weights appended to the
    model state and 10 trailing state-less param ids appended to the Adam group."""
    cfg = Config()
    learner = PPOLearner(EntityTransformer(cfg.model), cfg)
    loss = sum(p.sum() for p in learner.model.parameters())
    loss.backward()
    learner.optimizer.step()                      # every real param now has Adam state
    model_sd = dict(learner.model.state_dict())
    for k in _Q_KEYS:
        model_sd[k] = torch.zeros(1)
    optim_sd = learner.optimizer.state_dict()
    ids = optim_sd["param_groups"][0]["params"]
    optim_sd["param_groups"][0]["params"] = ids + list(range(len(ids), len(ids) + len(_Q_KEYS)))
    return cfg, model_sd, optim_sd


def test_old_q_head_weights_load_strictly():
    cfg, model_sd, _ = _old_style_checkpoint()
    EntityTransformer(cfg.model).load_state_dict(model_sd)          # strict=True: no raise


def test_warm_adam_survives_the_q_head_trim():
    cfg, model_sd, optim_sd = _old_style_checkpoint()
    first_moment = optim_sd["state"][0]["exp_avg"].clone()
    fresh = PPOLearner(EntityTransformer(cfg.model), cfg)
    fresh.optimizer.load_state_dict(_drop_retired_q_head_from_optimizer(optim_sd, model_sd))
    first_param = fresh.optimizer.param_groups[0]["params"][0]
    assert torch.equal(fresh.optimizer.state[first_param]["exp_avg"], first_moment)
