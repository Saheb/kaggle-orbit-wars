"""Faithful checkpoint loader — honors the discipline the checkpoint PERSISTS (gate,
forward_only, garrison_floor, reverse_edge_cooldown, AND sufficient_commit_factor),
exactly like eval.py's evaluate_checkpoint. Unlike expansion_probe.load_ckpt_agent,
it does NOT force sufficient_commit_factor=0.0 — so the seat plays as trained/officially
evaluated (phase4e persists suff=1.0, veto ON)."""
import torch
from config import Config
from eval import load_checkpoint, build_agent_fn
from model import EntityTransformer


def load_faithful_agent(ckpt_path, device="cpu", fire_threshold=0.5):
    cfg = Config()
    cfg.device = device
    sd, ckpt_action_decode = load_checkpoint(ckpt_path, cfg)   # sets discipline on cfg.model
    model = EntityTransformer(cfg.model).to(device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = int(cfg.model.reinforce_gate_min_planets)
    model.reinforce_forward_only = bool(cfg.model.reinforce_forward_only)
    model.reinforce_garrison_floor = float(cfg.model.reinforce_garrison_floor)
    model.reverse_edge_cooldown = int(getattr(cfg.model, "reverse_edge_cooldown", 0))
    model.sufficient_commit_factor = float(cfg.model.sufficient_commit_factor)   # FAITHFUL (not 0)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[faithful] gate={model.reinforce_gate_min_planets} forward_only={model.reinforce_forward_only} "
          f"floor={model.reinforce_garrison_floor} suff={model.sufficient_commit_factor} "
          f"decode={ckpt_action_decode}")
    return build_agent_fn(model, torch.device(device), fire_threshold=fire_threshold, sample=False,
                          ship_bin_mode=cfg.model.ship_bin_mode,
                          target_decode=(ckpt_action_decode == "target"))
