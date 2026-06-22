"""Offline action-sensitivity gate for the COMA Q-head (docs/q-head.md, build-order step 1).

The make-or-break question BEFORE any GPU: trained only on the TAKEN action's return, does the
Q-head become action-sensitive, or does it collapse toward V (action-insensitive → A_i→0 → back to
square one)? We train a Q-ONLY head (trunk + policy + V frozen) on one real torch_env+GAE rollout
batch (dump via `train_torch.py --dump-rollout-and-exit`), regressing q_sa → returns, then measure:

  (1) GATE  — Q(s,a) − Q(s, all-idle):  materially ≠ 0 ⇒ the Q-head reacts to actions at all.
  (2) per-slot  q_fire − q_idle on IDLE valid slots: > 0 ⇒ the Q-head thinks firing the idle spare
      would help (the recoverable signal); A_i = q_sa − (p·q_fire+(1−p)·q_idle).
  (3) co-firing — same-vs-fresh target Q-gain (mean-pool aggregation blindness).

NOTE: (2) here is over ALL idle slots (most are correctly idle). The CONCLUSIVE per-opportunity
version is q_head_opportunity_gate.py, which restricts to true spare-fire opportunities vs Ajay.
This module exposes load/train as functions so the opportunity gate reuses the exact Q-head.

Run from repo root:
  <venv>/bin/python orbit_wars_rl/q_head_offline_probe.py \
      --rollout /tmp/qhead_rollout.pt \
      --checkpoint gpu_run_artifacts/r32_stage_hlr/checkpoints/torch_step_2097152_*.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import eval as E
from config import Config
from model import EntityTransformer


def _index(batch, idx):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv[idx] for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _encode(model, mb):
    return model.encode_state(
        mb["planet_features"], mb["fleet_features"], mb["global_features"],
        mb["planet_mask"], mb["fleet_mask"],
        slot_valid=mb["slot_valid"], owned_indices=mb["owned_indices"],
        pairwise_features=mb.get("pairwise_features"),
    )


def _qc(model, encoded, mb):
    return model.q_counterfactual(
        encoded, mb["actions"]["fire"], mb["actions"]["ship"],
        mb["actions"]["target"], mb["slot_valid"],
    )


def _fire_prob(model, mb):
    """p_i = sigmoid(fire_logit at the taken target), from the FROZEN policy heads."""
    out = model(
        mb["planet_features"], mb["fleet_features"], mb["global_features"],
        mb["planet_mask"], mb["fleet_mask"], fire_mask=mb["fire_mask"],
        slot_valid=mb["slot_valid"], owned_indices=mb["owned_indices"],
        pairwise_features=mb.get("pairwise_features"),
    )
    tgt = mb["actions"]["target"].unsqueeze(-1)
    fire_logit = torch.gather(out["fire_logits"], -1, tgt).squeeze(-1)
    return torch.sigmoid(fire_logit)


# ---- reusable load / train (imported by q_head_opportunity_gate) -----------------

def load_model_with_fresh_q(checkpoint):
    """Resume trunk/policy/V from a checkpoint; Q-head fresh + the only trainable params."""
    cfg = Config()
    sd, _ = E.load_checkpoint(checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("q_"))
    return model


def load_rollout(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def split_train_val(batch, val_frac, seed):
    TN = batch["returns"].shape[0]
    n_val = int(TN * val_frac)
    perm = torch.randperm(TN, generator=torch.Generator().manual_seed(seed))
    return perm[n_val:], perm[:n_val]


def _chunks(idx, size):
    for i in range(0, len(idx), size):
        yield idx[i:i + size]


def train_q_head(model, batch, train_idx, val_idx, epochs=40, mb=512, lr=1e-3, verbose=True):
    """Regress q_sa → returns (Q-only; trunk/policy/V frozen). Trains in place."""
    q_params = [p for n, p in model.named_parameters() if n.startswith("q_")]
    opt = torch.optim.Adam(q_params, lr=lr)
    for ep in range(epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        tot, nb = 0.0, 0
        for mb_idx in _chunks(order, mb):
            b = _index(batch, mb_idx)
            with torch.no_grad():
                enc = _encode(model, b)
            loss = F.mse_loss(_qc(model, enc, b)["q_sa"], b["returns"])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose and (ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1):
            model.eval()
            qs, rs = [], []
            with torch.no_grad():
                for mb_idx in _chunks(val_idx, mb):
                    b = _index(batch, mb_idx)
                    qs.append(_qc(model, _encode(model, b), b)["q_sa"]); rs.append(b["returns"])
            qs, rs = torch.cat(qs), torch.cat(rs)
            corr = torch.corrcoef(torch.stack([qs, rs]))[0, 1]
            print(f"  ep {ep+1:>3}  train_mse {tot/nb:.4f}  val_mse {F.mse_loss(qs,rs):.4f}  "
                  f"corr(Q,return) {corr:+.3f}", flush=True)
    model.eval()


def _cofire(model, encoded, mb, q):
    sv = mb["slot_valid"].bool()
    fire_a = mb["actions"]["fire"] > 0
    fired = fire_a & sv
    idle = (~fire_a) & sv
    has = fired.any(1) & idle.any(1)
    if has.sum() == 0:
        return torch.zeros(0), torch.zeros(0)
    rows = has.nonzero(as_tuple=False).flatten()
    i_slot = fired.float().argmax(1)[rows]
    j_slot = idle.float().argmax(1)[rows]
    T = mb["actions"]["target"][rows, i_slot]
    oe = encoded["owned_enriched"][rows]; pep = encoded["planet_emb"][rows]; gt = encoded["global_token"][rows]
    sv_r = sv[rows].float(); n_valid = sv_r.sum(1).clamp(min=1.0)
    fire_r = mb["actions"]["fire"][rows]; ship_r = mb["actions"]["ship"][rows]
    tgt_r = mb["actions"]["target"][rows].clone()
    sa_fire, sa_idle = model._q_slot_tokens(oe, pep, ship_r, tgt_r)
    sa_taken = torch.where((fire_r > 0).unsqueeze(-1), sa_fire, sa_idle)
    action_pool = (sa_taken * sv_r.unsqueeze(-1)).sum(1) / n_valid.unsqueeze(-1)
    q_sa = model._q_from_pool(gt, action_pool)
    coef = (sv_r / n_valid.unsqueeze(-1))
    ar = torch.arange(len(rows))
    pool_fresh = action_pool + coef[ar, j_slot].unsqueeze(-1) * (sa_fire[ar, j_slot] - sa_taken[ar, j_slot])
    d_fresh = model._q_from_pool(gt, pool_fresh) - q_sa
    tgt_same = tgt_r.clone(); tgt_same[ar, j_slot] = T
    sa_fire_same, _ = model._q_slot_tokens(oe, pep, ship_r, tgt_same)
    pool_same = action_pool + coef[ar, j_slot].unsqueeze(-1) * (sa_fire_same[ar, j_slot] - sa_taken[ar, j_slot])
    d_same = model._q_from_pool(gt, pool_same) - q_sa
    return d_same, d_fresh


def probe_report(model, batch, val_idx, ret):
    g_sens, fire_help_idle, A_idle, A_fired, cofire_same, cofire_fresh = [], [], [], [], [], []
    with torch.no_grad():
        for mb_idx in _chunks(val_idx, 512):
            mb = _index(batch, mb_idx)
            enc = _encode(model, mb)
            q = _qc(model, enc, mb)
            sv = mb["slot_valid"].bool()
            fired = (mb["actions"]["fire"] > 0) & sv
            idle = (~(mb["actions"]["fire"] > 0)) & sv
            g_sens.append(q["q_sa"] - q["q_all_idle"])
            fire_help_idle.append((q["q_fire"] - q["q_idle"])[idle])
            p = _fire_prob(model, mb).clamp(1e-4, 1 - 1e-4)
            A = q["q_sa"].unsqueeze(1) - (p * q["q_fire"] + (1 - p) * q["q_idle"])
            A_idle.append(A[idle]); A_fired.append(A[fired])
            cs, cf = _cofire(model, enc, mb, q)
            cofire_same.append(cs); cofire_fresh.append(cf)
    g_sens = torch.cat(g_sens); fhi = torch.cat(fire_help_idle)
    A_idle_t, A_fired_t = torch.cat(A_idle), torch.cat(A_fired)
    cs, cf = torch.cat(cofire_same), torch.cat(cofire_fresh)
    ret_std = ret.std().item()
    print("\n" + "=" * 84)
    print("Q-HEAD OFFLINE ACTION-SENSITIVITY GATE (broad population)")
    print("=" * 84)
    print(f"(1) GATE  Q(s,a) − Q(s,all-idle):  mean {g_sens.mean():+.4f}  "
          f"|mean| {g_sens.abs().mean():.4f}  std {g_sens.std():.4f}  "
          f"(= {100*g_sens.abs().mean().item()/max(ret_std,1e-9):.1f}% / "
          f"{100*g_sens.std().item()/max(ret_std,1e-9):.1f}% of returns-std)")
    print(f"(2) q_fire − q_idle on IDLE slots : mean {fhi.mean():+.4f}  std {fhi.std():.4f}")
    print(f"    A_i FIRED {A_fired_t.mean():+.4f} (n={len(A_fired_t)}) | "
          f"IDLE {A_idle_t.mean():+.4f} (n={len(A_idle_t)})")
    print(f"(3) co-firing same {cs.mean():+.4f} | fresh {cf.mean():+.4f} | "
          f"|diff| {(cs-cf).abs().mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-q", type=str, default=None, help="save the trained model state_dict here")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = load_model_with_fresh_q(args.checkpoint)
    batch = load_rollout(args.rollout)
    train_idx, val_idx = split_train_val(batch, args.val_frac, args.seed)
    ret = batch["returns"]
    print(f"TN={ret.shape[0]}  train={len(train_idx)}  val={len(val_idx)}  "
          f"returns: mean {ret.mean():+.3f}  std {ret.std():.3f}  "
          f"[min {ret.min():+.2f}, max {ret.max():+.2f}]", flush=True)
    train_q_head(model, batch, train_idx, val_idx, args.epochs, args.mb, args.lr)
    probe_report(model, batch, val_idx, ret)
    if args.save_q:
        torch.save(model.state_dict(), args.save_q)
        print(f"saved trained model -> {args.save_q}")


if __name__ == "__main__":
    main()
