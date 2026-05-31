#!/usr/bin/env python3
"""Quick checkpoint info — run before any eval or export to verify what you have.

Usage:
    python3 orbit_wars_rl/ckpt_info.py <checkpoint.pt> [<checkpoint2.pt> ...]
"""
import sys, torch

def ckpt_info(path):
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return
    cfg = ck.get('config', {})
    steps = ck.get('total_steps', ck.get('step', '?'))
    n_bins = cfg.get('num_ship_bins', '?')
    mode = cfg.get('ship_bin_mode', '?')
    decode = cfg.get('action_decode', '?')
    arch = f"{mode} {n_bins}-bin"
    print(f"  {path}")
    print(f"    arch:   {arch}  decode={decode}")
    print(f"    steps:  {steps:,}" if isinstance(steps, int) else f"    steps:  {steps}")
    print(f"    phase1: {'YES' if mode == 'absolute' else 'NO (old arch)'}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 orbit_wars_rl/ckpt_info.py <checkpoint.pt> ...")
        sys.exit(1)
    for path in sys.argv[1:]:
        ckpt_info(path)
