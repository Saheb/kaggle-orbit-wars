"""Board-distribution A/B: is the stratified panel harder than natural random boards?

Runs the SAME checkpoint vs the SAME opponent with the SAME both-seats loop, differing ONLY
in the board seeds: the 128 stratified panel seeds vs 128 random seeds. If WR is ~equal, the
panel's board distribution is neutral (self-play boards == panel == LB). If random-WR >> panel-WR,
something board/seat-shaped remains beyond opponent strength.

Run: orbit_wars_rl/.venv/bin/python orbit_wars_rl/board_dist_ab.py <ckpt> <opponent.py>
"""
import sys, random
import torch
from config import Config
from model import EntityTransformer
from eval import load_checkpoint, build_agent_fn
from eval_panel import BY_ARCHETYPE
from kaggle_environments import make

CKPT = sys.argv[1] if len(sys.argv) > 1 else "gpu_run_artifacts/p2rev5/checkpoints/torch_step_4194304_p2rev5_20260612_083540.pt"
OPP = sys.argv[2] if len(sys.argv) > 2 else "opponents/candidate_ajay_1200.py"


def load_agent():
    cfg = Config()
    cfg.env.num_players = 2
    state_dict, ckpt_decode = load_checkpoint(CKPT, cfg)
    model = EntityTransformer(cfg.model).to(cfg.device)
    model.allow_reinforce = bool(getattr(cfg.model, "allow_reinforce", False))
    model.reinforce_gate_min_planets = 3      # p2rev5 masks: gate3 / floor0 / no-forward
    model.reinforce_forward_only = False
    model.reinforce_garrison_floor = 0.0
    model.sufficient_commit_factor = 0.0
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return build_agent_fn(model, torch.device(cfg.device), fire_threshold=0.5,
                          ship_bin_mode=cfg.model.ship_bin_mode, target_decode=True)


def run_seeds(agent_fn, seeds, label):
    w = t = w0 = t0 = w1 = t1 = 0
    for seed in seeds:
        for my_seat in (0, 1):
            agents = [agent_fn, OPP] if my_seat == 0 else [OPP, agent_fn]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(agents)
            final = env.steps[-1]
            rew = [s.reward if s.reward is not None else 0.0 for s in final]
            win = int(rew[my_seat] > rew[1 - my_seat])
            w += win; t += 1
            if my_seat == 0: w0 += win; t0 += 1
            else: w1 += win; t1 += 1
            if t % 16 == 0:
                print(f"  [{label}] {t}/{len(seeds)*2}  WR {100*w/t:.1f}% (s0 {100*w0/max(t0,1):.0f}% s1 {100*w1/max(t1,1):.0f}%)", flush=True)
    print(f">>> {label}: WR {100*w/t:.1f}%  ({w}/{t})  seat0 {100*w0/max(t0,1):.1f}%  seat1 {100*w1/max(t1,1):.1f}%", flush=True)
    return w, t


if __name__ == "__main__":
    print(f"ckpt={CKPT.split('/')[-1]}  opp={OPP.split('/')[-1]}")
    agent_fn = load_agent()
    panel_seeds = [s for seeds in BY_ARCHETYPE.values() for s in seeds]   # 128 stratified
    rng = random.Random(20260613)
    random_seeds = [rng.randint(0, 2**31) for _ in range(128)]           # 128 natural random
    print(f"panel seeds: {len(panel_seeds)}   random seeds: {len(random_seeds)}   (both seats → 256 games each)")
    print("=== PANEL (stratified) ===")
    run_seeds(agent_fn, panel_seeds, "PANEL")
    print("=== RANDOM (natural = self-play = LB) ===")
    run_seeds(agent_fn, random_seeds, "RANDOM")
