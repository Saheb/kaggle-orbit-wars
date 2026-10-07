"""Instrument the binary resolver on REAL states: what does it actually send to own targets?"""
import os, sys
sys.path.insert(0, "/Users/saheb/home/kaggle-orbit-wars/orbit_wars_rl")
os.chdir("/Users/saheb/home/kaggle-orbit-wars")
import numpy as np, torch
import eval as ev
from config import Config
from model import EntityTransformer
from features import extract_features
from action_mask import compute_action_masks, resolve_binary_commit_np, MIN_BINARY_COMMIT_SHIPS
from kaggle_environments import make

CK = ("gpu_run_artifacts/binarymarg100m_l4_from25m/checkpoints/"
      "torch_step_40108032_binarymarg100m_l4_from25m_20260714_163936.pt")
cfg = Config(); cfg.device = "cpu"
model, ad = ev.load_eval_model(CK, cfg)   # evaluate_checkpoint's own construction
agent = ev.build_agent_fn(model, torch.device("cpu"), fire_threshold=0.5,
                          ship_bin_mode=cfg.model.ship_bin_mode,
                          target_decode=(ad == "target"), num_players=2)

own_defend_over_S, own_ok, own_n = [], 0, 0
mass_soon_vals = []
states = []

def spy(obs):
    if len(states) < 400:
        states.append((obs["planets"], obs["fleets"], obs["player"], obs.get("step", 0),
                       obs.get("angular_velocity", 0.0), obs.get("initial_planets"),
                       obs.get("comet_planet_ids", [])))
    return agent(obs)

env = make("orbit_wars", configuration={"seed": 3}, debug=False)
env.run([spy, "opponents/candidate_ender.py"])

for (planets, fleets, player, step, av, ip, cids) in states[::3]:
    obs = {"planets": planets, "fleets": fleets, "player": player, "step": step,
           "angular_velocity": av, "initial_planets": ip or planets, "comet_planet_ids": cids}
    f = extract_features(obs, player, num_players=2, global_econ=False)
    pw = f["pairwise_features"].numpy()
    owned = [p for p in planets if int(p[1]) == player]
    src_ships = np.zeros(pw.shape[0], dtype=np.float32)
    m = compute_action_masks(obs, player)
    oi = m["owned_indices"] if "owned_indices" in m else None
    for slot in range(pw.shape[0]):
        idx = int(oi[slot]) if oi is not None else -1
        if 0 <= idx < len(planets):
            src_ships[slot] = float(planets[idx][5])
    sizes, feas = resolve_binary_commit_np(pw, src_ships)
    is_own = pw[..., 5] > 0.5
    S = src_ships[:, None]
    defend = np.rint(pw[..., 24] * 200.0)
    valid = is_own & (S > 0)
    own_n += int(valid.sum())
    own_ok += int((feas & valid).sum())
    r = np.where(S > 0, defend / np.maximum(S, 1), 0)
    own_defend_over_S.extend(r[valid].tolist())
    mass_soon_vals.extend((pw[..., 20] * 100.0)[valid].tolist())

a = np.array(own_defend_over_S)
ms = np.array(mass_soon_vals)
print(f"own-target (slot,target) cells with S>0: {own_n}")
print(f"  feasible (defend_ok): {own_ok} ({100*own_ok/max(own_n,1):.1f}%)")
print(f"  defend/S  : median {np.median(a):.3f}  mean {a.mean():.3f}")
for lo, hi in ((0,.05),(.05,.5),(.5,.95),(.95,1.01),(1.01,99)):
    n = ((a>=lo)&(a<hi)).sum()
    print(f"    {lo:.2f}-{hi:.2f}: {100*n/len(a):5.1f}%")
print(f"  ch20 enemy_mass_soon (denorm): median {np.median(ms):.1f} mean {ms.mean():.1f} "
      f"max {ms.max():.1f}  ==0: {100*(ms==0).mean():.1f}%")
print(f"  defend>=5 : {100*(np.rint(a*1)>=0).mean():.1f}% (placeholder)")
