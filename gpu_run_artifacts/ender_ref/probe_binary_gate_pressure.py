"""How much does each hardcoded gate in the binary action path actually bind?

Gates under test (all hand-tuned constants):
  MIN_BINARY_COMMIT_SHIPS = 5      -> S >= 5 for any commit
  capture_required                  -> single source must afford the capture ALONE (pincer wall)
                                       = cap_cost_at_arrival + is_enemy*prod*5.0*3.0 + 1.0
  maintain = enemy_mass_soon + 1    -> own-target size, gated at >= 5 (reinforce wall)
  _THREAT_ETA_WINDOW = 6.0          -> what counts as "soon"
"""
import os, sys
sys.path.insert(0, "/Users/saheb/home/kaggle-orbit-wars/orbit_wars_rl")
os.chdir("/Users/saheb/home/kaggle-orbit-wars")
import numpy as np, torch
import eval as ev
from config import Config
from model import EntityTransformer
from features import extract_features
from action_mask import compute_action_masks, MIN_BINARY_COMMIT_SHIPS, resolve_binary_commit_np
from kaggle_environments import make

CK = ("gpu_run_artifacts/binarymarg100m_l4_from25m/checkpoints/"
      "torch_step_40108032_binarymarg100m_l4_from25m_20260714_163936.pt")
cfg = Config(); cfg.device = "cpu"
model, ad = ev.load_eval_model(CK, cfg)   # evaluate_checkpoint's own construction
agent = ev.build_agent_fn(model, torch.device("cpu"), fire_threshold=0.5,
                          ship_bin_mode=cfg.model.ship_bin_mode,
                          target_decode=(ad == "target"), num_players=2)
states = []
def spy(obs):
    if len(states) < 500:
        states.append(dict(planets=obs["planets"], fleets=obs["fleets"], player=obs["player"],
                           step=obs.get("step", 0), angular_velocity=obs.get("angular_velocity", 0.0),
                           initial_planets=obs.get("initial_planets") or obs["planets"],
                           comet_planet_ids=obs.get("comet_planet_ids", [])))
    return agent(obs)

for seed in (3, 11, 17):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([spy, "opponents/candidate_ender.py"])

tot_own = tot_enemy_neu = 0
own_feas = atk_feas = 0
atk_fail_minship = atk_fail_afford = 0
own_fail_minship = own_fail_maintain = 0
for st in states[::3]:
    obs = dict(st)
    f = extract_features(obs, obs["player"], num_players=2, global_econ=False)
    pw = f["pairwise_features"].numpy()
    m = compute_action_masks(obs, obs["player"])
    oi = m["owned_indices"]
    S = np.zeros((pw.shape[0], 1), dtype=np.float32)
    for slot in range(pw.shape[0]):
        idx = int(oi[slot])
        if 0 <= idx < len(obs["planets"]):
            S[slot, 0] = float(obs["planets"][idx][5])
    is_own = pw[..., 5] > 0.5
    is_enemy = pw[..., 6] > 0.5
    is_tgt = pw[..., 9] > 0.5      # valid (slot,target) cell
    capture_required = pw[..., 10] * 200.0 + is_enemy.astype(np.float32) * pw[..., 8] * 5.0 * 3.0 + 1.0
    defend = np.rint(pw[..., 24] * 200.0)
    Sb = np.broadcast_to(S, pw.shape[:2])

    own_cells = is_own & is_tgt & (Sb > 0)
    atk_cells = (~is_own) & is_tgt & (Sb > 0)
    tot_own += int(own_cells.sum()); tot_enemy_neu += int(atk_cells.sum())

    minship_ok = Sb >= MIN_BINARY_COMMIT_SHIPS
    afford_ok = (Sb + 1e-3) >= capture_required
    own_feas += int((own_cells & (defend >= MIN_BINARY_COMMIT_SHIPS) & ((Sb+1e-3) >= defend)).sum())
    atk_feas += int((atk_cells & minship_ok & afford_ok).sum())
    atk_fail_minship += int((atk_cells & ~minship_ok).sum())
    atk_fail_afford += int((atk_cells & minship_ok & ~afford_ok).sum())
    own_fail_minship += int((own_cells & ~minship_ok).sum())
    own_fail_maintain += int((own_cells & minship_ok & (defend < MIN_BINARY_COMMIT_SHIPS)).sum())

def pc(n, d): return f"{100.0*n/max(d,1):5.1f}%"
print(f"\n--- ATTACK cells (target not owned): {tot_enemy_neu}")
print(f"  legal                          {atk_feas:6d}  {pc(atk_feas, tot_enemy_neu)}")
print(f"  blocked by S < 5 (MIN_COMMIT)  {atk_fail_minship:6d}  {pc(atk_fail_minship, tot_enemy_neu)}")
print(f"  blocked by capture_required    {atk_fail_afford:6d}  {pc(atk_fail_afford, tot_enemy_neu)}  <- the PINCER wall")
print(f"\n--- REINFORCE cells (own target): {tot_own}")
print(f"  legal                          {own_feas:6d}  {pc(own_feas, tot_own)}")
print(f"  blocked by S < 5               {own_fail_minship:6d}  {pc(own_fail_minship, tot_own)}")
print(f"  blocked by maintain < 5        {own_fail_maintain:6d}  {pc(own_fail_maintain, tot_own)}  <- the REINFORCE wall")
tot = tot_own + tot_enemy_neu
print(f"\n--- ALL commit options: {tot}   legal {atk_feas+own_feas} ({pc(atk_feas+own_feas, tot)})")
print(f"    => {pc(tot-atk_feas-own_feas, tot)} of the action space is removed by hardcoded gates")

# --- what does gates="minimal" open up on the SAME states? ---
mn_legal = mn_tot = 0
for st in states[::3]:
    obs = dict(st)
    f = extract_features(obs, obs["player"], num_players=2, global_econ=False)
    pw = f["pairwise_features"].numpy()
    m = compute_action_masks(obs, obs["player"])
    oi = m["owned_indices"]
    S = np.zeros(pw.shape[0], dtype=np.float32)
    for slot in range(pw.shape[0]):
        idx = int(oi[slot])
        if 0 <= idx < len(obs["planets"]):
            S[slot] = float(obs["planets"][idx][5])
    _, feas = resolve_binary_commit_np(pw, S, gates="minimal")
    is_tgt = pw[..., 9] > 0.5
    Sb = np.broadcast_to(S[:, None], pw.shape[:2])
    cells = is_tgt & (Sb > 0)
    mn_tot += int(cells.sum()); mn_legal += int((cells & feas).sum())
print(f"\n=== gates='minimal' on the same states ===")
print(f"  legal {mn_legal}/{mn_tot} = {100.0*mn_legal/max(mn_tot,1):.1f}%   (was 19.8% under 'full')")
