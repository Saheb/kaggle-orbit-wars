"""Re-run the model on replay states: for each FIRING source slot, inspect the
ship-head bin distribution and counterfactually mask bins 0-3.

  python orbit_wars_rl/ship0_counterfactual.py <ckpt> <replay_dir...> --our-name Saheb

Answers (pairs with ship0_why.py):
  (1) is a low-bin (1-4 ship) choice a confident spike, or a marginal argmax sitting
      on diffuse real-size mass? -> mean P(bins0-3) and mean P(argmax spike).
  (2) when bins 0-3 are masked (`--min-ship-bin 4`), where does the argmax go --
      5 ships (bin4), or the policy's real-size secondary mode? -> counterfactual count.

The fire head is separate from the ship head (model.py), so masking cannot force a
firing source to idle at decode -- it strictly upgrades probe->mass. Forced-small
garrisons (g<=4) are excluded (ship0_why shows they are ~1% of launches). Run from
repo root. Steps are subsampled (every other) for speed.
"""
import argparse
import glob
import json
import statistics
import sys

import torch

sys.path.insert(0, "orbit_wars_rl")
from config import Config
from eval import load_checkpoint
from model import EntityTransformer, SHIP_COUNTS
from features import extract_features
from action_mask import compute_action_masks

SC = torch.tensor(SHIP_COUNTS)


@torch.no_grad()
def analyze(model, paths, our_name, want_res, label):
    fire_n = ship0_n = cf_bin4 = 0
    p_low, p_spike, cf_counts, garr = [], [], [], []
    for path in paths:
        d = json.load(open(path))
        names = d["info"]["TeamNames"]
        if our_name not in names:
            continue
        seat = names.index(our_name)
        if ("WON" if d["rewards"][seat] > d["rewards"][1 - seat] else "LOST") != want_res:
            continue
        steps = d["steps"]
        for t in range(0, len(steps), 2):           # subsample for speed
            obs = steps[t][seat].get("observation") or {}
            if not obs.get("planets"):
                continue
            obs = dict(obs)
            obs["player"] = seat
            feats = extract_features(obs, seat, num_players=2)
            m = compute_action_masks(obs, seat)
            out = model(
                feats["planet_features"].unsqueeze(0), feats["fleet_features"].unsqueeze(0),
                feats["global_features"].unsqueeze(0), feats["planet_mask"].unsqueeze(0),
                feats["fleet_mask"].unsqueeze(0),
                fire_mask=m["fire_mask"], angle_mask=m["angle_mask"], slot_valid=m["slot_valid"],
                owned_indices=m["owned_indices"], owned_count=m["owned_count"],
                pairwise_features=feats["pairwise_features"].unsqueeze(0)
                    if "pairwise_features" in feats else None,
            )
            fl, tl, sl = out["fire_logits"][0], out["target_logits"][0], out["ship_logits"][0]
            sv, mx = m["slot_valid"][0], m["max_ships"][0]
            for s in range(fl.shape[0]):
                if not bool(sv[s]):
                    continue
                g = int(mx[s])
                if g <= 4:                          # exclude forced-small
                    continue
                tgt = int(tl[s].argmax())           # the slot's preferred target
                if torch.sigmoid(fl[s, tgt]) <= 0.5:  # not firing
                    continue
                fire_n += 1
                garr.append(g)
                dist = torch.softmax(sl[s, tgt], dim=-1)
                amax = int(dist.argmax())
                if amax <= 3:
                    ship0_n += 1
                    p_low.append(float(dist[:4].sum()))
                    p_spike.append(float(dist[amax]))
                    masked = sl[s, tgt].clone()
                    masked[:4] = -1e9
                    nb = int(masked.argmax())
                    cf_counts.append(min(int(SC[nb]), g))
                    cf_bin4 += int(nb == 4)
    mean = lambda x: sum(x) / len(x) if x else 0.0
    print(f"\n===== {label} =====  firing slots(g>4)={fire_n}  "
          f"ship0(argmax bin0-3)={ship0_n} ({100 * ship0_n / max(fire_n, 1):.0f}%)")
    if ship0_n:
        print(f"  low-bin choice: mean P(bins0-3)={mean(p_low):.2f}  mean P(argmax spike)={mean(p_spike):.2f}  "
              f"(spike on one probe size; rest diffuse over real bins)")
        print(f"  COUNTERFACTUAL mask 0-3: new ship count median={statistics.median(cf_counts):.0f} "
              f"mean={mean(cf_counts):.0f}  | -> bin4(=5 ships) {100 * cf_bin4 / len(cf_counts):.0f}%  "
              f"(garrison median {statistics.median(garr):.0f} -- affordable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--our-name", default="OURS", help="team name of our seat (e.g. Saheb)")
    ap.add_argument("--max-files", type=int, default=80, help="cap replays per dir (speed)")
    args = ap.parse_args()
    paths = []
    for dd in args.dirs:
        paths += sorted(glob.glob(dd + "/*.json"))[:args.max_files] if not dd.endswith(".json") else [dd]
    cfg = Config()
    sd, decode = load_checkpoint(args.checkpoint, cfg)
    model = EntityTransformer(cfg.model)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"loaded {args.checkpoint}  min_ship_bin={getattr(cfg.model, 'min_ship_bin', 0)} "
          f"game_phase={cfg.model.game_phase_features} decode={decode}")
    for res in ("LOST", "WON"):
        analyze(model, paths, args.our_name, res, res)


if __name__ == "__main__":
    main()
