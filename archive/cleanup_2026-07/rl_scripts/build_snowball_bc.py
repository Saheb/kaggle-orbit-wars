"""Build a full-policy (fire+ship+target) pairwise-15 BC dataset from snowball replays.

Phase 2 warmstart: clone the WINNER of each short-decisive ("snowball") game so the
agent starts with the aggressive-cohort prior — fast expansion, forward-staging, and
the empire-size-gated reinforce ramp (own-planet launches) the LB top tier uses. The
reinforce ramp is a LATE / large-empire behaviour, so we clone the FULL game (no
opening-only cap by default), not just the opening.

Two things this gets right that the older opening-only builders did not:
  * TIMING — the action at steps[t] was decided on the obs at steps[t-1] (verified
    100% launch-legality vs t-1 obs, 9.9% vs t). We pair obs[t-1] with action[t].
  * REINFORCEMENT — `trajectory_to_training_sample` resolves each launch's target
    planet from its aim angle, so own-planet (reinforce) launches become real target
    labels that train the model's `is_mine` target-input weight from step 0.

The output is a standard bc.py sample .pkl (pairwise-15, because features are computed
by the current features.py at build time — no feature dim is baked into the replay).

Usage:
  python orbit_wars_rl/build_snowball_bc.py \
      --replay-dir /tmp/snowball \
      --samples-out /tmp/snowball_bc.pkl \
      --summary-out /tmp/snowball_bc_summary.json
"""
import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orbit_wars_rl.bc import trajectory_to_training_sample


def _winner_seat(d, player=None, exclude=None):
    """Seat to clone for this game, or None to skip. Default = unique winner."""
    names = d.get("info", {}).get("TeamNames", [])
    rew = d.get("rewards") or []
    if not names:
        return None, names
    if player:
        return (names.index(player) if player in names else None), names
    if not rew or rew.count(max(rew)) != 1:   # need a unique winner
        return None, names
    me = rew.index(max(rew))
    if me >= len(names) or (exclude and names[me] == exclude):
        return None, names
    return me, names


def build(replay_dir, player=None, exclude=None, steps_max=0):
    samples = []
    n_games = 0
    subjects = Counter()
    n_fire_frames = n_reinforce_labels = n_target_labels = 0
    drop_no_planets = 0

    for f in sorted(glob.glob(os.path.join(replay_dir, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        steps = d.get("steps") or []
        if len(steps) < 5:
            continue
        me, names = _winner_seat(d, player, exclude)
        if me is None:
            continue
        n_games += 1
        subjects[names[me]] += 1

        t_end = len(steps) if steps_max <= 0 else min(len(steps), steps_max + 1)
        for t in range(1, t_end):
            if me >= len(steps[t]) or me >= len(steps[t - 1]):
                continue
            obs = steps[t - 1][me].get("observation")
            action = steps[t][me].get("action") or []
            if not obs or not obs.get("planets"):
                drop_no_planets += 1
                continue
            obs = dict(obs)
            obs["player"] = me                       # clone this seat
            sample = trajectory_to_training_sample({"obs": obs, "action": action})
            if sample is None:
                continue
            samples.append(sample)
            fired = (sample["fire_target"] == 1) & sample["slot_valid"].bool()
            if fired.any():
                n_fire_frames += 1
            planets = obs["planets"]
            tt = sample["target_target"]
            for slot in range(tt.shape[0]):
                if fired[slot] and int(tt[slot]) >= 0:
                    n_target_labels += 1
                    ti = int(tt[slot])
                    # reinforce label = fired slot whose resolved target planet is OURS
                    if ti < len(planets) and int(planets[ti][1]) == me:
                        n_reinforce_labels += 1

    summary = {
        "games_cloned": n_games,
        "subjects": dict(subjects.most_common(12)),
        "samples": len(samples),
        "fire_frames": n_fire_frames,
        "target_labels": n_target_labels,
        "reinforce_labels": n_reinforce_labels,
        "reinforce_share": (round(n_reinforce_labels / n_target_labels, 3) if n_target_labels else 0.0),
        "dropped_no_planets": drop_no_planets,
        "steps_max": steps_max,
        "mode": player or (f"winners (excl {exclude})" if exclude else "winners"),
    }
    return samples, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--player", default=None,
                    help="clone this named player regardless of outcome (default: each game's winner)")
    ap.add_argument("--exclude", default=None,
                    help="winner-mode: skip games won by this player")
    ap.add_argument("--steps-max", type=int, default=0,
                    help="cap cloned steps per game (0 = full game; keep 0 to capture the late reinforce ramp)")
    ap.add_argument("--samples-out", required=True)
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--game-phase-features", action="store_true",
                    help="Extract 15-global features (11->15: phase one-hot + comet-cycle) so the "
                         "pkl can init a 15-global Stage-B model. Comets are re-extracted either way.")
    args = ap.parse_args()

    if args.game_phase_features:
        # Set the flag on the EXACT features module that bc.extract_features reads (the script-dir
        # `features` import), not a second `orbit_wars_rl.features` copy — they have separate globals.
        import sys
        from orbit_wars_rl.bc import extract_features
        sys.modules[extract_features.__module__].set_game_phase_features(True)

    samples, summary = build(args.replay_dir, args.player, args.exclude, args.steps_max)
    os.makedirs(os.path.dirname(os.path.abspath(args.samples_out)), exist_ok=True)
    with open(args.samples_out, "wb") as fh:
        pickle.dump(samples, fh)
    summary["samples_out"] = args.samples_out
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        with open(args.summary_out, "w") as fh:
            json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
