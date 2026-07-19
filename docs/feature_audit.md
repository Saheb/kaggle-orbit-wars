# Feature audit — what's raw signal vs hand-computed conclusion (2026-07-19)

Motivation: the recurring §1 anti-pattern (writeup_lessons.md) — *"our ROI/deflation stack
hand-computes conclusions the winners let the model draw from a raw resolved timeline, and each
patch carried bugs."* Now that the K=24 timeline exists, many hand-computed scalars are **suspect
of redundancy**: the model already has the raw trace to derive them, at a horizon *it* picks
instead of our arbitrary ones. This categorizes every input so we can ablate the suspects (queue
#3b) and know what's safe to prune.

Legend: **KEEP** (raw state / exact geometry / winner-blessed) · **SUSPECT** (derivable from raw +
timeline; winners omit; often a *magic horizon*) · **REACTIVE** (encodes *potential* future
launches — NOT in the no-new-launch timeline, so not strictly redundant, but hand-tuned form).

## Planet — 20 base + 96 timeline

| ch | feature | category | note |
|---|---|---|---|
| 0,1 | norm x, y | **KEEP** | raw position |
| 2 | owner | **KEEP** | raw |
| 3 | radius | **KEEP** | raw |
| 4 | log ships (garrison) | **KEEP** | raw |
| 5 | production | **KEEP** | raw |
| 6,7 | is_orbiting, is_comet | **KEEP** | raw type |
| 8 | dist to sun | **KEEP** | raw geometry |
| 9 | orbital radius / comet-steps | **KEEP** | raw |
| **10,11** | **pred x,y (5-turn lookahead)** | **SUSPECT** | *magic 5-turn horizon*; model has pos+orbital_r+angular_vel → can predict any horizon itself (lesson 7) |
| **12,13** | **friendly / enemy pressure (untimed sums)** | **SUSPECT** | §1's named gap: untimed scalar sums the timeline now supplies *timed* (owner flips per step) |
| 14 | capture cost | **SUSPECT** | derivable from garrison+production |
| 15 | dist to nearest owned | **KEEP** | raw geometry |
| 16 | is_home | **KEEP** | raw |
| **17,18** | **owned within r=15 / r=30** | **SUSPECT** | *magic radii*; connectivity derivable from positions |
| 19 | active mask | **KEEP** | structural |
| 20–115 | **projected timeline** (mine/enemy/neutral + log-garrison × 24) | **KEEP** ⭐ | the winner-blessed raw future; everything above marked SUSPECT is a summary *of this* |

## Pairwise — 36 (per source×target)

| ch | feature | category | note |
|---|---|---|---|
| 0–3 | arrival sin/cos/dist/eta | **KEEP** | exact geometry |
| 4 | sun_safe | **KEEP** | geometry |
| 5–7 | is_mine/enemy/neutral | **KEEP** | raw |
| 8 | target production | **KEEP** | raw |
| 9 | valid | **KEEP** | mask |
| 10 | ships-at-arrival (garrison + prod·eta) | **SUSPECT** | the timeline's projected garrison *at the arrival step* is this, resolved |
| 11 | capture-gap | **SUSPECT** | ships-at-arrival − cost; both derivable |
| **12,13** | **roi_20, roi_50** | **SUSPECT** ⭐ | *the ones you flagged.* Value over a *magic* 20/50 horizon; K=24 timeline gives the full trace so the model picks its own horizon (§1 + lesson 7) |
| 14 | enemy_contest (enemy inbound sum) | **SUSPECT** | inbound fleets are resolved into the timeline already |
| 15 | reachable_enemy_mass | **REACTIVE** | distance-decayed enemy garrison that *could launch* — not in the no-launch timeline |
| 16 | capture_value_40 | **SUSPECT** | *magic 40 horizon* value; derivable |
| 17 | reactive_roi_40 | **REACTIVE**/SUSPECT | reactive cost, but *magic 40* + hand-tuned |
| 18 | friendly_reachable_mass | **REACTIVE** | friendly garrison that *could* support — potential, not in-flight |
| 19 | keepability_margin | **REACTIVE**/SUSPECT | friendly support − enemy reaction; hand-tuned combo |
| **20** | **enemy_mass_soon** | **SUSPECT** | *magic `_THREAT_ETA_WINDOW=6`*; timeline has the timing |
| 21 | threat_imminence (1/min-eta) | **SUSPECT** | min enemy eta; timeline has it |
| 22–25 | resolved intent sizes (cap/defend/maintain/all-in) | **RESOLVER** | sizing hints (Arm B kept as features while deleting the action gates) — derivable but cheap |
| 26–31 | **candidate target counterfactual** | **KEEP** ⭐ | Ender's most-credited feature (outcome *assuming this launch*) |
| 32–35 | candidate source counterfactual | **KEEP** ⭐ | source-side of the same |

## Fleet — 13 & Global — 15 (+48 econ)

- **Fleet 13** (pos, owner, angle, ships, speed, dist-to-sun, dest ETA/dist, threatens_owned,
  target_production, mask): **KEEP** — raw fleet state + geometry.
- **Global 15** (player, step, angular_velocity, economy stats, enemy split on-planets/in-fleets,
  mode, game-phase 11–14): **KEEP** — raw scalars. **+48 econ** (projected prod/material Δ ×24):
  **KEEP** ⭐ (winner-blessed global series; being tested in `econblock`).

## The pattern & what to do

**The SUSPECT column is one family:** hand-computed value/threat conclusions at *arbitrary
horizons* (5, 6, 15, 20, 30, 40, 50) — each a guess at "how far ahead to look" that the K=24
timeline makes unnecessary (the model pools its own horizon; SimJeg/Yijie 1-D-CNN over the series).
Candidates to ablate, in rough priority:
1. **roi_20 / roi_50 (12,13)** — you're right to doubt these; clearest redundancy.
2. **planet pressure 12,13 + pred 10,11 + owned-within 17,18** — untimed/magic summaries the
   timeline supersedes.
3. **enemy_mass_soon 20 / threat_imminence 21 / capture_value_40 16 / ships-at-arrival 10 /
   capture-gap 11** — same family.

**REACTIVE features (15,17,18,19) are the exception** — they encode what the enemy/we *could
launch*, which the no-new-launch timeline does NOT contain. Don't lump them with roi_20/50; if we
prune, the reactive signal may need to survive in some form.

**Test cheaply (queue #3b, "delete the magic horizons"):** remove the SUSPECT scalars and check if
it's a NO-OP. If the timeline truly subsumes them, removal costs nothing and simplifies the input;
if it hurts, the timeline (or its linear encoder, #5) isn't extracting them — also worth knowing.
Pair with the friendly_contest ablation (does roi-deflation even land?) for a full "are our
hand-computed conclusions doing anything" verdict.
