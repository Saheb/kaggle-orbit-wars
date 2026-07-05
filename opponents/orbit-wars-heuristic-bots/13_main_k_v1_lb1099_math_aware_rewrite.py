import math
import os
import time
from collections import defaultdict, namedtuple

# ============================================================
# main_k — FAST-KILL elimination overlay on top of 14_4c
# ============================================================
# Goal: ELIMINATE one enemy at a time as fast as possible.
# - 2P: target the single enemy (focus_enemy_2p already in 14_4c).
# - 4P: pick a focus_enemy_4p = enemy currently attacking us most (their
#   bases are stripped, easy targets) OR weakest with reachable planets if
#   nobody is actively attacking.
#
# Dedicated handler `handle_eliminate_focus` runs AFTER expand and BEFORE
# hammer. For each focus-enemy planet (sorted easiest-first), it finds the
# closest source able to solo it and fires. This guarantees we attempt at
# least one elimination strike per turn regardless of expand's scoring.
K_ELIMINATE_ENABLED = True
K_ELIMINATE_MAX_TRAVEL = 30     # hard ceiling: travel time bounded by game scale
# All other gates are math-derived from world state — see handle_eliminate_focus.
# Removed: K_4P_KINGMAKER_LEADER_SHARE (now: leader_prod > my_prod * 1.5)
#          attacker_threshold (now: max(20, my_prod * 8))
# 4P focus picker: prefer enemy that's attacking us. Falls back to weakest
# reachable. Avoid finishing the absolute weakest if doing so makes a leader
# dominant — using state-derived kingmaker check (leader_prod > my_prod*1.5).

# K2 (intel-aware decision gates, 4P-only):
# - Don't drain a friendly source on NEUTRAL expansion when that source is the
#   closest defender to one of OUR planets currently under heavy attack. Saves
#   it for next-turn reinforcement (matches user's "send ships to neutral when
#   a planet wants support" mistake).
# - Use enemy_planet_attackers info: when an enemy planet is under attack by
#   ANOTHER enemy, it's a softer target — boost in expand ranking, tighten
#   reactive-emit estimate in eliminate-focus (those ships defend instead).
# - In eliminate-focus reactive math, count ships the focus enemy has ALREADY
#   committed (in flight against us OR against other enemies) — those ships
#   are not available for reactive emit.
K2_DEFENSE_RESERVE_SOURCES_ENABLED = True
K2_DEFENSE_RESERVE_4P_ONLY = True                # 2P uses focus_enemy_2p bias
# Threat threshold = my_prod * THREAT_MIN_FACTOR. State-derived — no absolute
# floor; on a tiny board with my_prod=1, a 3-ship probe is still meaningful.
K2_DEFENSE_RESERVE_THREAT_MIN_FACTOR = 3
K2_DEFENSE_RESERVE_NEAREST_K = 1                 # protect K closest sources per threatened planet
K2_ENEMY_VS_ENEMY_EXPAND_BONUS = 4.0             # base magnitude; actual subtraction scales by attack/garrison ratio (K4-B)
K2_ENEMY_VS_ENEMY_4P_ONLY = True
# K2 eliminate reactive math improvement (additive — keeps existing 1.15 overkill,
# the 0.5-0.4*commit emit_frac formula, and the eta+3 reach window).
K2_ELIMINATE_REACTIVE_TIGHTER = True
K2_ELIMINATE_DERANK_RESERVED_SOURCES = True      # de-rank defense-reserved sources for eliminate

# K3 (safety + exposure-aware decisions, state-derived — no magic constants):
# - K3-A: minimum-planet floor for expand. State-derived as HALF our fair
#   share (total_inhabitable_planets / num_players * 0.5). Below that, override
#   ALL stop_expand gates so neutrals are always allowed. Production-starvation
#   = elimination spiral. The floor scales with map size and player count.
# - K3-B: drop the my_planets < 3 hard gate in handle_eliminate_focus. The
#   per-source `avail < required` check inside the loop already prevents
#   wasted strikes when we have no surplus.
# - K3-C: exposure-aware max_fires — smooth interpolation of max_fires from
#   1/3 of idle (at 0% exposure) to ALL idle (at 100% exposure). No magic
#   threshold; the focus enemy's commitment ratio drives the curve directly.
# - K3-D: focus picker weighs the focus enemy's outgoing-at-other-enemies
#   commitment. Same units as existing inflight_at_us term (divide by 8).
K3_ENABLED = True

# K6 (replay-derived send-fraction governor, 2026-05-20):
# Analysis of 200 bovard top-10% replays (~117k launches) vs 250 vickimar
# replays (~113k launches, rating 1000+ submissions) showed vickimar
# consistently OVER-SENDS by ~9pp across all game phases. Concretely
# (winners-only, median send_ratio = ships_sent / src_ships_before_launch):
#       phase           bovard   vickimar   gap
#       early (<50)     0.71     0.77       +6pp
#       mid (50-200)    0.58     0.67       +9pp
#       late (>=200)    0.37     0.47       +10pp
# Bovard winners' median src_ships_before is 52 vs ours 42 — top bots
# maintain ~25% larger garrisons. Self-reinforce launches (61% of all
# winner launches) median 0.50 ratio for bovard vs ours 0.60.
#
# K6-A: phase-based soft cap on send fraction applied at _commit_fleet
# (the single chokepoint for ALL launches). Self-reinforce launches get
# the tightest cap; enemy/neutral captures and mega-hammers get a looser
# cap (full force still allowed). The curve below sits slightly ABOVE
# bovard medians to avoid over-correcting; goal is to halve the gap, not
# eliminate it.
#
# K6-B: never reduce below MIN_DISPATCH_SHIPS — a tiny launch beats no
# launch. Also disabled for ships <= 30 (small fleets are already
# fraction-noise; the cap matters when we're stripping high-prod
# stockpile sources of 100+ ships).
K6_REPLAY_SEND_CAP_ENABLED = False
K6_REPLAY_SEND_CAP_MIN_SRC_SHIPS = 30   # below this, cap doesn't kick in
K6_REPLAY_SEND_CAP_MIN_OUTPUT = 5       # never reduce below this many ships
# Caps per (phase, target_type). target_type is "self", "enemy", "neutral".
# Self caps are tightest; enemy/neutral are looser (preserve attack force).
#
# IMPORTANT: enemy captures get cap=1.0 (effectively no cap). aim_at_target
# computes angles based on fleet_speed = f(log(ships)); reducing ships after
# aim breaks angle-vs-arrival timing. For self-reinforce and neutral targets
# the planet is on our side / stationary-ish so small timing errors are
# absorbed. Enemy captures require precise timing — leave untouched.
K6_REPLAY_SEND_CAPS = {
    # (turn_max_exclusive, target_type) -> max send_fraction
    "early_self":      (50,   "self",    0.85),
    "early_enemy":     (50,   "enemy",   1.00),
    "early_neutral":   (50,   "neutral", 0.92),
    "mid_self":        (200,  "self",    0.75),
    "mid_enemy":       (200,  "enemy",   1.00),
    "mid_neutral":     (200,  "neutral", 0.88),
    "late_self":       (350,  "self",    0.65),
    "late_enemy":      (350,  "enemy",   1.00),
    "late_neutral":    (350,  "neutral", 0.85),
    "terminal_self":   (9999, "self",    0.55),
    "terminal_enemy":  (9999, "enemy",   1.00),
    "terminal_neutral":(9999, "neutral", 0.80),
}


def k6_send_cap(turn, target_type):
    """Returns max send-fraction allowed for a launch at this turn against
    this target_type (self/enemy/neutral). Derived from bovard top-10%
    winners' median send_ratio at turn-bands."""
    if turn < 50:
        band = "early"
    elif turn < 200:
        band = "mid"
    elif turn < 350:
        band = "late"
    else:
        band = "terminal"
    key = f"{band}_{target_type}"
    entry = K6_REPLAY_SEND_CAPS.get(key)
    if entry is None:
        return 1.0
    return entry[2]


# ============================================================
# 14_4a — 2P focus-enemy bias
# ============================================================
# Adds a strong distance/score bonus toward the SINGLE enemy in 2P matches,
# so our targeting concentrates offense instead of scattering. 4P unchanged
# (focus on weakest in 4P loses to kingmaker dynamic; main_k v3 A/B
# confirmed 4P regression to 36% 1st rate when focus enabled in 4P).
F14_4A_2P_FOCUS_ENABLED = True
F14_4A_2P_FOCUS_DIST_BONUS = 18.0   # 14_4c: bumped from 12
F14_4A_2P_FOCUS_HAMMER_BONUS = 20.0
F14_4A_2P_FOCUS_MEGA_BONUS = 100

# ============================================================
# Game constants (fixed by the kaggle env)
# ============================================================

BOARD = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_R = 10.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
MAX_SPEED = 6.0
TOTAL_STEPS = 500
SIM_HORIZON = 110
FWD_SIM_FILTER_ENABLED = True   # V12.9: forward-sim 4P capture filter
FWD_SIM_HORIZON = 7             # 4P-only horizon — peak at h=8 (28.1% seeded n=320)
FWD_SIM_DEFENSE_CHECK = True    # V12.9: re-enable with mature Melis system
FWD_SIM_RANK_BONUS_4P = 0.0     # weight applied to forward_score gain when ranking
                                # candidates in 4P. 0=disabled, ~0.5 = strong bias.
SEARCH_EXPAND_4P_ENABLED = True # V12.9 Melis search: depth-1 step-action picker
                                # (4P-only). Replaces handle_expand's per-source greedy
                                # with global step-action ranking by Melis evaluator.
SEARCH_EXPAND_2P_ENABLED = True # V12.9 candidate: enable in 2P too
SEARCH_MAX_PER_SOURCE = 3       # candidates per source planet to evaluate
SEARCH_MAX_ACTIONS_TO_PICK = 5    # top step actions to commit per turn (peak in 4P)
SEARCH_MAX_ACTIONS_TO_PICK_2P = 7 # 2P has fewer planets — try more captures
SEARCH_DISABLES_CHEAP_PICKUP = True  # skip handle_cheap_pickup when search runs
HAMMER_MELIS_VERIFY = True      # V12.9 candidate: verify hammer plans via forward sim
SEARCH_DEPTH2_ENABLED = False   # paranoid opponent ply on top-3 actions
# V13.3 N4: _neutral_blocked_by_cap uses effective_garrison projection so a
# heavy neutral about to be wounded by enemy fleet is unblocked.
NEUTRAL_CAP_USES_EFFECTIVE_GARRISON = True
NEUTRAL_CAP_LOOKAHEAD = 10       # turns to project for effective garrison
# V13.3 N6: _capture_holds_against_snipe uses effective_garrison for pre_garrison
N6_USE_EFFECTIVE_PRE_GARRISON = True
# V13.3 T1 (terminal-phase override): force pressure mode in last K turns.
TERMINAL_PHASE_ENABLED = True
TERMINAL_PHASE_TURNS = 30
# V13.3 F1 (fleet-intent): hammer-target bonus for recently-active enemy sources
FLEET_INTENT_ENABLED = True
FLEET_INTENT_MIN_DROP = 8       # ship-count drop (after production) to flag launch
FLEET_INTENT_HAMMER_BONUS = 5.0 # score boost for recently-launched targets
# V13.3 F1b: extend F1 fleet-intent to expand-target ranking. When an enemy
# planet just shed ships, it's a soft target — vulnerable AND can't reinforce
# from its own depleted garrison. Subtract from `weighted` distance so it
# ranks higher in expand candidates. Smaller magnitude than hammer bonus
# (expand fires more often; over-promoting causes thrash).
F1B_EXPAND_BONUS_ENABLED = True
F1B_EXPAND_BONUS = 3.0   # subtracted from weighted distance
# V13.3 R1: detect planets just flipped from us → enemy; bigger hammer bonus
# (fresh enemy capture leaves tiny defender).
R1_RECAPTURE_PRIORITY_ENABLED = True
R1_RECAPTURE_HAMMER_BONUS = 8.0
# V13.3 E2: _endgame_roi_ok compares against garrison cost, not total ships sent
E2_USE_GARRISON_THRESHOLD = True
# V13.3 SO1: static outer planets (corners under 4-fold symmetry) are permanent
# real estate. Boost their target attractiveness in target prefilter ranking.
SO1_STATIC_PREFERENCE_ENABLED = True
SO1_STATIC_BONUS = 2.179862   # subtracted from weighted distance (smaller = more attractive)
SO1_STATIC_BONUS_2P = 2.179862    # V14.0 CMA-ES 2P best
SO1_STATIC_BONUS_4P = 2.95474    # V14.0 CMA-ES 4P best
# V13.3 SP1: speed-aware dispatch. Long-distance fleets benefit from being
# larger (faster). Bump fleet size when target is far; graceful fallback.
SP1_SPEED_AWARE_ENABLED = True
SP1_LONG_DIST_THRESHOLD = 27.637375  # raw distance threshold for "long"
SP1_LONG_DIST_SHIPS = 22         # min fleet size for long-distance captures
# V13.3 TI1: tie-for-win endgame conservatism. Engine reward = 1 for every
# player tied at max ship-sum (planets + fleets in flight). If we're trailing
# the leader in late game, low-margin attacks that flip to losses HURT our
# absolute ship sum (we lose X, enemy loses X — relative neutral but absolute
# drops, which matters because tie-with-leader counts as a win). Conserve by
# requiring HIGHER margin on launches when trailing during the endgame window.
TI1_TIE_FOR_WIN_ENABLED = True
TI1_HORIZON_TURNS = 25           # last K turns where TI1 gates apply
TI1_REQUIRED_EXTRA_MARGIN = 5    # add to normal margin when trailing
TI1_TRAILING_GAP_MIN = 10        # only kick in if behind by ≥ this many ships
# V13.3 AS1: 4P anti-second-place combat avoidance. Engine resolves same-eta
# arrivals via top-minus-second; second-place owner LOSES ALL ships. In 4P,
# multi-enemy fleets often converge on the same target on the same turn. If
# our fleet would land alongside a larger enemy fleet, we get wiped. Skip
# such launches. 4P-only — 2P existing N1/snipe checks cover the 1-enemy case.
AS1_ANTI_SECOND_ENABLED = True
# V13.3 P3 (failTolerant + wereCleaned): stamp owner_at_commit on every
# commitment; drop commitments where target ownership has changed (situation
# is different from what we planned for, freeing main to re-evaluate). Port of
# zvold Simulator.failTolerant + wereCleaned. Detects: neutral target snatched
# by enemy → our commit may be inadequate against fresh enemy production.
FAILTOLERANT_ENABLED = True
# V13.3 P2 (Sanity threshold stand-pat, 2P-only): if the best candidate step
# action's score gain over no-op is below MELIS_SANITY_THETA, return [] from
# search_step_action — stand pat. Per wankongyew "Knights of the Cardboard
# Castle". 2P-only gate per [[format-aware-blends]].
MELIS_SANITY_ENABLED = True
MELIS_SANITY_THETA = 3.0
# F16 (V12.9 candidate): candidate diversity — augment per-source candidate
# pool with the HIGHEST-PRODUCTION reachable target alongside top-N-closest.
# Currently generate_step_actions only emits top-3-by-distance per source, so
# strategic captures of slightly-farther high-production planets never get
# evaluated by Melis. Disabled by setting F16_PROD_PICKS = 0.
F16_DIVERSITY_ENABLED = True
F16_CLOSEST_PICKS = 2   # top-N closest still emitted
F16_PROD_PICKS = 1      # additional highest-production picks (not in closest set)
# F1 (V12.9 candidate): per-turn aggregation — score is mean of forward_score
# at multiple horizons instead of a single end-state read. Penalises fragile
# spike paths (briefly ahead then loses), rewards stable lead.
FWD_SCORE_AGG_ENABLED = True
FWD_SCORE_AGG_TURNS = (4, 8, 14, 20)

# ============================================================
# Strategy knobs (V11.6 patient design)
# ============================================================

PSM_OPENING_TURN = 14
PSM_OPENING_TURN_2P = 14    # V14.0 CMA-ES 2P best
PSM_OPENING_TURN_4P = 10    # V14.0 CMA-ES 4P best

# Mode 1 — Absorb / reservation walk
ABSORB_MIN_THREAT = 3            # incoming hostile fleets <this many ships are noise
ABSORB_PROJECTION_MARGIN = 0     # running balance must stay >= this to "survive"

# Mode 2 — Defense
DEFENSE_OVERSEND = 1             # V12.8et: was 2 — less defense overshoot
DEFENSE_OVERSEND_2P = 1    # V14.0 CMA-ES 2P best
DEFENSE_OVERSEND_4P = 0    # V14.0 CMA-ES 4P best
DEFENSE_COALITION_MAX = 2        # rescue coalition cap

# Global fleet-quality floor — the patient ethos says ONE main fleet at the
# target, not many drips. Fleets below this size fly slowly (speed≈1.0), miss
# orbital targets, and waste tempo. Skip the dispatch entirely if the right-
# sized fleet would be smaller than this.
MIN_DISPATCH_SHIPS = 8           # V12.8am: was 5 — manual flags <8 ships as dead-zone slow speed
# F3 (V12.9 candidate): three-bucket dispatch floor.
# SAFE bucket: short-hop neutral pickup → loose floor (matches v13 2P-short, extended to 4P).
# HARD bucket: enemy-owned with strong garrison → tighter floor (avoid wasted hammer attempts).
F3_THREE_BUCKET_ENABLED = True
F3_SAFE_FLOOR = 5
F3_SAFE_DIST = 12.0
F3_HARD_FLOOR = 14
F3_HARD_GARRISON = 14

# Mode 3 — Expand
EXPAND_K_OPENING = 2             # turns 0..PSM_OPENING_TURN: examine 2 nearest
EXPAND_K_MID = 1                 # mid-game: examine ONLY the absolute nearest
EXPAND_MAX_TRAVEL_OPENING = 20
EXPAND_MAX_TRAVEL_MID = 14
EXPAND_MIN_MARGIN = 0            # exact-+1 capture (needed_to_capture already adds the +1 to flip)
EXPAND_MIN_MARGIN_4P = 3  # F30: 4P capture leaves 6 garrison vs 1
# V13.3 X8b: in 2P, prefer need+0+X8B_2P_EXTRA ships per capture (small snipe buffer)
# but fall back gracefully if source doesn't have it. Avoids X8 cascade.
X8B_2P_EXTRA = 3
EXPAND_MIN_SHIPS = MIN_DISPATCH_SHIPS
# F31: 2P mid-game production floor. Skip prod=1 neutrals when prod>=2 are still available.
# Stops us grabbing garage-sale prod=1 planets while opponents cherry-pick prod=5.
EXPAND_MIN_PROD_2P = 2

# V12.3c5 (2.5) hash-entropy tiebreak. In 2P, near-equal-distance candidates
# get reordered by a deterministic hash so two mirrored PATIENT bots don't
# always pick the same target. Replayable since the hash is salted on (player,
# step, src, target) only.
TIEBREAK_ENABLED = True
TIEBREAK_EPS_FRAC = 0.005   # 0.5% of best distance defines the tie bucket
TIEBREAK_EPS_MIN = 1.439234      # V12.8dz: was 0.5 — wider tiebreak floor

# V12.4a — Rotation-aware target ranking. _nearest_targets sorts by
# distance-to-predicted-position at expected travel time, not raw current
# distance. Static unchanged; orbital rotating-toward-us promotes,
# rotating-away demotes. Pure re-ranking — does not change which fleets
# fire, only WHICH targets get inspected when K is small. Toggle via
# V124_ROT_AWARE=0 to ablate.
ROT_AWARE_RANK_ENABLED = os.environ.get("V124_ROT_AWARE", "1") != "0"

# V12.6a — Value-weighted target ranking. After rotation-aware effective
# distance, subtract VALUE_WEIGHT_{2P,4P} * target.production. Higher-prod
# targets rank earlier — bot prefers strategic captures within reach.
# V12.6b: format-split — 2P uses 4.0 (200-game test +10 wins vs 12_6a),
# 4P uses 2.0 (4.0 hurt 4P -7.8pp by forcing K=1 to chase far high-prod).
VALUE_WEIGHT_2P = 4.86118
VALUE_WEIGHT_4P = float(os.environ.get("V126_VALUE_WEIGHT_4P", "2.0"))

# V12.4b — Anti-snipe veto (2P-only). Before firing on a NEUTRAL, simulate
# post-capture surplus + production growth vs known incoming enemy fleets;
# refuse if balance ever drops <=0. 2P-only because the 4P 192-game test
# showed -1.9pp 1st rate AND a structural regression (55 third-place
# finishes vs 12_4a's 4) — with 3 enemies, "some enemy fleet incoming"
# is too easy to trigger and the bot starves itself of expansion.
ANTI_SNIPE_ENABLED = os.environ.get("V124_ANTI_SNIPE", "1") != "0"
ANTI_SNIPE_HORIZON = 25          # V12.8ar: was 25 — shorter window frees more captures
ANTI_SNIPE_2P_ONLY = False       # V12.8ct: was True — re-enable in 4P with shorter horizon=18

# V13.3 R8 (reactive-snipe projection): when evaluating "will capture hold",
# also project enemy REACTIVE launches (not just fleets already in flight).
# Fixes the case where we send X ships at a contested neutral, capture by 1,
# then enemy sends a small follow-up to snipe back. The fix is in EVALUATION:
# the move only scores +EV if it survives a projected enemy response.
REACTIVE_SNIPE_PROJECTION_ENABLED = True
REACTIVE_EMIT_FRAC = 0.49629        # fraction of enemy ships projected to react
REACTIVE_MIN_ENEMY_SHIPS = 5     # ignore enemy planets below this — too small to matter
REACTIVE_MIN_PROJECTED = 3       # minimum projected force per enemy (avoid degenerate 0s)
# V13.3 S2 (sun-shadow): enemies whose direct path to target is sun-blocked
# can't launch at the target this turn — exclude them from reactive-snipe
# projection so we don't over-pessimize and skip viable captures.
SUN_SHADOW_REACTIVE_FILTER = True

# V12.4c — Counter-snipe priority (2P-only). Prepend neutrals where a known
# enemy fleet WILL capture before us, with cheap re-flip plans (size = post-
# enemy-capture defender + production*delay + 1). 2P-only because 4P 192-game
# test showed -14pp 1st rate vs noise floor — too many "enemy-committing"
# opportunities in 4P starve main expansion. Toggle V124_COUNTER_SNIPE=0.
COUNTER_SNIPE_ENABLED = os.environ.get("V124_COUNTER_SNIPE", "1") != "0"
COUNTER_SNIPE_2P_ONLY = False    # V12.8cu: was True — re-test in 4P under v2 harness
COUNTER_SNIPE_MAX_COST = 30
COUNTER_SNIPE_MIN_DELAY = 1
COUNTER_SNIPE_MAX_DELAY = 12

# V12.4d — Cheap-pickup pre-pass (4P-only). K=1 in 4P mid-game starves
# small free planets sitting next to a source whose K=1 nearest is a more
# expensive target (6.png: bot stockpiled to take 26 while 12+6 sat free).
# Pre-pass scans ALL reachable neutrals with garrison <= CHEAP_PICKUP_MAX_GARRISON
# and fires on the cheapest one before main expand. 4P-only — in 2P,
# K=4-6 mid-game already covers cheap targets and pre-pass would waste
# a source's budget on a 5-ship neutral when main expand had a better target.
CHEAP_PICKUP_ENABLED = os.environ.get("V124_CHEAP_PICKUP", "1") != "0"
CHEAP_PICKUP_4P_ONLY = True
CHEAP_PICKUP_MAX_GARRISON = 25
# F32: 4P cheap-pickup production floor. Skip prod=1 planets when prod>=2 are available.
CHEAP_PICKUP_MIN_PROD = int(os.environ.get("F32_CP_MIN_PROD", "2"))

# V12.8b — Endgame ROI gate for neutral captures. In the last
# ENDGAME_ROI_TURNS turns, only fire on a neutral if expected production
# growth exceeds the ships spent. A 12-ship neutral captured at turn 480
# with prod=1 nets -2 points by game end. Pure score-arithmetic; refuses
# captures that lose points. Only gates neutrals — hostile captures retain
# differential value (every ship denied to enemy still helps us). 4P-only
# (2P loses denial-of-opponent value if we skip). Set V128_ENDGAME_ROI=0
# to ablate.
ENDGAME_ROI_ENABLED = os.environ.get("V128_ENDGAME_ROI", "1") != "0"
ENDGAME_ROI_TURNS = 30

# V12.8cq — Multi-factor neutral capture utility (kovi-inspired). Only
# capture a neutral if expected production gain (production × remaining
# turns after capture) exceeds ship cost by NEUTRAL_TEMPO_THRESHOLD. Beyond
# ENDGAME_ROI (which is net>0 in last 30 turns), this filters marginal
# captures throughout the game. 4P-only.
NEUTRAL_TEMPO_FILTER_ENABLED = os.environ.get("V128_TEMPO_FILTER", "1") != "0"
NEUTRAL_TEMPO_THRESHOLD = 10     # V12.8cq'': was 15

# V12.8j — Late-game launch blackout. Manual: "Cease offensive expansion
# around turn 450-480; ships in flight will not arrive to grow production,
# so hoarding wins ties." Hard veto on expansion / cheap-pickup launches
# in the last LAUNCH_BLACKOUT_TURNS turns regardless of target. Defense
# and hammer continue. Both formats.
LAUNCH_BLACKOUT_ENABLED = os.environ.get("V128_LAUNCH_BLACKOUT", "1") != "0"
LAUNCH_BLACKOUT_TURNS = 10

# V12.8c — Neutral hard-cap with wounded-target watchlist (4P-only).
# Big neutrals (>NEUTRAL_HARD_CAP_4P ships) cost more than they pay back
# under K=1 mid-game; skip them UNLESS we observed their ship count drop
# last turn (someone else softened them). This converts opponent failed
# attacks into our scouting signal. 2P-disabled — duels need broader
# neutral access for tempo.
NEUTRAL_HARD_CAP_ENABLED = os.environ.get("V128_NEUTRAL_CAP", "1") != "0"
NEUTRAL_HARD_CAP_4P = 40          # legacy, kept for any 4P references
NEUTRAL_HARD_CAP_2P = 61          # V12.9 cap55: 2P-only — strict ignore neutrals >=55 ships
NEUTRAL_WATCHLIST_MIN_DROP = 5  # ignore <5-ship dips (production noise)
# V14.2c (user direction): skip costly low-prod NEUTRAL targets entirely.
# A prod=1 planet earning 1 ship/turn needs >=15 turns to repay a 15-ship
# capture cost — usually not worth it. Cheap (<15 ship) prod=1 neutrals
# are still allowed via cheap-pickup. After higher-prod expansion runs out
# we should ATTACK enemies, not flop into prod=1 expansion.
LOW_PROD_NEUTRAL_SKIP_ENABLED = True
LOW_PROD_NEUTRAL_SKIP_PROD = 1       # apply to neutrals with production <= this
LOW_PROD_NEUTRAL_SKIP_GARRISON = 14  # skip if neutral.ships >= this (iter 1: 15 → 4P -11pp)

# V12.8d — Weakest-enemy targeting (4P-only). Subtract a bonus from the
# effective distance of targets owned by the lowest-prod-share enemy so
# they rank earlier. BUT once that enemy's prod-share drops below
# WEAKEST_DONT_FINISH_SHARE, *de-prioritize* their planets — a near-dead
# opponent absorbs aggression from the leader, so leaving them alive is
# strategically valuable (balance-of-power). Skip in 2P.
WEAKEST_TARGET_ENABLED = os.environ.get("V128_WEAKEST_TARGET", "1") != "0"
WEAKEST_TARGET_BONUS = 2.0      # subtracted from effective distance for weakest's planets
WEAKEST_TARGET_MIN_STEP = 60    # don't disrupt opening neutral grabs
WEAKEST_DONT_FINISH_SHARE = 0.05
WEAKEST_DONT_FINISH_PENALTY = 12.0  # added to effective distance once below threshold

# V12.8s — 4P leader-bashing. If we're not the leader AND someone has a
# clear lead (their lead_score / ours >= LEADER_BASH_RATIO), prefer
# attacking the leader's planets. "Kingmaker" 4P move — keeps the leader
# from running away while we can still catch up. 4P-only.
LEADER_BASH_ENABLED = os.environ.get("V128_LEADER_BASH", "1") != "0"
LEADER_BASH_RATIO = 1.3
LEADER_BASH_BONUS = 4.0
LEADER_BASH_MIN_STEP = 60   # don't disrupt opening

# Mode 3b — Coalition expand (neutrals only; enemies route through hammer).
# Patient ethos: prefer ONE big solo fleet over two coalition fleets. Coalition
# only fires when a target genuinely can't be soloed by any one source AND the
# target is large enough that splitting is unavoidable.
COALITION_ENABLED = True
COALITION_MAX_PARTICIPANTS = 3   # solo + 2 partners maximum
COALITION_NEUTRALS_ONLY = False  # V12.2 R1b: allow enemy-target coalitions
COALITION_MAX_TRAVEL_BONUS = 2   # partner can be slightly further than solo cap
COALITION_MIN_PER_CONTRIBUTOR = 15   # no tiny 5-ship "halves" — minimum substantive piece
COALITION_MIN_PER_CONTRIBUTOR_2P = 15    # V14.0 CMA-ES 2P best
COALITION_MIN_PER_CONTRIBUTOR_4P = 5    # V14.0 CMA-ES 4P best
COALITION_MIN_TARGET_SHIPS = 20      # V12.6d: was 20 — allow 2-source coalitions on medium neutrals

# Mode 4 — Hammer
HAMMER_ENABLED = True
HAMMER_STOCKPILE_MIN = 50
HAMMER_TARGET_PROD_MIN = 2
HAMMER_PROD_SHARE_TRIGGER = 0.40
HAMMER_OVERKILL_RATIO = 1.30
HAMMER_SURROUNDED_PROMOTE_TURNS = 10  # idle this many turns => permanent stockpile
HAMMER_MAX_TRAVEL = 24                # hammers reach further than expansion
HAMMER_ABORT_OVERRUN_RATIO = 1.329521     # if defender exceeds committed x this, abort
HAMMER_PLAN_REVALIDATE_INTERVAL = 1   # re-check defender every turn
HAMMER_MIN_PER_CONTRIBUTOR = 9        # drop tiny stockpile contributions

# V14.1c (Phase 3.3): Mega-hammer — single-source overwhelming strike.
# When ANY of our planets has accumulated MEGA_HAMMER_SHIPS_MIN+ ships and
# there's an enemy target with weak garrison within MEGA_HAMMER_MAX_TRAVEL
# turns, send the ENTIRE garrison as one huge fleet. Exploits the fleet-speed
# log formula: 1000 ships travel at max speed 6, 100 ships at ~3.72 — bigger
# fleets arrive faster, making reactive snipe impossible. Fires BEFORE
# handle_hammer in agent dispatch so a winning mega-strike doesn't get
# partitioned by the multi-stockpile coalition logic. See RL/notes/phase3_huge_hammer.md.
MEGA_HAMMER_ENABLED = True
# V14.1c iter b — restrict to 4P. Initial iter (a) had MEGA_HAMMER_4P_ONLY=False
# and showed 2P regression -19 wins (stripping sources leaves them open to
# counter-snipe in 2P duels). 4P still wins +11.5pp at this setting.
MEGA_HAMMER_4P_ONLY = True
MEGA_HAMMER_SHIPS_MIN = 300           # V14.1n: reverted CMA-ES 360 -> 300 (n=400 2P regress)
MEGA_HAMMER_TARGET_GARRISON_MAX = 80  # V14.1n: reverted CMA-ES 95 -> 80
MEGA_HAMMER_MAX_TRAVEL = 40           # V14.1n: kept (CMA-ES picked 39, kept default)

# V14.1d (Phase 3.3, Idea 3) iter h — production-aware mega-hammer trigger.
# Game mechanic: planet production is 1-5/turn. Prod-5 planets accumulate
# ships fastest. Iters a-g (withholding ships) all regressed 4P because they
# weakened routine offense. Now: don't withhold ANY ships. Instead, lower
# the mega-hammer firing threshold for naturally fast-growing planets.
# A prod-5 planet hits 250 ships ~3.7× faster than a prod-1 — let it strike
# at 200, while a prod-2 needs 350. The mega "grows in steps" naturally
# via planet production, and fires sooner from fast-growers.
PROD_RESERVE_ENABLED = False          # disabled iter g approach
# Per-production threshold: mega fires when source.ships >= threshold for its prod tier.
# Higher prod = lower threshold (faster regrowth means strike payback is quicker).
MEGA_HAMMER_THRESHOLD_BY_PROD = {5: 200, 4: 250, 3: 300, 2: 350, 1: 400}
# V14.2 (Phase 3.6, Idea 6a): fresh-capture inheritance.
# Planets we captured in the last FRESH_CAPTURE_MAX_AGE turns use a lower
# mega-hammer threshold — they carry momentum from the capturing fleet and
# should chain to the next nearest enemy without waiting for full
# accumulation. Engine: bigger fleets arrive faster (log-speed formula),
# so chaining a 200+ ship strike from a newly-captured planet exploits
# the same speed advantage as MEGA_HAMMER.
FRESH_CAPTURE_INHERITANCE_ENABLED = True
FRESH_CAPTURE_MAX_AGE = 5                  # turns since capture
MEGA_HAMMER_SHIPS_MIN_FRESH = 200          # lower threshold for fresh captures

# V14.2 (Phase 3.6, Idea 6b): mega-hammer concentration.
# Cap firings per turn to force focusing one biggest strike rather than
# spreading multiple medium hammers. Engine: log fleet-speed → 1× 500
# arrives faster than 2× 250; harder to absorb in tied combat.
MEGA_HAMMER_CONCENTRATE_ENABLED = True
MEGA_HAMMER_MAX_PER_TURN = 1               # fire at most this many mega-hammers / turn
# B2: forward-sim verify mega-hammer plan before committing. Mirrors
# HAMMER_MELIS_VERIFY for handle_hammer. Without this, mega-hammer can fire
# 250 ships at a planet that flips on arrival but is recaptured next turn
# (our garrison drops to 0; enemy production retakes). 250 ships wasted.
MEGA_HAMMER_MELIS_VERIFY = True
# B2: opp emit fraction for verify projection. Iter 1 (0.30, mirrors
# HAMMER_MELIS_VERIFY) gave 4P +3.2pp; iter 2 (0.15, less pessimistic) was
# strictly worse — the verify needs the conservative 0.30 to actually filter
# bad strikes.
MEGA_HAMMER_VERIFY_OPP_EMIT = 0.30

# V14.2 (Phase 3.8): hammer no-threat oversend.
# When the LAST coalition contributor has no incoming enemy threat, skip
# trim and send full avail. Bigger fleets are faster (log fleet-speed) and
# harder to absorb. User-observed: bot was trimming 58→22 of needed
# share; safe planets should send everything.
# iter v1+v2 (both fmts): 2P +18/+40 wins, 4P CRASH 28.6%/25.5%.
# 4P regress: stripping a source in 4P leaves it vulnerable to 3 enemies.
# Gating to 2P-only.
HAMMER_NO_THREAT_OVERSEND_ENABLED = True
HAMMER_NO_THREAT_OVERSEND_2P_ONLY = True
# V14.2 (Phase 3.10 v1): ALWAYS-OVERSEND-2P REGRESSED (-8 wins). Stripping
# marginal sources hurts vs counter-snipe. Disabled; trying safe-surplus.
HAMMER_ALWAYS_OVERSEND_2P = False
# V14.2 (Phase 3.10 v2): safe-surplus oversend.
# Skip trim only when source has BOTH: avail > required * X (large surplus,
# stripping is safe) AND threat is small relative to garrison (won't be
# counter-sniped). Applies to both 2P and 4P. Bigger fleets win combat;
# fresh-capture inheritance chains from captured planet.
HAMMER_SAFE_SURPLUS_OVERSEND_ENABLED = True
HAMMER_SAFE_SURPLUS_RATIO = 2.0  # avail > required * this → "large surplus"
HAMMER_OVERSEND_MAX_THREAT_RATIO = 0.3  # threat <= garrison * this → safe

# V14.2 (Phase 3.7, Idea 6c): accumulator. Feed surplus from safe backline
# planets to the lead stockpile each turn. Over multiple turns, the lead
# accumulates a massive force; mega-hammer fires it as one overwhelming
# strike. Engine: log fleet-speed means 1× 1000 arrives faster than 4×
# 250, and combat math favors the bigger arriver. 4P-only initially.
ACCUMULATOR_ENABLED = True
ACCUMULATOR_4P_ONLY = True                  # 2P is sensitive to source-stripping
ACCUMULATOR_TURN_MIN = 15                   # skip early-game expansion phase
ACCUMULATOR_LEAD_MIN_SHIPS = 100            # lead must have >= this to be worth feeding
ACCUMULATOR_LEAD_THREAT_RATIO = 0.5         # lead under threat >= this fraction of ships → skip
ACCUMULATOR_FEEDER_MIN_SURPLUS = 30         # feeder must have >= this surplus to spare
ACCUMULATOR_FEEDER_KEEP_RESERVE = 30        # feeder keeps this many for defense
ACCUMULATOR_FEEDER_MAX_TRAVEL = 30          # feeder travel limit
ACCUMULATOR_MAX_FEEDS_PER_TURN = 3          # cap simultaneous feeds
# B1 (one-brain pre-pass): reserve the accumulator-lead from handle_expand
# BEFORE expand runs so the lead doesn't get nibbled into 30-ship pickups.
# Mirrors handle_accumulator's lead picker so all three handlers (expand,
# accumulator, mega-hammer) agree on which planet is "the lead". Defense is
# unaffected (it doesn't gate on mode_log) and rightly preempts the reservation.
BRAIN_LEAD_RESERVE_ENABLED = True
BRAIN_LEAD_RESERVE_4P_ONLY = True            # match the chain's 4P-only scope initially
# Decoupled from ACCUMULATOR_LEAD_MIN_SHIPS=100. The accumulator picks any
# 100+-ship planet as a lead, but reserving at that threshold starves expand
# from valuable solo captures (B1 iter 1 = -19pp 4P). Only reserve planets
# already close to mega-hammer firing — tied to the lowest prod-tier threshold
# (200 for prod-5 fresh) so reservation costs us nothing expand wouldn't have
# done anyway by that ship count.
BRAIN_LEAD_RESERVE_MIN_SHIPS = 200
# B1 iter 3: tested viable-target gate (only reserve if mega-hammer has a real
# target this turn). Result: no 4P gain over iter 2 (~58% both); 2P noise drift.
# Disabled — keep the code path for potential future use, but the gate at
# min_ships=200 is sufficient on its own.
BRAIN_LEAD_RESERVE_REQUIRE_TARGET = False
# B3b tested: lead picker scoring by avail - frontier_dist*weight (weight=2.0).
# Result: 4P -1.0pp (neutral), 2P -5.2pp (noise). Reverted — max-avail with
# threat-ratio cap is sufficient.
BRAIN_LEAD_PREFER_FRONTIER = False
BRAIN_LEAD_FRONTIER_WEIGHT = 2.0
# Original MEGA_HAMMER_SHIPS_MIN=300 is now a fallback for unknown prod.
# Also raise target_garrison ceiling so we can hit slightly tougher prey.
MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H = 100  # V14.1n: reverted CMA-ES 106 -> 100

# Mode 4b - Multi-prong forcing (V12.3c1, 2P only).
# When a hammer plan is active against target T and an enemy reinforcer E is
# pumping ships into T, open a second prong at E using surplus ships. Strict
# credibility gates keep this from splitting offense into two underweight prongs.
MULTIPRONG_ENABLED = False  # V14.2 (Phase 3.9): user-observed split-fleet flaw + wasted probes at high-cap neutrals. Multiprong was firing small secondary fleets at feint targets, causing both. Disabling to test.
MULTIPRONG_2P_ONLY = True
# Reinforcer must have at least this much in flight toward T (relative to T's
# arrival deficit) to count as a real reinforcer (vs a probe).
MULTIPRONG_REINFORCER_MIN_RATIO = 1.0
# Second prong must land with > E_home * this to satisfy credibility (E really
# takeable post-launch).
MULTIPRONG_E_OVERKILL = 1.05
# We must be prong-credible: committed_T + planned_E >= needed(T) + needed(E) * this.
MULTIPRONG_CREDIBILITY_FACTOR = 0.6
MULTIPRONG_MAX_TRAVEL = 40       # V12.8dg: was 35
MULTIPRONG_MIN_PER_CONTRIBUTOR = 8
MULTIPRONG_MAX_PARTICIPANTS = 3

# Late-game flush (only when patient farming has saturated and time is short)
LATE_FLUSH_REMAINING_TURNS = 25  # V12.8ee: was 20 — between 20 and 30
LATE_FLUSH_OVERKILL_RATIO = 1.05      # tolerate thinner margins under time pressure

# Per-turn budget guard
SOFT_DEADLINE_FRACTION = 0.82

# Race-to-neutral (V12.1a)
RACE_ENABLED = True
RACE_HORIZON_TURNS = 18          # V12.8at: was 25 — match anti-snipe horizon, fewer race losses ignored
RACE_MAX_NEUTRAL_DIST = 20     # V12.8dh: was 50 — tighter race window
RACE_TIE_GOES_TO_LARGER = True   # we still race when arrivals tie, since combat resolves by ship count

# Adaptive personality (V12.1b)
PERSONALITY_ENABLED = True
PERSONALITY_AGG_HIGH = 0.30      # enemy_ships_in_flight / total_enemy_ships above this => PRESSURE
PERSONALITY_AGG_LOW = 0.10       # below this => OPPORTUNISTIC
PERSONALITY_MIN_SAMPLE = 50      # below this many enemy ships, signal too weak — stay PATIENT

MODE_PARAMS = {
    "patient": {
        "expand_k_opening": 2,            # V12.8h: was 2 — partial widening (4 collapsed to parity)
        "expand_max_travel_opening": 22,  # V12.8h: was 20
        "expand_k_mid": 1,
        "expand_max_travel_mid": 14,
        "hammer_prod_share": 0.2,
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,       # V12.3b 2.3: explicit (was global)
    },
    "opportunistic": {
        "expand_k_opening": 3,            # V12.8h: was 2
        "expand_max_travel_opening": 22,  # V12.8h: was 20
        "expand_k_mid": 2,                # examine 2 nearest mid-game (vs 1)
        "expand_max_travel_mid": 18,      # +4 reach
        "hammer_prod_share": 0.35,        # slightly more eager to hammer
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,
    },
    "pressure": {
        "expand_k_opening": 3,            # V12.8h: was 2
        "expand_max_travel_opening": 22,  # V12.8h: was 20
        "expand_k_mid": 0,
        "expand_max_travel_mid": 9,      # slight reach increase to grab contested
        "hammer_prod_share": 0.30,        # much more eager to hammer
        "hammer_overkill": 1.20,          # thinner overkill (strike before reinforced)
        "hammer_stockpile_min": 50,
    },
}

# V12.2 R2: 2P-only overrides (heads-up duel rewards broader search and more
# eager offense than 4P FFA, where third parties absorb pressure). Active only
# when the game starts with exactly 2 players.
# V12.3b (2.2b): widen opening K + travel in 2P. Step 16 image showed bot
# unable to see cheap nearby neutrals because opening capped at K=2,
# travel=20 — easy targets sat outside the candidate window. In a 1v1 the
# downside of overextension is null (no third party to exploit), so wider
# opening expansion just costs us tempo it should reclaim by capturing the
# extra targets.
MODE_PARAMS_2P = {
    "patient": {
        "expand_k_opening": 5,            # V12.3b 2.2b: was 2 — see image 2 fix
        "expand_max_travel_opening": 35,  # V12.3b 2.2b: was 20 — engage cross-map
        "expand_k_mid": 4,                # was 1 — duels need broader target search
        "expand_max_travel_mid": 28,      # was 14 — engage past midline of 100x100 map
        "hammer_prod_share": 0.30,        # was 0.40 — symmetric duels rarely hit 40%
        "hammer_overkill": 1.15,          # was 1.30 — only one sniper to defend against
        "hammer_stockpile_min": 25,       # V12.3b 2.3: was 50 — duel planets churn ships
    },
    "opportunistic": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 35,
        "expand_k_mid": 6,
        "expand_max_travel_mid": 30,
        "hammer_prod_share": 0.28,
        "hammer_overkill": 1.15,
        "hammer_stockpile_min": 25,
    },
    "pressure": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 35,
        "expand_k_mid": 2,
        "expand_max_travel_mid": 52,      # V12.5d: was 30 — extend cross-map mid-game reach
        "hammer_prod_share": 0.25,        # V12.8co'': was 0.30
        "hammer_overkill": 1.177645,
        "hammer_stockpile_min": 25,
    },
}

# V12.2 R2: forced-pressure timeout breaks PATIENT-vs-PATIENT deadlocks in 2P.
# After this many turns of intended-PATIENT with no production-share gain,
# escalate to OPPORTUNISTIC; double the threshold to escalate to PRESSURE.
TWO_P_PATIENT_NUDGE_TURNS = 10
TWO_P_PATIENT_ESCALATE_TURNS = 20
TWO_P_PROD_SHARE_HISTORY = 10
TWO_P_PROD_SHARE_PROGRESS_EPS = 0.005   # 0.5pp gain over the window resets streak

# V14.1a (Phase 3.1, variant V2): 2P stop-expanding gate based on prod-share.
# When we have a stable prod-share lead in 2P, drop neutrals from expand
# candidates so sources route to enemy targets via existing hammer paths.
# Brainstorm in RL/notes/phase3_stop_expand.md (V1-V8).
STOP_EXPAND_2P_ENABLED = True
# V14.1a iter c — raised to 0.65. Instrumentation (RL/notes/instrument_stop_expand)
# showed at 0.50 the gate fired in 100% of games for 119+ turns (too eager).
# At 0.65 it should fire only when we have a CLEAR lead (parity with V14.0's
# existing `is_late` flag at >0.55+remaining<80, but without the remaining gate).
STOP_EXPAND_PROD_SHARE_2P = 0.65    # V14.1n: reverted CMA-ES 0.50 -> 0.65 (n=400 2P regress)
STOP_EXPAND_TURN_MIN_2P = 30        # V14.1n: reverted CMA-ES 23 -> 30

# V14.1f (Phase 3.5, Idea 4): smart stop-expand experiments.
# iter 8a-v1 (combat-contact, both formats, eager): REGRESSED 2P -22, 4P -3pp
# iter 8a-v2 (combat-contact, 4P-only, raised thresholds): inconclusive at n=192
#   (2P swing 47 wins between v1/v2 with identical 2P code = harness noise dominates)
# iter 8a DISABLED. Pivoting to iter 8b (prod-lag) — more principled rationale.
COMBAT_STOP_EXPAND_ENABLED = False      # iter 8a parked (noise-dominated, no clear win)
COMBAT_STOP_EXPAND_4P_ONLY = True
COMBAT_STOP_EXPAND_TURN_MIN = 25
COMBAT_CONTACT_MIN_SHIPS = 15
COMBAT_CHEAP_GARRISON = 10              # also used by iter 8b cheap-pick exception
COMBAT_CHEAP_DIST = 12.0

# V14.1f iter 8b: prod-lag stop-expand. [PROMOTED]
# When my_prod_share is below threshold, expansion to neutrals can't catch
# up to the leader's lead — only enemy captures can flip the gap. Drop
# non-cheap neutrals while behind. Both 2P and 4P. Cheap-pick exception
# preserves free pickups.
PROD_LAG_STOP_EXPAND_ENABLED = True
PROD_LAG_STOP_EXPAND_TURN_MIN = 25
PROD_LAG_STOP_EXPAND_THRESH_2P = 0.40   # 2P parity = 0.50; lag = below 0.40
PROD_LAG_STOP_EXPAND_THRESH_4P = 0.22   # 4P avg = 0.25; lag = below 0.22

# V14.1g iter 8c: enemy-tempo stop-expand. [PROMOTED — V14.1g]
# When 2+ enemy planets just launched (per _enemy_recently_launched), enemy
# is committing offense. A/B'd 2P +1 wins (52.1%) and 4P 60.4% 1st rate —
# both >50%, CAND beat V14.1f BASE. Stacks on V14.1f's prod-lag.
ENEMY_TEMPO_STOP_EXPAND_ENABLED = True
ENEMY_TEMPO_STOP_EXPAND_TURN_MIN = 20
ENEMY_TEMPO_STOP_EXPAND_MIN_LAUNCHES = 2

# V14.1g iter 8d: easy-enemy stop-expand. [PARKED — regressed]
# When N+ enemy planets with low garrison exist within reach of our planets,
# attacking them dominates neutral expansion. A/B'd 2P -15 wins (47.9%) and
# 4P 49.5% — CAND lost to V14.1g BASE. Gate fires too eagerly (low-garrison
# enemy targets exist from early-game). Disabled; code retained.
EASY_ENEMY_STOP_EXPAND_ENABLED = False
EASY_ENEMY_STOP_EXPAND_TURN_MIN = 15
EASY_ENEMY_MAX_GARRISON = 20
EASY_ENEMY_MAX_DIST = 25.0
EASY_ENEMY_MIN_COUNT = 1

# V14.1j iter 8i: turn-cutoff stop-expand.
# Past turn N (before TERMINAL_PHASE), drop all neutral candidates. Late-
# game neutral expansion almost never pays back in remaining turns, even
# if cheap. Both formats.
TURN_CUTOFF_STOP_EXPAND_ENABLED = True
TURN_CUTOFF_STOP_EXPAND_TURN = 80   # TERMINAL_PHASE fires at remaining<30, so ~90+

# V14.1i iter 8h: prod-lead stop-expand for 4P.
# 2P already has stop_expanding_2p (prod_share >= 0.50). 4P had no
# equivalent — when we're ahead in 4P, neutral expansion is wasted; the
# game is won by attacking the leader / converting our prod-share lead
# into captures. Strict (no cheap-pick exception) like stop_expanding_2p,
# 4P-only.
PROD_LEAD_STOP_EXPAND_4P_ENABLED = True
PROD_LEAD_STOP_EXPAND_4P_TURN_MIN = 25
PROD_LEAD_STOP_EXPAND_4P_THRESH = 0.35   # 4P avg = 0.25; lead = 0.35+

# V14.1h iter 8g: stockpile-readiness stop-expand.
# When any of our planets has accumulated huge garrison (idle ships sitting
# unused), neutral expansion just adds more production we don't need —
# better to convert the stockpile into enemy captures. Fires often in mid/
# late game on prod-3/4/5 planets that haven't launched recently.
STOCKPILE_STOP_EXPAND_ENABLED = True
STOCKPILE_STOP_EXPAND_TURN_MIN = 20
STOCKPILE_STOP_EXPAND_MAX_GARRISON = 250  # any my-planet ships >= this triggers

# V14.1h iter 8e: neutral-saturation stop-expand.
# When NO cheap neutrals (garrison <= 10) exist within reach of any of our
# planets, neutral expansion has nothing left to offer — sources should
# look at enemy targets.
# iter 8e-v1 (both formats): 2P +20 wins (56.2%) WIN, 4P 37.0% REGRESS.
# Per user's mixed-result rule: format-gate to 2P-only.
NEUTRAL_SATURATION_STOP_EXPAND_ENABLED = False  # parked — noise-dominated
NEUTRAL_SATURATION_2P_ONLY = True
NEUTRAL_SATURATION_TURN_MIN = 20
NEUTRAL_SATURATION_CHEAP_GARRISON = 10
NEUTRAL_SATURATION_REACH_DIST = 30.0


# ============================================================
# Types
# ============================================================

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


# ============================================================
# Physics
# ============================================================

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def orbital_radius(p):
    return dist(p.x, p.y, CENTER_X, CENTER_Y)


def is_static_planet(p):
    return orbital_radius(p) + p.radius >= ROTATION_LIMIT


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_sq))
    return dist(px, py, x1 + t * dx, y1 + t * dy)


def segment_hits_sun(x1, y1, x2, y2):
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + SUN_SAFETY


def launch_point(sx, sy, sr, angle):
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def safe_geometry(sx, sy, sr, tx, ty, tr):
    """Direct-line angle + clear travel distance, or None if the path crosses the sun."""
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_d = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    ex = lx + math.cos(angle) * hit_d
    ey = ly + math.sin(angle) * hit_d
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_d


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_geometry(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def predict_planet_position(planet, initial_by_id, ang_vel, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new = cur + ang_vel * turns
    return CENTER_X + r * math.cos(new), CENTER_Y + r * math.sin(new)


# V13.3 R4: behind-sun wait. When aim_at_target's first estimate fails (sun
# blocks current direct path), try future projected positions of the target.
# We can still launch NOW aiming at where the target WILL be — fleet flies
# straight, target swings into intercept. Better than rejecting the shot.
R4_BEHIND_SUN_WAIT_ENABLED = True
R4_FUTURE_HORIZON = 10   # max turns ahead to project for clear-path search


# V13.3 Q1: comet future-position prediction. Comets follow a precomputed path
# (obs.comets[group].paths[idx][path_index..end]). Aim_at_target previously
# treated comets as static (predict_planet_position falls back to current x,y
# for non-orbital planets), so fleets aimed at a comet missed by 5-10 turns'
# travel. Port from lb1224 (Roman Tamrazov) predict_comet_position.
def predict_comet_position(planet_id, comets, turns):
    for group in comets:
        pids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", []) if isinstance(group, dict) else []
        path_index = group.get("path_index", 0) if isinstance(group, dict) else 0
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = int(path_index) + int(turns)
        if 0 <= future_idx < len(path):
            return float(path[future_idx][0]), float(path[future_idx][1])
        return None
    return None


def predict_target_position(target, world, turns):
    """Dispatch: comets use their precomputed path; orbital planets use angular
    extrapolation; static planets stay put. Returns (x, y) or None if a comet
    has expired by `turns`."""
    if target.id in world.comet_ids:
        pos = predict_comet_position(target.id, world.comets, turns)
        if pos is not None:
            return pos
        # Fall through for safety: comet expired but caller didn't check
    return predict_planet_position(target, world.initial_by_id, world.ang_vel, turns)


AIM_MAX_ITERS = 6          # was 5 — orbital targets sometimes need more
AIM_CONVERGE_TURNS = 2
AIM_CONVERGE_DIST = 0.6


def aim_at_target(src, target, ships, initial_by_id, ang_vel, world=None):
    """Returns (angle, turns) for sending `ships` from src to hit target.
    Iterates orbital prediction. Returns None if the path is blocked by the
    sun OR if convergence isn't reached — better to skip a target than fire
    a fleet that wanders past it because our aim didn't settle.

    V13.3 Q1: when target is a comet AND world is passed, use comet path for
    future-position; otherwise existing orbital extrapolation.

    V13.3 R4 (behind-sun wait): if the FIRST estimate fails (current path
    blocked by sun), try aiming at projected future positions of the target
    where the orbital motion may have cleared the path. We launch NOW aiming
    at where the target WILL be — fleet flies straight, target swings into
    place. Better than rejecting the shot entirely."""
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None and R4_BEHIND_SUN_WAIT_ENABLED and world is not None:
        # V13.3 R4: try aiming at future positions of the target.
        for future_t in range(2, R4_FUTURE_HORIZON, 2):
            if target.id in world.comet_ids:
                pos = predict_comet_position(target.id, world.comets, future_t)
            else:
                init = initial_by_id.get(target.id)
                if init is None:
                    pos = None
                elif dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
                    pos = None  # static; no future to project
                else:
                    pos = predict_planet_position(target, initial_by_id, ang_vel, future_t)
            if pos is None:
                continue
            est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
            if est is not None:
                break
    if est is None:
        return None
    # V13.3 Q1: comet branch — iterate using precomputed path positions.
    is_comet = world is not None and target.id in world.comet_ids
    if not is_comet:
        init = initial_by_id.get(target.id)
        if init is None:
            return est
        if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
            return est

    angle, turns = est
    tx, ty = target.x, target.y
    for _ in range(AIM_MAX_ITERS):
        if is_comet:
            pos = predict_comet_position(target.id, world.comets, turns)
            if pos is None:
                # Comet expires before our arrival — refuse the shot
                return None
            ntx, nty = pos
        else:
            ntx, nty = predict_planet_position(target, initial_by_id, ang_vel, turns)
        nest = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if nest is None:
            return None
        nangle, nturns = nest
        if (abs(ntx - tx) < AIM_CONVERGE_DIST
                and abs(nty - ty) < AIM_CONVERGE_DIST
                and abs(nturns - turns) <= AIM_CONVERGE_TURNS):
            return nangle, nturns
        angle, turns = nangle, nturns
        tx, ty = ntx, nty
    # Did not converge — refuse the shot rather than fire a wandering fleet.
    return None


def fleet_target_planet(fleet, planets, initial_by_id=None, ang_vel=0.0):
    """Which planet this in-flight fleet hits, and when (in turns from now).

    Two-pass: static planets via cheap straight-line intersection, orbital
    planets via per-turn forward simulation. The naive straight-line check
    against the planet's CURRENT position misses orbital targets — the
    planet has rotated since the fleet launched, so the ray won't intersect
    its current XY but WILL intersect its future orbital position. Without
    accounting for this, incoming hostile fleets at our orbital planets
    don't show up in arrivals_by_planet, and the reservation walk wrongly
    decides our planet is safe and lets it fire offensively.
    """
    dx_dir = math.cos(fleet.angle)
    dy_dir = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    def _is_orbital(p):
        if initial_by_id is None:
            return False
        init = initial_by_id.get(p.id)
        if init is None:
            return False
        return dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius < ROTATION_LIMIT

    best_p, best_t = None, float(SIM_HORIZON) + 1.0

    # Pass 1 — static planets: straight-line intersection. Also include orbital
    # planets here as a baseline (will be overridden if a better orbital match
    # exists in pass 2).
    for p in planets:
        if _is_orbital(p):
            continue
        dx = p.x - fleet.x
        dy = p.y - fleet.y
        proj = dx * dx_dir + dy * dy_dir
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        rr = p.radius * p.radius
        if perp_sq >= rr:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, rr - perp_sq)))
        t = hit_d / speed
        if t <= SIM_HORIZON and t < best_t:
            best_t, best_p = t, p

    # Pass 2 — orbital planets: walk forward turn-by-turn and test true positions.
    # V12.8ez: tiebreak co-arrival turns by closest center-distance instead of
    # iteration order.
    if initial_by_id is not None:
        best_dsq = None
        max_t = int(math.ceil(min(best_t, float(SIM_HORIZON))))
        for t in range(1, max_t + 1):
            fx = fleet.x + dx_dir * speed * t
            fy = fleet.y + dy_dir * speed * t
            for p in planets:
                if not _is_orbital(p):
                    continue
                px, py = predict_planet_position(p, initial_by_id, ang_vel, t)
                rr = p.radius * p.radius
                dsq = (fx - px) ** 2 + (fy - py) ** 2
                if dsq < rr:
                    if t < best_t or (t == best_t and (best_dsq is None or dsq < best_dsq)):
                        best_t, best_p, best_dsq = float(t), p, dsq
            if best_p is not None and best_t <= t:
                break

    if best_p is None:
        return None, None
    return best_p, max(1, int(math.ceil(best_t)))


# ============================================================
# Capture math
# ============================================================

def garrison_at_arrival(target, travel_turns):
    """Defender ship count at the moment our fleet lands."""
    if target.owner == -1:
        return int(target.ships)  # neutrals don't grow
    return int(target.ships) + int(target.production) * int(travel_turns)


def needed_to_capture(target, travel_turns):
    """Ships required at arrival to flip ownership (combat: survivor > garrison)."""
    return garrison_at_arrival(target, travel_turns) + 1


# V13.3 N1 (effective garrison): walks incoming hostile fleets that arrive
# BEFORE our travel_turns, resolves combat, and returns the projected defender
# count we'll face. Catches the case where an enemy fleet hits a neutral
# before us, flipping it to enemy and growing the defense by enemy production.
# Without this, needed_to_capture(neutral with 10 ships) returns 11 — we send
# 12, arrive after enemy took the planet and now it has 40 enemy ships,
# we lose all 12 for nothing.
EFFECTIVE_GARRISON_ENABLED = True
# K7-A toggle: engine-faithful simultaneous-arrival combat resolution.
# Mathematically correct but regressed when paired with CMA-tuned heuristics
# that depended on the previous projection. Off pending heuristic re-tune.
K7A_ENGINE_FAITHFUL_SIM_COMBAT = False

# K8 (soft-target detection in expand ranking, 2026-05-20):
# Existing expand ranking scores enemy/neutral targets by distance-VALUE
# (eff_dist - prod*VALUE_WEIGHT) plus several small bonuses (F1B, K2, SO1).
# It does NOT consult `effective_garrison_at_arrival`, which already
# computes the PROJECTED defender count after pre-arrival fleets resolve.
# Result: an enemy planet whose garrison will be wiped to 0 by an inbound
# enemy-vs-enemy fight (4P) or by our own arriving fleet (2P) sits at the
# same rank as a fully-defended planet of the same prod & distance.
#
# K8 fixes this asymmetry: compute the SHIP-COST RATIO at arrival
# (projected_garrison / current_garrison). When that ratio is well below
# 1.0, the target is "softening" and we should prefer it over equidistant
# alternatives. main.py does not consult this projection in expand
# ranking — so K8 gives main_k a genuine information edge over main.py.
#
# Bonus magnitude is bounded so a strong target far away can't beat a
# moderate target nearby — the bonus only breaks ties between targets of
# similar distance-value. Cap is half the smallest VALUE_WEIGHT
# (max ~2.0) so a 5-ship-prod planet at the same distance always wins
# over a 1-ship-prod soft target unless the soft target is REALLY soft.
K8_SOFT_TARGET_ENABLED = True
K8_SOFT_TARGET_THRESHOLD = 0.6   # ratio at which softening starts
K8_SOFT_TARGET_MAX_BONUS = 8.0   # bigger than F1B (3.0), smaller than LEADER_BASH
K8_SOFT_TARGET_4P_BONUS_SCALE = 0.5   # 4P: scale bonus down — kingmaker risk

def effective_garrison_at_arrival(target, travel_turns, world):
    """Defender count at our arrival, accounting for pre-arrival fleets.
    Returns (projected_owner, projected_ships) at travel_turns.

    K7-A (2026-05-20 fix): engine resolves simultaneous arrivals ATOMICALLY:
    sum ships per owner, top-vs-second contest → survivor (top - second),
    then survivor vs planet garrison. The prior walk processed fleets one
    at a time, so a planet hit by 3 fleets on the same turn from 3 owners
    produced a wrong owner/ships projection (often flipped ownership when
    the engine would keep the defender). This produced systematic
    over-sending and wrong target selection in late-game multi-front fights.

    We now batch arrivals by eta and resolve each batch via the engine's
    combat rules. PRESERVE the 4P-only filter on our own fleets: in 4P,
    a hostile counter-snipe between our first fleet's flip and our follow-up
    is unmodelled future-enemy-emission risk, so we plan conservatively as
    if our first fleet didn't pre-soften the target. In 2P there's only one
    enemy to project, so including our own fleets is safe.
    """
    if not EFFECTIVE_GARRISON_ENABLED:
        return target.owner, garrison_at_arrival(target, travel_turns)
    arrivals = world.arrivals_by_planet.get(target.id, [])
    # 4P keeps the historical "exclude our own fleets" filter to model
    # unmodelled counter-snipe risk; 2P includes our fleets.
    if world.is_2p:
        relevant = sorted(
            ((eta, owner, ships) for eta, owner, ships in arrivals
             if 1 <= eta <= travel_turns and ships > 0 and owner != -1),
            key=lambda x: x[0],
        )
    else:
        relevant = sorted(
            ((eta, owner, ships) for eta, owner, ships in arrivals
             if 1 <= eta <= travel_turns and owner != world.player and ships > 0
             and owner != -1),
            key=lambda x: x[0],
        )
    if not relevant:
        return target.owner, garrison_at_arrival(target, travel_turns)
    owner = int(target.owner)
    ships = int(target.ships)
    prod = max(0, int(target.production))
    last_t = 0

    if not K7A_ENGINE_FAITHFUL_SIM_COMBAT:
        # ORIGINAL walk: sequential, one fleet at a time. Mathematically wrong
        # for simultaneous arrivals but matches the projection downstream
        # heuristics were tuned against.
        for eta, fleet_owner, fleet_ships in relevant:
            if owner != -1:
                ships += prod * (eta - last_t)
            if fleet_owner == owner:
                ships += fleet_ships
            else:
                if fleet_ships > ships:
                    owner = int(fleet_owner)
                    ships = fleet_ships - ships
                elif fleet_ships < ships:
                    ships -= fleet_ships
                else:
                    ships = 0
            last_t = eta
        if owner != -1:
            ships += prod * (travel_turns - last_t)
        return owner, ships

    # K7-A: engine-faithful, batch by eta and resolve atomically.
    by_eta = defaultdict(list)
    for eta, ow, sh in relevant:
        by_eta[eta].append((int(ow), int(sh)))
    for eta in sorted(by_eta.keys()):
        if owner != -1:
            ships += prod * (eta - last_t)
        player_ships = defaultdict(int)
        for fo, fs in by_eta[eta]:
            player_ships[fo] += fs
        sorted_players = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
        top_owner, top_ships = sorted_players[0]
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            if top_ships == second_ships:
                survivor_ships = 0
            survivor_owner = top_owner if survivor_ships > 0 else -1
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships
        if survivor_ships > 0:
            if owner == survivor_owner:
                ships += survivor_ships
            else:
                ships -= survivor_ships
                if ships < 0:
                    owner = int(survivor_owner)
                    ships = -ships
        last_t = eta
    if owner != -1:
        ships += prod * (travel_turns - last_t)
    return owner, ships


def effective_needed_to_capture(target, travel_turns, world):
    """needed_to_capture with effective_garrison_at_arrival projection."""
    _, defender_ships = effective_garrison_at_arrival(target, travel_turns, world)
    return defender_ships + 1


# ============================================================
# Reservation walk — load-bearing primitive
# ============================================================

def collect_arrivals(planet_id, fleets, planets, initial_by_id=None, ang_vel=0.0):
    """For a given planet, return [(eta, owner, ships)] of all fleets converging on it."""
    out = []
    for f in fleets:
        if int(f.ships) <= 0:
            continue
        target, eta = fleet_target_planet(f, planets, initial_by_id, ang_vel)
        if target is None or target.id != planet_id:
            continue
        out.append((eta, int(f.owner), int(f.ships)))
    return out


def compute_planet_reserve(planet, arrivals, player):
    """The minimum ships we must keep on the surface so the running balance never
    dips below ABSORB_PROJECTION_MARGIN through every incoming fleet's arrival,
    factoring production growth and friendly reinforcements.

    Returns (reserve, holds, deficit, deadline).
        reserve   int, ships that must NOT be sent out this turn.
        holds     True if reserve <= planet.ships (planet survives on its own).
        deficit   ships we still need from outside if !holds (else 0).
        deadline  earliest turn balance dips below margin if !holds (else None).

    V12.3c4 (2.4 redesign): per-fleet ABSORB_MIN_THREAT filter replaced
    with window-aggregated check. Window = garrison/production (the
    planet's natural absorb cycle). If sum(hostile_in_window) < threshold,
    ignore all hostile fleets within the window. Hostile fleets outside
    the window are always counted (they're far out enough that natural
    growth doesn't cover them and they aren't simple noise). Closes the
    Stackelberg-leader exploit (firing many sub-threshold fleets) without
    triggering absorb on transient noise the planet would have absorbed.
    """
    if planet.owner != player:
        return 0, True, 0, None

    prod = max(0, int(planet.production))
    ships_now = max(0, int(planet.ships))
    if prod > 0:
        absorb_window = max(1, ships_now // prod)
    else:
        absorb_window = SIM_HORIZON

    hostile_in_window = 0
    for eta, owner, ships in arrivals:
        if ships <= 0 or owner == player or owner == -1:
            continue
        if int(eta) <= absorb_window:
            hostile_in_window += int(ships)
    # V12.6f (issue #4): scale the noise-floor with current garrison. A planet
    # at 0-2 ships dies to a single probe — flat threshold=3 silently dropped
    # those threats. For ships < 9 we shrink the threshold linearly (min 1);
    # for larger planets the original ABSORB_MIN_THREAT still applies so the
    # Stackelberg-drip filter is unchanged.
    absorb_min_threat = max(1, min(ABSORB_MIN_THREAT, ships_now // 3))
    skip_in_window_hostiles = hostile_in_window < absorb_min_threat

    # V12.8ew: engine combat is "top-minus-second" per planet-turn — when two
    # enemies attack the same planet on the same turn, they fight EACH OTHER
    # too, so the defender effectively faces only the LARGEST single-owner
    # hostile (not the sum). Bucket per turn per owner; the defender's net loss
    # at that turn is max(hostile_owner_totals).
    friendly_events = defaultdict(int)
    hostile_by_owner = defaultdict(lambda: defaultdict(int))
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        if owner == player:
            friendly_events[eta] += ships
        elif owner == -1:
            continue
        else:
            if skip_in_window_hostiles and int(eta) <= absorb_window:
                continue
            hostile_by_owner[eta][owner] += int(ships)

    events = defaultdict(int)
    for eta, ships in friendly_events.items():
        events[eta] += ships
    for eta, owner_totals in hostile_by_owner.items():
        # Engine: top-minus-second of per-owner hostile totals reaches defender.
        # Tie at top → all attackers destroyed (survivor=0).
        sorted_h = sorted(owner_totals.values(), reverse=True)
        if len(sorted_h) == 1:
            survivor = sorted_h[0]
        elif sorted_h[0] == sorted_h[1]:
            survivor = 0
        else:
            survivor = sorted_h[0] - sorted_h[1]
        events[eta] -= survivor

    if not events:
        return 0, True, 0, None

    growth = int(planet.production)
    bal = int(planet.ships)
    last_t = 0
    min_bal = bal
    deadline = None

    for turn in sorted(events):
        bal += growth * (turn - last_t)
        bal += events[turn]
        if bal < min_bal:
            min_bal = bal
        if bal < ABSORB_PROJECTION_MARGIN and deadline is None:
            deadline = turn
        last_t = turn

    if min_bal >= ABSORB_PROJECTION_MARGIN:
        excess = min_bal - ABSORB_PROJECTION_MARGIN
        reserve = max(0, int(planet.ships) - excess)
        return reserve, True, 0, None

    deficit = ABSORB_PROJECTION_MARGIN - min_bal
    return int(planet.ships), False, int(deficit), deadline


# ============================================================
# V12.9 Forward simulator (minimal Melis-style futures projection).
# ============================================================

def forward_project(world, our_capture_target=None, our_capture_turn=None,
                    our_capture_ships=None, horizon=20,
                    project_opponent_moves=False,
                    opponent_emit_fraction=0.4,
                    snapshot_turns=None):
    """Project every planet's owner+ship count forward `horizon` turns.

    Inputs:
      world — current World snapshot.
      our_capture_target/turn/ships — optional our planned capture (treated
        as a hypothetical friendly fleet arrival).
      horizon — how many turns to project.
      project_opponent_moves — if True, each enemy planet launches a fraction
        of its CURRENT surplus toward its closest non-friendly target every
        few turns. Increases accuracy at cost of pessimism for our holdings.
      opponent_emit_fraction — fraction of surplus the projected launch sends.
    Returns:
      dict planet_id -> (owner_at_H, ships_at_H).

    Model:
      - Existing in-flight fleets arrive at their projected ETA (engine
        combat math: attackers fight each other top-minus-second, then
        survivor reinforces or attacks defender garrison).
      - Production accumulates each turn for owned planets.
      - Phantom launches: each enemy planet within max-speed reach of
        our_capture_target projects a fleet of size phantom_factor*ships
        with optimistic ETA. This catches the dominant snipe risk that
        existing arrivals_by_planet misses (the enemy hasn't launched yet
        but COULD before our planet stabilises).
    """
    # Per-planet arrival ledger from in-flight fleets.
    by_pid = defaultdict(list)
    for pid, arrs in world.arrivals_by_planet.items():
        for eta, owner, ships in arrs:
            if 0 < eta <= horizon:
                by_pid[pid].append((int(eta), int(owner), int(ships)))

    # (Reactive counter-snipe projection tested — net 0 vs h12 alone, dropped.)

    # Add our hypothetical capture.
    if our_capture_target is not None and our_capture_turn is not None:
        by_pid[our_capture_target].append(
            (int(our_capture_turn), int(world.player), int(our_capture_ships))
        )

    # State table.
    state = {}
    for p in world.planets:
        state[p.id] = [int(p.owner), int(p.ships), int(p.production)]

    # Pre-compute planet positions and per-pair distance for opponent move
    # projection. Engine planets rotate but for short horizons we use static
    # positions — accuracy cost is small relative to "static enemies" bug.
    planet_pos_map = {p.id: (float(p.x), float(p.y)) for p in world.planets}
    pid_list = list(state.keys())

    # Per-planet production rates (used by opponent move projection).
    prod_by_pid = {p.id: max(0, int(p.production)) for p in world.planets}

    snapshots = {} if snapshot_turns else None
    snapshot_set = set(snapshot_turns) if snapshot_turns else None
    for t in range(1, horizon + 1):
        # 1. Production for owned planets (engine: production happens first).
        for pid, st in state.items():
            if st[0] != -1:
                st[1] += st[2]
        # 2. Project per-player launches (us + opponents): every 4 turns
        # each owned planet with surplus > 10 launches a fraction of its
        # ships at its CURRENT closest non-friendly target.
        if project_opponent_moves and t % 4 == 0:
            for pid, st in state.items():
                if st[0] == -1 or st[1] < 10:
                    continue
                src_x, src_y = planet_pos_map[pid]
                src_owner = st[0]
                best_d = float("inf")
                best_op = None
                for opid, ost in state.items():
                    if opid == pid or ost[0] == src_owner:
                        continue
                    ox, oy = planet_pos_map[opid]
                    d = ((src_x - ox) ** 2 + (src_y - oy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best_op = d, opid
                if best_op is None:
                    continue
                # Smaller emit for "us" to reflect our defensive caution
                if src_owner == world.player:
                    frac = opponent_emit_fraction * 0.5
                else:
                    frac = opponent_emit_fraction
                emit = int(st[1] * frac)
                if emit < 5:
                    continue
                ratio = math.log(max(2, emit)) / math.log(1000.0)
                speed = 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)
                eta_arrive = max(1, int(math.ceil(best_d / speed)))
                arrival_t = t + eta_arrive
                if arrival_t > horizon:
                    continue
                by_pid[best_op].append((arrival_t, src_owner, emit))
                st[1] -= emit
        # 3. Arrivals at this turn (engine: combat resolution).
        # Engine math (orbit_wars.py:631-669):
        #   - Sum incoming fleets per owner (defender garrison NOT included).
        #   - Top-minus-second of attacker totals = "survivor".
        #   - Survivor (if >0) reinforces or attacks defender garrison.
        for pid, arrs in by_pid.items():
            this_turn = [(o, s) for et, o, s in arrs if et == t]
            if not this_turn:
                continue
            st = state[pid]
            defender_owner, garrison = st[0], st[1]
            from_owner = defaultdict(int)
            for o, s in this_turn:
                from_owner[o] += s
            sorted_owners = sorted(from_owner.items(), key=lambda x: -x[1])
            top_owner, top_ships = sorted_owners[0]
            if len(sorted_owners) >= 2:
                second_ships = sorted_owners[1][1]
                if top_ships == second_ships:
                    survivor_ships = 0
                    survivor_owner = -1
                else:
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner
            else:
                survivor_ships = top_ships
                survivor_owner = top_owner
            if survivor_ships > 0:
                if defender_owner == survivor_owner:
                    st[1] = garrison + survivor_ships
                else:
                    new_garrison = garrison - survivor_ships
                    if new_garrison < 0:
                        st[0] = survivor_owner
                        st[1] = -new_garrison
                    else:
                        st[1] = new_garrison
        if snapshot_set is not None and t in snapshot_set:
            snapshots[t] = {pid: (st[0], st[1]) for pid, st in state.items()}

    final = {pid: (st[0], st[1]) for pid, st in state.items()}
    if snapshot_turns is not None:
        return final, snapshots
    return final


def _depth2_penalty(world, our_action, top_opp_actions=2):
    """For our action, project worst-case opponent reply.
    Each enemy planet within reach of our_action's target tries to launch a
    counter-snipe. Returns the WORST (lowest from our POV) Melis score among
    those counter-snipe scenarios.

    Used to penalize our actions that invite easy counter-snipes.
    """
    target_id = our_action["target_id"]
    tgt = world.planet_by_id.get(target_id)
    if tgt is None:
        return 0.0
    worst_delta = 0.0
    candidates_evaluated = 0
    for ep in world.planets:
        if ep.owner == world.player or ep.owner == -1:
            continue
        if int(ep.ships) < 9:
            continue
        d = ((tgt.x - ep.x) ** 2 + (tgt.y - ep.y) ** 2) ** 0.5
        if d > 30.0:
            continue
        # Hypothetical opponent counter-snipe
        opp_ships = max(8, int(ep.ships) - 5)
        ratio = math.log(max(2, opp_ships)) / math.log(1000.0)
        speed = 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)
        opp_eta = max(1, int(math.ceil(d / speed)))
        if opp_eta > FWD_SIM_HORIZON + 4:
            continue
        # Project state with both actions injected
        proj = forward_project(
            world,
            our_capture_target=our_action["target_id"],
            our_capture_turn=our_action["arrival_turn"],
            our_capture_ships=our_action["ships"],
            horizon=FWD_SIM_HORIZON + 6,
            project_opponent_moves=True,
            opponent_emit_fraction=0.30,
        )
        # Inject opponent counter as additional arrival
        # (state already includes their projected emit; this models aggressive
        # immediate counter)
        end_owner, end_ships = proj.get(target_id, (-1, 0))
        # Penalty if their counter would still flip us
        if end_owner != world.player and opp_ships > end_ships:
            worst_delta = min(worst_delta, -opp_ships)
        candidates_evaluated += 1
        if candidates_evaluated >= top_opp_actions:
            break
    return worst_delta


def search_step_action(world, max_per_source=3, max_actions_to_eval=10,
                       use_depth2=False):
    """Depth-1 alpha-beta over step actions.

    1. Generate candidate step actions via generate_step_actions.
    2. Evaluate each via melis_evaluate (sim+score).
    3. Return list sorted by score (highest first), up to `max_actions_to_eval`.

    Each action has additional key "score". Caller picks top action(s) and
    commits via _commit_fleet.
    """
    actions = generate_step_actions(world, max_per_source=max_per_source)
    if not actions:
        return []
    baseline_score = melis_evaluate(world, our_step_action=None)
    # V12.10 time discount (2P-only): prefer earlier captures (compound
    # production). In 4P this preferred small fast neutrals over strategic
    # later captures (4P 1st-rate dropped 32%->22% at n=320), so we gate.
    apply_decay = world.is_2p
    scored = []
    for act in actions[:max_actions_to_eval]:
        act_score = melis_evaluate(world, our_step_action=act)
        gain = act_score - baseline_score
        if apply_decay and gain > 0:
            gain *= 0.97 ** int(act["arrival_turn"])
        act["score"] = gain
        scored.append(act)
    scored.sort(key=lambda a: (-a["score"], a.get("raw_dist", 0.0)))
    if use_depth2:
        # Apply depth-2 penalty to top-3 actions; re-rank
        for act in scored[:3]:
            act["score"] += _depth2_penalty(world, act)
        scored.sort(key=lambda a: (-a["score"], a.get("raw_dist", 0.0)))
    # V13.3 P2 (2P-only): if even the best action's gain is below
    # MELIS_SANITY_THETA, stand pat for this turn.
    if MELIS_SANITY_ENABLED and world.is_2p and scored and scored[0]["score"] < MELIS_SANITY_THETA:
        return []
    return scored


def generate_step_actions(world, max_per_source=3):
    """Generate candidate "step actions" — Melis style. Each step action is
    a single capture targeting one planet, sourced from one of our planets.

    Returns list of dicts: {"target_id", "source_id", "angle", "arrival_turn",
                            "ships", "raw_dist"}.

    Pruning:
      - Skip targets that aren't reachable within max_travel + 4
      - Skip neutral targets blocked by NEUTRAL_HARD_CAP
      - Take top `max_per_source` per source (closest by raw distance)
    """
    actions = []
    if not world.my_planets:
        return actions
    # Mode-dependent travel limits
    is_opening = world.is_opening
    if is_opening:
        max_travel = world.mode_params.get(
            "expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        max_travel = world.mode_params["expand_max_travel_mid"]

    for src in world.my_planets:
        avail = max(0, int(src.ships))
        if avail < MIN_DISPATCH_SHIPS:
            continue
        targets = []
        for t in world.planets:
            if t.owner == world.player:
                continue
            if not is_targetable(world, t):
                continue
            if _neutral_blocked_by_cap(world, t):
                continue
            raw = dist(src.x, src.y, t.x, t.y)
            if raw / MAX_SPEED > max_travel + 4:
                continue
            targets.append((raw, t))
        targets.sort(key=lambda x: x[0])
        # F16: assemble candidate target set as top-closest + top-production-extra.
        if F16_DIVERSITY_ENABLED:
            n_close = min(F16_CLOSEST_PICKS, max_per_source)
            picks = list(targets[:n_close])
            picked_ids = {p[1].id for p in picks}
            extras = [(raw, t) for raw, t in targets if t.id not in picked_ids]
            extras.sort(key=lambda x: (-int(x[1].production), x[0]))
            picks.extend(extras[:F16_PROD_PICKS])
        else:
            picks = targets[:max_per_source]
        for raw, t in picks:
            plan = plan_solo_capture(world, src, t, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            actions.append({
                "target_id": int(t.id),
                "source_id": int(src.id),
                "angle": float(angle),
                "arrival_turn": int(turns),
                "ships": int(ships),
                "raw_dist": float(raw),
            })
    return actions


def melis_evaluate(world, our_step_action=None, horizon=12, future_horizon=8,
                   opp_emit=0.20):
    """Melis full-attack-future evaluator.

    Inputs:
      world — current World snapshot.
      our_step_action — optional dict {"target_id", "arrival_turn", "ships"}.
        If provided, simulates our planned capture as part of the projection.
      horizon — short-term sim horizon for our action's effect.
      future_horizon — additional "all-attack-future" projection turns where
        every planet (us + opponents) keeps emitting surplus toward closest
        non-friendly. Captures position quality beyond the immediate move.
      opp_emit — fraction of surplus opponents launch in projection. 0.30
        is the calibrated default; lower = more capture-friendly.

    Returns: scalar score from our player's POV (higher = better).
    """
    target = arrival = ships = None
    if our_step_action is not None:
        target = our_step_action.get("target_id")
        arrival = our_step_action.get("arrival_turn")
        ships = our_step_action.get("ships")
    H = horizon + future_horizon
    n = 2 if world.is_2p else 4
    if FWD_SCORE_AGG_ENABLED:
        snap_turns = tuple(t for t in FWD_SCORE_AGG_TURNS if t <= H)
        if not snap_turns:
            snap_turns = (H,)
        final, snaps = forward_project(
            world,
            our_capture_target=target,
            our_capture_turn=arrival,
            our_capture_ships=ships,
            horizon=H,
            project_opponent_moves=True,
            opponent_emit_fraction=opp_emit,
            snapshot_turns=snap_turns,
        )
        total = 0.0
        count = 0
        for t in snap_turns:
            snap = snaps.get(t)
            if snap is None:
                continue
            total += forward_score(snap, world.player, n, world)
            count += 1
        if H not in snap_turns:
            total += forward_score(final, world.player, n, world)
            count += 1
        return total / max(1, count)
    state = forward_project(
        world,
        our_capture_target=target,
        our_capture_turn=arrival,
        our_capture_ships=ships,
        horizon=H,
        project_opponent_moves=True,
        opponent_emit_fraction=opp_emit,
    )
    return forward_score(state, world.player, n, world)


def forward_score(state, player, n_seats, world=None):
    """Score a forward-projected state from `player`'s POV.

    Combines: ship advantage + 5×planet-count advantage + 8×production advantage.
    Weights chosen so an extra owned planet is worth ~5 ships (a typical garrison)
    and an extra production unit is worth ~8 ships (≈2 turns of growth)."""
    n_planets = [0] * n_seats
    n_prod = [0] * n_seats
    n_ships = [0] * n_seats
    for pid, (o, s) in state.items():
        if 0 <= o < n_seats:
            n_ships[o] += s
            n_planets[o] += 1
            if world is not None:
                p = world.planet_by_id.get(pid)
                if p is not None:
                    n_prod[o] += int(p.production)
    if n_seats <= 1:
        return n_ships[player]
    others = [i for i in range(n_seats) if i != player]
    leader_ships = max(n_ships[i] for i in others)
    leader_planets = max(n_planets[i] for i in others)
    leader_prod = max(n_prod[i] for i in others)
    return ((n_ships[player] - leader_ships)
            + 5 * (n_planets[player] - leader_planets)
            + 8 * (n_prod[player] - leader_prod))


# ============================================================
# World snapshot
# ============================================================

class World:
    def __init__(self, obs, inferred_step=None):
        # V14.0: format-gated globals — declare at top of __init__ so the
        # reassignment further down (after is_2p is determined) is valid Python.
        global COALITION_MIN_PER_CONTRIBUTOR, DEFENSE_OVERSEND, PSM_OPENING_TURN, SO1_STATIC_BONUS
        self.player = _read(obs, "player", 0)
        obs_step = _read(obs, "step", 0) or 0
        self.step = max(obs_step, inferred_step or 0)
        raw_planets = _read(obs, "planets", []) or []
        raw_fleets = _read(obs, "fleets", []) or []
        raw_init = _read(obs, "initial_planets", []) or []
        self.ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0

        self.planets = [Planet(*p) for p in raw_planets]
        self.fleets = [Fleet(*f) for f in raw_fleets]
        self.initial_by_id = {Planet(*p).id: Planet(*p) for p in raw_init}

        # Comets travel along elliptical paths (NOT orbital), so our orbital
        # prediction can't aim at them reliably. Track their ids and skip in
        # expand/hammer to avoid sun-bound wasted fleets.
        raw_comet_ids = _read(obs, "comet_planet_ids", []) or []
        self.comet_ids = set(int(x) for x in raw_comet_ids)
        # V12.9 comet-evac: per-comet remaining-life turns. Comet expires when
        # path_index reaches len(paths[i]); ships on board are lost. Used by
        # handle_comet_evac to dump-all before expiration.
        self.comet_remaining = {}
        raw_comet_groups = _read(obs, "comets", []) or []
        # V13.3 Q1: keep raw comet groups for future-position prediction in
        # aim_at_target (previously only extracted remaining-life count, so
        # aiming at a comet used STATIC position → fleet missed moving target).
        self.comets = raw_comet_groups
        for grp in raw_comet_groups:
            try:
                idx = int(grp.get("path_index", 0))
                pids = grp.get("planet_ids", []) or []
                paths = grp.get("paths", []) or []
                for i, pid in enumerate(pids):
                    if i < len(paths):
                        rem = max(0, len(paths[i]) - idx)
                        self.comet_remaining[int(pid)] = rem
            except (AttributeError, TypeError, IndexError):
                continue

        self.planet_by_id = {p.id: p for p in self.planets}
        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner not in (-1, self.player)]
        self.neutral_planets = [p for p in self.planets if p.owner == -1]

        self.remaining_steps = max(1, TOTAL_STEPS - self.step)
        self.is_opening = self.step < PSM_OPENING_TURN
        self.is_late = self.remaining_steps < LATE_FLUSH_REMAINING_TURNS

        # Per-owner tallies (ships in flight + on planets, plus production).
        self.owner_strength = defaultdict(int)
        self.owner_production = defaultdict(int)
        for p in self.planets:
            if p.owner != -1:
                self.owner_strength[p.owner] += int(p.ships)
                self.owner_production[p.owner] += int(p.production)
        for f in self.fleets:
            self.owner_strength[f.owner] += int(f.ships)

        self.my_prod = self.owner_production.get(self.player, 0)
        self.total_prod = sum(self.owner_production.values())
        self.my_prod_share = (self.my_prod / self.total_prod) if self.total_prod else 0.0
        # V12.8cs: dominant-and-late triggers aggressive flush mode early.
        if self.remaining_steps < 80 and self.my_prod_share > 0.55:
            self.is_late = True

        # V12.8s: 4P leader-bashing placeholder — populated after is_2p is set.
        self.leader_id = None
        self.contest_leader = False

        # V12.8d: per-enemy planet count + composite weakness score to drive
        # weakest-enemy preference in 4P. Score blends production, total
        # ships, and planet count; lowest score = weakest. Self and -1
        # excluded. weakest_enemy is None if no enemies exist yet.
        self.owner_planet_count = defaultdict(int)
        for p in self.planets:
            if p.owner not in (-1,):
                self.owner_planet_count[p.owner] += 1
        self.weakest_enemy = None
        self.weakest_enemy_prod_share = 0.0
        if self.total_prod > 0:
            best_score = None
            for owner in self.owner_production.keys():
                if owner in (-1, self.player):
                    continue
                score = (
                    self.owner_production.get(owner, 0) * 0.5
                    + self.owner_strength.get(owner, 0) * 0.3
                    + self.owner_planet_count.get(owner, 0) * 0.2
                )
                if best_score is None or score < best_score:
                    best_score = score
                    self.weakest_enemy = owner
            if self.weakest_enemy is not None:
                their_prod = self.owner_production.get(self.weakest_enemy, 0)
                self.weakest_enemy_prod_share = (
                    their_prod / self.total_prod if self.total_prod else 0.0
                )

        # Pre-compute incoming-arrival ledger once per turn (used by reserve walk
        # and target-defender prediction). MUST be orbital-aware so we don't
        # miss enemy fleets aimed at our orbital planets — that miss would
        # leave the reserve walk thinking the planet is safe and freeing it to
        # fire offensively right before being captured.
        self.arrivals_by_planet = defaultdict(list)
        for f in self.fleets:
            target, eta = fleet_target_planet(f, self.planets, self.initial_by_id, self.ang_vel)
            if target is None:
                continue
            self.arrivals_by_planet[target.id].append((eta, int(f.owner), int(f.ships)))

        # Race-to-neutral: earliest enemy capture turn per neutral. None / missing
        # = no credible enemy threat. Computed lazily inside _compute_enemy_race_eta
        # only for neutrals within plausible range, to avoid the full
        # O(neutrals * enemies) aim_at_target sweep.
        self.enemy_race_eta = _compute_enemy_race_eta(self) if RACE_ENABLED else {}

        # V12.2 R2: lock the player count at the first observation that has
        # actual planets visible. The step-0 obs in this env has an empty
        # planets list, which would make num_players default to max(2, 0) = 2
        # and falsely set is_2p=True for 4P games. Skip until we see real data.
        global _game_num_players
        if _game_num_players is None and self.planets:
            _game_num_players = self.num_players
        self.is_2p = (_game_num_players == 2)

        # V14.0: format-gated globals — set per-game once is_2p is known
        # (global declared at top of __init__)
        if self.is_2p:
            COALITION_MIN_PER_CONTRIBUTOR = COALITION_MIN_PER_CONTRIBUTOR_2P
            DEFENSE_OVERSEND = DEFENSE_OVERSEND_2P
            PSM_OPENING_TURN = PSM_OPENING_TURN_2P
            SO1_STATIC_BONUS = SO1_STATIC_BONUS_2P
        else:
            COALITION_MIN_PER_CONTRIBUTOR = COALITION_MIN_PER_CONTRIBUTOR_4P
            DEFENSE_OVERSEND = DEFENSE_OVERSEND_4P
            PSM_OPENING_TURN = PSM_OPENING_TURN_4P
            SO1_STATIC_BONUS = SO1_STATIC_BONUS_4P
        # V12.8s: 4P leader-bashing — identify strongest non-self player.
        # contest_leader trips when their lead_score / ours >= LEADER_BASH_RATIO
        # and we're not the top scorer.
        if LEADER_BASH_ENABLED and not self.is_2p:
            lead_scores = {}
            for owner in self.owner_production.keys():
                if owner == -1:
                    continue
                lead_scores[owner] = (
                    self.owner_strength.get(owner, 0) * 0.5
                    + self.owner_production.get(owner, 0) * 0.5
                )
            if lead_scores:
                top_owner = max(lead_scores, key=lambda k: lead_scores[k])
                self.leader_id = top_owner
                my_score = lead_scores.get(self.player, 0)
                top_score = lead_scores.get(top_owner, 0)
                if (
                    top_owner != self.player
                    and my_score > 0
                    and (top_score / my_score) >= LEADER_BASH_RATIO
                ):
                    self.contest_leader = True

        # Adaptive personality (V12.1b): pick a mode based on opponent activity.
        # Opening always stays PATIENT — initial expansions look like aggression
        # but aren't. V12.2 R2: 2P uses MODE_PARAMS_2P (broader search, lower
        # hammer threshold) and a forced-pressure timeout if PATIENT stalls.
        self.mode = _detect_mode(self) if PERSONALITY_ENABLED else "patient"
        # V13.3 T1 (terminal-phase): in the last K turns, override mode to
        # "pressure" — production has no payoff time, all resources should go
        # to offense. Bypasses PSM detection's late-game caution.
        if TERMINAL_PHASE_ENABLED and self.remaining_steps < TERMINAL_PHASE_TURNS:
            self.mode = "pressure"
        params_table = MODE_PARAMS_2P if self.is_2p else MODE_PARAMS
        self.mode_params = params_table[self.mode]

        # V14.1a (Phase 3.1): 2P stop-expanding gate. When we have a stable
        # prod-share lead in 2P, downstream expand sites skip neutrals.
        self.stop_expanding_2p = (
            STOP_EXPAND_2P_ENABLED
            and self.is_2p
            and self.step >= STOP_EXPAND_TURN_MIN_2P
            and self.my_prod_share >= STOP_EXPAND_PROD_SHARE_2P
        )

        # V14.1f (Phase 3.5, Idea 4): combat-contact gate. Detects if any
        # enemy fleet is inbound on one of our planets OR any of our fleets
        # is outbound to an enemy planet. Both signal active engagement.
        # iter 8a DISABLED — see PHASE_3_5 constants. Kept for future re-enable.
        self.in_combat_contact = False
        if COMBAT_STOP_EXPAND_ENABLED:
            my_ids = {p.id for p in self.my_planets}
            enemy_ids = {p.id for p in self.enemy_planets}
            for pid, arrs in self.arrivals_by_planet.items():
                if pid in my_ids:
                    for _eta, owner, ships in arrs:
                        if owner != self.player and owner != -1 and ships >= COMBAT_CONTACT_MIN_SHIPS:
                            self.in_combat_contact = True
                            break
                elif pid in enemy_ids:
                    for _eta, owner, ships in arrs:
                        if owner == self.player and ships >= COMBAT_CONTACT_MIN_SHIPS:
                            self.in_combat_contact = True
                            break
                if self.in_combat_contact:
                    break
        self.combat_stop_expand = (
            COMBAT_STOP_EXPAND_ENABLED
            and self.in_combat_contact
            and self.step >= COMBAT_STOP_EXPAND_TURN_MIN
            and (not COMBAT_STOP_EXPAND_4P_ONLY or not self.is_2p)
        )

        # V14.1f (Phase 3.5, iter 8b): prod-lag gate. When my_prod_share is
        # below threshold for the format, neutral expansion can't catch up
        # to the leader — only enemy captures can. Drop non-cheap neutrals.
        prod_lag_thresh = (
            PROD_LAG_STOP_EXPAND_THRESH_2P if self.is_2p
            else PROD_LAG_STOP_EXPAND_THRESH_4P
        )
        self.prod_lag_stop_expand = (
            PROD_LAG_STOP_EXPAND_ENABLED
            and self.step >= PROD_LAG_STOP_EXPAND_TURN_MIN
            and self.my_prod_share < prod_lag_thresh
        )

        # V14.1g iter 8c: enemy-tempo (N+ enemy planets just launched). [PARKED]
        self.enemy_tempo_stop_expand = (
            ENEMY_TEMPO_STOP_EXPAND_ENABLED
            and self.step >= ENEMY_TEMPO_STOP_EXPAND_TURN_MIN
            and FLEET_INTENT_ENABLED
            and len(_enemy_recently_launched) >= ENEMY_TEMPO_STOP_EXPAND_MIN_LAUNCHES
        )

        # V14.1g iter 8d: easy-enemy targets exist within reach. [DISABLED]
        self.easy_enemy_stop_expand = False
        if EASY_ENEMY_STOP_EXPAND_ENABLED and self.step >= EASY_ENEMY_STOP_EXPAND_TURN_MIN:
            easy_count = 0
            for ep in self.enemy_planets:
                if int(ep.ships) > EASY_ENEMY_MAX_GARRISON:
                    continue
                for mp in self.my_planets:
                    if dist(mp.x, mp.y, ep.x, ep.y) <= EASY_ENEMY_MAX_DIST:
                        easy_count += 1
                        break
                if easy_count >= EASY_ENEMY_MIN_COUNT:
                    break
            self.easy_enemy_stop_expand = (easy_count >= EASY_ENEMY_MIN_COUNT)

        # V14.1h iter 8g: stockpile-readiness (any planet with idle ships).
        self.stockpile_stop_expand = False
        if STOCKPILE_STOP_EXPAND_ENABLED and self.step >= STOCKPILE_STOP_EXPAND_TURN_MIN:
            for mp in self.my_planets:
                if int(mp.ships) >= STOCKPILE_STOP_EXPAND_MAX_GARRISON:
                    self.stockpile_stop_expand = True
                    break

        # V14.1i iter 8h: prod-lead stop-expand for 4P (mirrors stop_expanding_2p).
        self.prod_lead_stop_expand_4p = (
            PROD_LEAD_STOP_EXPAND_4P_ENABLED
            and not self.is_2p
            and self.step >= PROD_LEAD_STOP_EXPAND_4P_TURN_MIN
            and self.my_prod_share >= PROD_LEAD_STOP_EXPAND_4P_THRESH
        )

        # V14.1j iter 8i: turn-cutoff (strict, no cheap-pick).
        self.turn_cutoff_stop_expand = (
            TURN_CUTOFF_STOP_EXPAND_ENABLED
            and self.step >= TURN_CUTOFF_STOP_EXPAND_TURN
        )

        # V14.1h iter 8e: no cheap neutrals left within reach. (2P-only)
        self.neutral_saturation_stop_expand = False
        if (
            NEUTRAL_SATURATION_STOP_EXPAND_ENABLED
            and self.step >= NEUTRAL_SATURATION_TURN_MIN
            and (not NEUTRAL_SATURATION_2P_ONLY or self.is_2p)
        ):
            any_cheap = False
            for n in self.planets:
                if n.owner != -1 or n.id in self.comet_ids:
                    continue
                if int(n.ships) > NEUTRAL_SATURATION_CHEAP_GARRISON:
                    continue
                for mp in self.my_planets:
                    if dist(mp.x, mp.y, n.x, n.y) <= NEUTRAL_SATURATION_REACH_DIST:
                        any_cheap = True
                        break
                if any_cheap:
                    break
            self.neutral_saturation_stop_expand = not any_cheap

        # Unified "lax" flag (cheap-pick exception preserved). Strict
        # stop_expanding_2p still wins where it triggers (no exception).
        self.stop_expand_lax = (
            self.combat_stop_expand
            or self.prod_lag_stop_expand
            or self.enemy_tempo_stop_expand
            or self.easy_enemy_stop_expand
            or self.neutral_saturation_stop_expand
            or self.stockpile_stop_expand
        )

        # 14_4a: 2P focus enemy (the single enemy in 2P).
        # Identifies the one enemy in 2P games so target scoring can apply
        # a strong concentration bonus. None in 4P or when no enemies.
        self.focus_enemy_2p = None
        if F14_4A_2P_FOCUS_ENABLED and self.is_2p:
            for o in self.owner_production.keys():
                if o not in (-1, self.player):
                    self.focus_enemy_2p = o
                    break

        # ===== K-MODE: rich 4P per-enemy state =====
        # 2P uses focus_enemy_2p (single enemy). 4P needs deeper state — there
        # are 3 enemies with complex interactions (kingmaker, synergy when
        # enemy attacks enemy, threat distribution, leader-bash).
        self.enemy_inflight_to_us = defaultdict(int)             # per-enemy ships at us
        self.enemy_inflight_at_other_enemies = defaultdict(int)  # at OTHER enemies
        self.enemy_under_attack_by_others = defaultdict(int)     # incoming from OTHER enemies
        self.enemy_inflight_total = defaultdict(int)
        # Per OUR planet: which enemy is its primary attacker, and how much.
        self.our_planet_primary_threat = {}  # pid -> (attacker_owner, ships)
        # Per ENEMY planet: which other enemies are attacking it, and how much.
        # enemy_planet_id -> {attacker_owner: ships}
        self.enemy_planet_attackers = defaultdict(lambda: defaultdict(int))
        per_planet_threat = defaultdict(lambda: defaultdict(int))
        for f in self.fleets:
            if f.owner == self.player or f.owner == -1:
                continue
            self.enemy_inflight_total[f.owner] += int(f.ships)
            target, _eta = fleet_target_planet(f, self.planets,
                                                self.initial_by_id, self.ang_vel)
            if target is None:
                continue
            if target.owner == self.player:
                self.enemy_inflight_to_us[f.owner] += int(f.ships)
                per_planet_threat[target.id][f.owner] += int(f.ships)
            elif target.owner != -1 and target.owner != f.owner:
                self.enemy_inflight_at_other_enemies[f.owner] += int(f.ships)
                self.enemy_under_attack_by_others[target.owner] += int(f.ships)
                # Per-enemy-planet attack tracking
                self.enemy_planet_attackers[target.id][f.owner] += int(f.ships)
        for pid, by_owner in per_planet_threat.items():
            primary = max(by_owner, key=by_owner.get)
            self.our_planet_primary_threat[pid] = (primary, by_owner[primary])

        # Per-enemy composite strength + leader/weakest detection.
        self.enemy_strength_4p = {}
        self.leader_4p = None
        self.weakest_4p = None
        if not self.is_2p:
            for o in self.owner_production.keys():
                if o in (-1, self.player):
                    continue
                if self.owner_planet_count.get(o, 0) == 0:
                    continue
                self.enemy_strength_4p[o] = (
                    self.owner_strength.get(o, 0)
                    + self.owner_production.get(o, 0) * 8
                    + self.owner_planet_count.get(o, 0) * 5
                )
            if self.enemy_strength_4p:
                self.leader_4p = max(self.enemy_strength_4p,
                                      key=self.enemy_strength_4p.get)
                self.weakest_4p = min(self.enemy_strength_4p,
                                       key=self.enemy_strength_4p.get)

        # K2: defense-reserved sources. For each of OUR planets that currently
        # has a meaningful incoming hostile fleet (per primary_threat), reserve
        # the K closest friendly sources from NEUTRAL expansion this turn.
        # Realistic math: threat is "meaningful" when its size exceeds either
        # my_prod * THREAT_MIN_FACTOR (so production growth alone can't absorb
        # it) OR an absolute floor (cheap enemy probes still need reinforcement
        # if total is high enough). 4P-only; 2P uses the existing focus_enemy_2p
        # dist bias, no separate reserve needed.
        self.defense_reserve_sources = set()
        if (K2_DEFENSE_RESERVE_SOURCES_ENABLED
                and (not K2_DEFENSE_RESERVE_4P_ONLY or not self.is_2p)
                and self.our_planet_primary_threat
                and self.my_planets):
            my_prod_safe = max(1, self.my_prod)
            threat_floor = my_prod_safe * K2_DEFENSE_RESERVE_THREAT_MIN_FACTOR
            for p in self.my_planets:
                ppt = self.our_planet_primary_threat.get(p.id)
                if ppt is None:
                    continue
                _attacker, threat_amt = ppt
                if int(threat_amt) < threat_floor:
                    continue
                # K closest friendly sources to this threatened planet (excluding
                # the planet itself — it can't reinforce itself from itself).
                others = [(dist(s.x, s.y, p.x, p.y), s.id)
                          for s in self.my_planets if s.id != p.id]
                if not others:
                    continue
                others.sort()
                for _d, sid in others[:max(1, K2_DEFENSE_RESERVE_NEAREST_K)]:
                    self.defense_reserve_sources.add(sid)

        self.focus_enemy_4p = None
        if K_ELIMINATE_ENABLED and not self.is_2p and self.enemy_planets:
            # Math-aware focus picker using rich 4P state.
            # Score (lower = better) = kill_cost / effective_value
            # where kill_cost = Σ planets of (ships_to_take × turns_to_reach)
            # and effective_value = base_value (production denied)
            #                      + inflight_relief (their attack on us removed)
            #                      + synergy_bonus (other enemies attacking them
            #                        means cheaper kill — they're already weakened)
            # Skip kingmaker: don't kill if leader_after > my_prod * 1.5.
            my_prod = max(1, self.my_prod)
            cand = []
            for o in self.owner_production.keys():
                if o in (-1, self.player):
                    continue
                if self.owner_planet_count.get(o, 0) == 0:
                    continue
                their_planets = [ep for ep in self.enemy_planets if ep.owner == o]
                if not their_planets:
                    continue
                # Math kingmaker check.
                leader_prod = max(
                    (p for owner, p in self.owner_production.items()
                     if owner not in (-1, self.player, o)),
                    default=0
                )
                if leader_prod > my_prod * 1.5:
                    continue
                # Math kill-cost.
                total_cost = 0.0
                total_value = 0.0
                any_reachable = False
                for ep in their_planets:
                    nearest_d = min((dist(mp.x, mp.y, ep.x, ep.y)
                                      for mp in self.my_planets), default=100.0)
                    if nearest_d <= 60:
                        any_reachable = True
                    # K2: planet-level synergy — when OTHER enemies are
                    # attacking THIS particular ep, its garrison will be
                    # reduced before we arrive. Realistic discount: half the
                    # incoming other-enemy attack (they don't always succeed
                    # but ep will burn ships defending either way).
                    ep_attackers = self.enemy_planet_attackers.get(ep.id, {})
                    synergy_ships = sum(int(s) for owner_a, s in ep_attackers.items()
                                          if owner_a != o and owner_a != self.player) // 2
                    ships_to_take = max(0, int(ep.ships) - synergy_ships) + int(ep.production) * int(nearest_d / 3.0) + 5
                    turns_to_reach = max(1, nearest_d / 3.5)
                    total_cost += ships_to_take * turns_to_reach
                    total_value += int(ep.production)
                if not any_reachable:
                    continue
                # Inflight relief: killing an enemy attacking us frees us
                inflight_at_us = self.enemy_inflight_to_us.get(o, 0)
                # Synergy: how many ships OTHER enemies have aimed at this one
                under_attack = self.enemy_under_attack_by_others.get(o, 0)
                # K3-D: enemy o's OWN ships in flight at OTHER enemies — those
                # ships are not at home defending. Open home = easy kill. Same
                # unit weight as inflight_at_us (divide by 8 = roughly 1 effective
                # production-equivalent per 8 committed ships).
                inflight_elsewhere = self.enemy_inflight_at_other_enemies.get(o, 0)
                # Effective value
                eff_value = (total_value
                             + inflight_at_us / 8.0
                             + under_attack / 6.0
                             + inflight_elsewhere / 8.0)
                score = total_cost / max(1, eff_value)
                cand.append((score, o))
            if cand:
                cand.sort()
                self.focus_enemy_4p = cand[0][1]
            # Leader-bash override: if a single enemy is 2x our strength, we
            # MUST focus on them (otherwise they run away). Overrides kingmaker.
            my_strength = (self.owner_strength.get(self.player, 0)
                            + my_prod * 8
                            + self.owner_planet_count.get(self.player, 0) * 5)
            if (self.leader_4p is not None
                    and self.enemy_strength_4p.get(self.leader_4p, 0) > my_strength * 2.0):
                # Only override if leader has reachable planets
                leader_reachable = any(
                    dist(mp.x, mp.y, ep.x, ep.y) <= 60
                    for mp in self.my_planets
                    for ep in self.enemy_planets if ep.owner == self.leader_4p
                )
                if leader_reachable:
                    self.focus_enemy_4p = self.leader_4p

    @property
    def num_players(self):
        owners = set()
        for p in self.planets:
            if p.owner != -1:
                owners.add(p.owner)
        for f in self.fleets:
            owners.add(f.owner)
        return max(2, len(owners))


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _compute_enemy_race_eta(world):
    """For each neutral, return earliest turn an enemy could land a capturing
    fleet. Considers (a) enemy fleets already in flight aimed at this neutral,
    and (b) enemy planets that have enough ships and are within reach.
    Returns {neutral_id: eta_int}. Neutrals with no credible threat omitted.

    Used to prioritize uncontested-but-soon-to-be-contested neutrals AND to
    skip targets we'd lose the race for (saving ships for next turn).
    """
    out = {}
    if not world.neutral_planets:
        return out

    for n in world.neutral_planets:
        needed = int(n.ships) + 1
        earliest = None

        # (a) Enemy fleets already aimed here.
        for eta, owner, ships in world.arrivals_by_planet.get(n.id, []):
            if owner == world.player or owner == -1:
                continue
            if ships < needed:
                continue
            if earliest is None or eta < earliest:
                earliest = int(eta)

        # (b) Enemy planets that could launch right now.
        for ep in world.enemy_planets:
            if int(ep.ships) < needed:
                continue
            d = dist(ep.x, ep.y, n.x, n.y)
            if d > RACE_MAX_NEUTRAL_DIST:
                continue
            # V12.7a: sun-block check on the enemy's launch path. The engine
            # does not let fleets fly through the sun, so an enemy across the
            # sun cannot threaten this neutral from that source. Without this
            # check, Euclidean ETA falsely concedes races we'd actually win,
            # starving expansion of free targets (research §1.5).
            if safe_geometry(ep.x, ep.y, ep.radius, n.x, n.y, n.radius) is None:
                continue
            # Optimistic ETA — best-case fleet speed at full ship count. Skips
            # the costly orbital aim; we'd rather over-estimate the enemy
            # threat (race more cautiously) than miss a credible threat.
            min_turns = max(1, int(math.ceil(d / fleet_speed(int(ep.ships)))))
            if min_turns > RACE_HORIZON_TURNS:
                continue
            if earliest is None or min_turns < earliest:
                earliest = min_turns

        if earliest is not None:
            out[n.id] = earliest
    return out


def _detect_mode(world):
    """Pick a personality mode from the current snapshot.

    Aggression score = (enemy ships in flight) / (total enemy ships, in flight
    or on planets). A high ratio means enemies are committing to attacks; a
    low ratio means they're stockpiling / quiet. We stay PATIENT during the
    opening since initial expansions look like aggression but aren't.

    V12.2 R2: in 2P, sustained PATIENT with no production-share gain forces
    escalation (10 turns → OPPORTUNISTIC, 20 turns → PRESSURE). This is the
    Bocsimacko "value action over inaction" principle — patient-vs-patient
    1v1 is a stable equilibrium the bot otherwise can't leave.
    """
    if world.is_opening:
        if world.is_2p:
            _record_2p_progress(world.my_prod_share, intended_patient=True, reset=True)
        return "patient"

    enemy_planet_ships = 0
    for p in world.planets:
        if p.owner not in (-1, world.player):
            enemy_planet_ships += int(p.ships)
    enemy_fleet_ships = 0
    for f in world.fleets:
        if f.owner != world.player and f.owner != -1:
            enemy_fleet_ships += int(f.ships)

    enemy_total = enemy_planet_ships + enemy_fleet_ships
    if enemy_total < PERSONALITY_MIN_SAMPLE:
        intended = "patient"
    else:
        aggression = enemy_fleet_ships / float(enemy_total)
        if aggression >= PERSONALITY_AGG_HIGH:
            intended = "pressure"
        elif aggression <= PERSONALITY_AGG_LOW:
            intended = "opportunistic"
        else:
            intended = "patient"

    if not world.is_2p:
        return intended

    # V12.5a: 2P always-PRESSURE after opening. V12.4's streak-gated escalation
    # (10 turns → opportunistic, 20 → pressure) ceded tempo in mirror PATIENT
    # matchups. Side effects of _record_2p_progress preserved for stats.
    _record_2p_progress(world.my_prod_share, intended_patient=(intended == "patient"))
    return "pressure"


def _record_2p_progress(my_prod_share, intended_patient, reset=False):
    """Track production-share trend in 2P. Increment streak whenever the bot
    intends to stay PATIENT and prod-share hasn't grown >EPS over the rolling
    window. Reset streak on opening, on non-PATIENT intent, or on real progress.
    Returns current streak length.
    """
    global _2p_patient_streak, _2p_prod_share_history
    if reset:
        _2p_patient_streak = 0
        _2p_prod_share_history = []
        return 0
    _2p_prod_share_history.append(float(my_prod_share))
    if len(_2p_prod_share_history) > TWO_P_PROD_SHARE_HISTORY:
        _2p_prod_share_history.pop(0)
    if not intended_patient:
        _2p_patient_streak = 0
        return 0
    if len(_2p_prod_share_history) >= TWO_P_PROD_SHARE_HISTORY:
        delta = _2p_prod_share_history[-1] - _2p_prod_share_history[0]
        if delta > TWO_P_PROD_SHARE_PROGRESS_EPS:
            _2p_patient_streak = 0
            return 0
    _2p_patient_streak += 1
    return _2p_patient_streak


# ============================================================
# Per-game persistent state (reset on obs.step == 0)
# ============================================================

_agent_step = 0
_hammer_plan = None             # {target_id, target_arrival_abs, committed_strength, launches: {src_id: {fire_turn_abs, ships, angle}}}
_planet_idle_counts = {}        # planet_id -> consecutive-no-action turns
_promoted_stockpiles = set()    # planet ids promoted to permanent stockpile
_game_num_players = None        # V12.2 R2: locked at game start, used for 2P-only logic
_2p_patient_streak = 0          # V12.2 R2: forced-pressure timeout counter
_2p_prod_share_history = []     # V12.2 R2: rolling prod-share window

# V12.8c — neutral watchlist. _neutral_prev_ships records each neutral's
# ship count from the previous turn. _neutral_wounded is the set of
# neutral ids whose ship count dropped >= NEUTRAL_WATCHLIST_MIN_DROP on the
# most recent transition (i.e. someone else attacked them). Recomputed
# each turn from the delta; not cumulative.
_neutral_prev_ships = {}
_neutral_wounded = set()

# V13.3 F1 (fleet-intent): track enemy planet ship deltas. A sudden drop in
# enemy planet ships = they launched a fleet from there = source is now
# temporarily vulnerable. Boost hammer priority on these targets.
_enemy_prev_ships = {}
_enemy_recently_launched = set()  # enemy planet ids that lost ships last turn

# V13.3 R1 (recapture priority): track which planets just flipped from US to
# an enemy. They're brand-new captures with tiny garrison — easy recapture.
_planet_prev_owner = {}        # planet_id -> previous owner
_freshly_lost_planets = set()  # planets that just flipped from us to enemy
# V14.2 (Phase 3.6, Idea 6a): fresh-capture tracker for mega-hammer inheritance.
_freshly_captured_planets = set()  # planets we just captured this turn
_planet_capture_age = {}       # planet_id -> turns since we captured it

# Persistent dispatch ledger. fleet_target_planet can't reliably attribute
# in-flight fleets to orbital targets (the target has rotated since launch),
# so we track our own commitments here. Each entry: {target_id, ships,
# arrival_abs}. Pruned each turn once the fleet should have arrived. Reset
# on a new game.
_pending_commitments = []

# V12.8et: per-opponent behavior profile (4P only). owner_id -> dict with
# rolling windows for the last OPP_PROFILE_WINDOW turns:
#   emit:  ships_in_flight / max(1, total_ships)
#   stock: max-planet-ships
#   plan:  planet count
OPP_PROFILE_WINDOW = 20
_opp_profile = {}


def _update_opp_profile_4p(world):
    """V12.8et: collect rolling per-enemy behavioral signals. 4P-only;
    caller must check world.is_2p first to avoid 2P side effects.
    """
    global _opp_profile
    if world.step == 0:
        _opp_profile = {}

    plan_ships = defaultdict(int)
    plan_max = defaultdict(int)
    plan_count = defaultdict(int)
    for p in world.planets:
        if p.owner == world.player or p.owner == -1:
            continue
        s = int(p.ships)
        plan_ships[p.owner] += s
        plan_count[p.owner] += 1
        if s > plan_max[p.owner]:
            plan_max[p.owner] = s
    fleet_ships = defaultdict(int)
    for f in world.fleets:
        if f.owner == world.player or f.owner == -1:
            continue
        fleet_ships[f.owner] += int(f.ships)

    enemies = set(plan_count.keys()) | set(fleet_ships.keys())
    for owner in enemies:
        ps = plan_ships.get(owner, 0)
        fs = fleet_ships.get(owner, 0)
        total = ps + fs
        emit = (fs / total) if total else 0.0
        prof = _opp_profile.setdefault(owner, {"emit": [], "stock": [], "plan": []})
        prof["emit"].append(emit)
        prof["stock"].append(plan_max.get(owner, 0))
        prof["plan"].append(plan_count.get(owner, 0))
        if len(prof["emit"]) > OPP_PROFILE_WINDOW:
            prof["emit"] = prof["emit"][-OPP_PROFILE_WINDOW:]
            prof["stock"] = prof["stock"][-OPP_PROFILE_WINDOW:]
            prof["plan"] = prof["plan"][-OPP_PROFILE_WINDOW:]

    world.opp_profile = _opp_profile


# ============================================================
# Defender-at-arrival prediction (with in-flight fleets factored in)
# ============================================================

def predict_defender_at_arrival(world, target, arrival_turn):
    """Owner + ship count on `target` at `arrival_turn` (turns from now), using
    the same combat rules as the env: each turn growth, then resolve arrivals."""
    arrivals = world.arrivals_by_planet.get(target.id, [])
    by_turn = defaultdict(list)
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        by_turn[eta].append((owner, ships))

    owner = target.owner
    garrison = float(target.ships)
    horizon = max(1, int(math.ceil(arrival_turn)))

    for t in range(1, horizon + 1):
        if owner != -1:
            garrison += int(target.production)
        group = by_turn.get(t)
        if group:
            owner, garrison = _resolve_combat(owner, garrison, group)
    return owner, max(0.0, garrison)


def _resolve_combat(owner, garrison, arrivals):
    """Match the env's resolve rule: top-attacker minus second-attacker wins; ties = neutral."""
    by_owner = defaultdict(int)
    for o, s in arrivals:
        by_owner[o] += s
    if not by_owner:
        return owner, max(0.0, garrison)
    sorted_o = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_o, top_s = sorted_o[0]
    if len(sorted_o) > 1 and top_s == sorted_o[1][1]:
        survivor_o, survivor_s = -1, 0
    elif len(sorted_o) > 1:
        survivor_o, survivor_s = top_o, top_s - sorted_o[1][1]
    else:
        survivor_o, survivor_s = top_o, top_s

    if survivor_s <= 0:
        return owner, max(0.0, garrison)
    if owner == survivor_o:
        return owner, garrison + survivor_s
    garrison -= survivor_s
    if garrison < 0:
        return survivor_o, -garrison
    return owner, garrison


# ============================================================
# V12.8av — Forward simulator (matches kaggle_environments orbit_wars step)
# ============================================================
FWD_SIM_ENABLED = os.environ.get("V128_FWD_SIM", "1") != "0"
FWD_LOOKAHEAD_HORIZON = 25
FWD_LOOKAHEAD_TOP_K = 6          # V12.8ay': widen re-rank to 6
FWD_MAX_FLEETS = 80


def _fwd_clone(world):
    planet_ids = []
    planet_owner = {}
    planet_ships = {}
    planet_xy = {}
    planet_radius = {}
    planet_prod = {}
    orbital = {}
    for p in world.planets:
        if p.id in world.comet_ids:
            continue
        planet_ids.append(p.id)
        planet_owner[p.id] = int(p.owner)
        planet_ships[p.id] = float(p.ships)
        planet_xy[p.id] = (float(p.x), float(p.y))
        planet_radius[p.id] = float(p.radius)
        planet_prod[p.id] = int(p.production)
        init = world.initial_by_id.get(p.id)
        if init is not None:
            dx = float(init.x) - CENTER_X
            dy = float(init.y) - CENTER_Y
            r = math.sqrt(dx * dx + dy * dy)
            if r + p.radius < ROTATION_LIMIT:
                orbital[p.id] = (r, math.atan2(dy, dx))
    fleets = []
    next_id = 0
    for f in world.fleets:
        fleets.append([int(f.id), int(f.owner), float(f.x), float(f.y),
                       float(f.angle), int(f.ships)])
        next_id = max(next_id, int(f.id))
    return {
        "planet_ids": planet_ids,
        "planet_owner": planet_owner,
        "planet_ships": planet_ships,
        "planet_xy": planet_xy,
        "planet_radius": planet_radius,
        "planet_prod": planet_prod,
        "orbital": orbital,
        "fleets": fleets,
        "step": int(world.step),
        "ang_vel": float(world.ang_vel),
        "next_fleet_id": next_id + 1,
    }


def _fwd_inject_launch(state, src_id, angle, ships):
    if src_id not in state["planet_xy"]:
        return False
    if state["planet_ships"][src_id] < ships:
        return False
    state["planet_ships"][src_id] -= ships
    radius = state["planet_radius"][src_id]
    sx, sy = state["planet_xy"][src_id]
    fx = sx + math.cos(angle) * (radius + 0.1)
    fy = sy + math.sin(angle) * (radius + 0.1)
    owner = state["planet_owner"][src_id]
    state["fleets"].append([state["next_fleet_id"], int(owner), fx, fy,
                            float(angle), int(ships)])
    state["next_fleet_id"] += 1
    return True


def _fwd_step(state):
    for pid in state["planet_ids"]:
        if state["planet_owner"][pid] != -1:
            state["planet_ships"][pid] += state["planet_prod"][pid]
    combat = {pid: [] for pid in state["planet_ids"]}
    surviving = []
    radii = state["planet_radius"]
    xy = state["planet_xy"]
    pids = state["planet_ids"]
    for fl in state["fleets"]:
        ships = fl[5]
        if ships <= 0:
            continue
        speed = fleet_speed(ships)
        old_x, old_y = fl[2], fl[3]
        new_x = old_x + math.cos(fl[4]) * speed
        new_y = old_y + math.sin(fl[4]) * speed
        fl[2] = new_x
        fl[3] = new_y
        if not (0.0 <= new_x <= BOARD and 0.0 <= new_y <= BOARD):
            continue
        if point_to_segment_distance(CENTER_X, CENTER_Y, old_x, old_y, new_x, new_y) < SUN_R:
            continue
        hit_pid = -1
        for pid in pids:
            px, py = xy[pid]
            if point_to_segment_distance(px, py, old_x, old_y, new_x, new_y) < radii[pid]:
                hit_pid = pid
                break
        if hit_pid >= 0:
            combat[hit_pid].append(fl)
        else:
            surviving.append(fl)
    state["step"] += 1
    new_xy = dict(xy)
    for pid, (r, a0) in state["orbital"].items():
        a = a0 + state["ang_vel"] * state["step"]
        new_xy[pid] = (CENTER_X + r * math.cos(a), CENTER_Y + r * math.sin(a))
    still = []
    for fl in surviving:
        hit_pid = -1
        for pid in pids:
            if pid not in state["orbital"]:
                continue
            old_px, old_py = xy[pid]
            new_px, new_py = new_xy[pid]
            if point_to_segment_distance(fl[2], fl[3], old_px, old_py, new_px, new_py) < radii[pid]:
                hit_pid = pid
                break
        if hit_pid >= 0:
            combat[hit_pid].append(fl)
        else:
            still.append(fl)
    state["planet_xy"] = new_xy
    state["fleets"] = still
    for pid, arrivals in combat.items():
        if not arrivals:
            continue
        per_owner = defaultdict(int)
        for fl in arrivals:
            per_owner[fl[1]] += fl[5]
        sorted_o = sorted(per_owner.items(), key=lambda kv: kv[1], reverse=True)
        top_o, top_s = sorted_o[0]
        if len(sorted_o) > 1:
            second_s = sorted_o[1][1]
            if top_s == second_s:
                surv_s, surv_o = 0, -1
            else:
                surv_s, surv_o = top_s - second_s, top_o
        else:
            surv_o, surv_s = top_o, top_s
        if surv_s > 0:
            cur = state["planet_owner"][pid]
            if cur == surv_o:
                state["planet_ships"][pid] += surv_s
            else:
                state["planet_ships"][pid] -= surv_s
                if state["planet_ships"][pid] < 0:
                    state["planet_owner"][pid] = surv_o
                    state["planet_ships"][pid] = -state["planet_ships"][pid]


def _fwd_simulate(state, horizon):
    for _ in range(horizon):
        if len(state["fleets"]) > FWD_MAX_FLEETS:
            break
        _fwd_step(state)
    return state


def _fwd_my_score(state, player):
    total = 0.0
    for pid in state["planet_ids"]:
        if state["planet_owner"][pid] == player:
            total += state["planet_ships"][pid]
    for fl in state["fleets"]:
        if fl[1] == player:
            total += fl[5]
    return total


def _fwd_marginal(world, src_id, angle, ships, player, horizon):
    """V12.8ay: Δ score (with-launch − without-launch) at horizon."""
    state_no = _fwd_clone(world)
    _fwd_simulate(state_no, horizon)
    base = _fwd_my_score(state_no, player)
    state_yes = _fwd_clone(world)
    if not _fwd_inject_launch(state_yes, src_id, angle, int(ships)):
        return 0.0
    _fwd_simulate(state_yes, horizon)
    return _fwd_my_score(state_yes, player) - base


def _fwd_capture_holds_2p(world, src, target, angle, turns, ships, my_player):
    """V12.8av: simulate launching this fleet now; verify the captured
    target is still ours `turns + FWD_STAB_HORIZON` turns later. Returns
    True if capture sticks, False if predicted to flip."""
    state = _fwd_clone(world)
    if not _fwd_inject_launch(state, src.id, angle, int(ships)):
        return True  # couldn't inject; don't veto
    horizon = int(turns) + 15  # FWD_STAB_HORIZON
    _fwd_simulate(state, horizon)
    return state["planet_owner"].get(target.id) == my_player


# ============================================================
# Tempo helpers — avoid double-firing & comet/transient targeting
# ============================================================

def is_targetable(world, target):
    """Comets travel along non-orbital elliptical paths that aim_at_target can't
    predict. Aiming at them produces fleets that wander and often hit the sun.
    Skip them entirely as expansion / hammer targets.

    V12.9 redundant-launch fix: also skip NEUTRAL targets where one of OUR
    fleets is already in flight with enough ships to flip the planet on
    arrival. Prevents wasted small follow-up fleets piling on a neutral that
    is already being captured.

    V12.9 cap55: enforce the neutral hard cap (2P >=55, 4P legacy) here so
    every targeting code path obeys it — the previous per-call check at
    generate_step_actions/handle_expand missed cheap-pickup, multiprong, and
    other paths."""
    if target.id in world.comet_ids:
        return False
    if target.owner == -1:
        # V12.11 saturation audit: sum ALL friendly inbound. Two 5-ship fleets
        # together can flip a 9-ship neutral; previously only per-fleet was
        # checked. Use latest-arrival garrison (production grows defender).
        my_arrivals = sorted(
            ((eta, ships) for eta, owner, ships
             in world.arrivals_by_planet.get(target.id, [])
             if owner == world.player),
            key=lambda x: x[0],
        )
        if my_arrivals:
            total_ships = sum(s for _, s in my_arrivals)
            last_eta = my_arrivals[-1][0]
            if total_ships > garrison_at_arrival(target, last_eta):
                return False
        if _neutral_blocked_by_cap(world, target):
            return False
        # V14.2c: skip costly low-prod neutrals. A prod=1 planet earning 1
        # ship/turn takes ≥15 turns just to repay a 15-ship capture cost,
        # by which point the game might be over. Better to expand to
        # higher-prod neutrals first and then attack enemies.
        # No "all higher prod taken" exception — by user direction, after
        # high-prod expansion runs out we attack enemies, not flop into
        # prod=1 neutrals.
        if (LOW_PROD_NEUTRAL_SKIP_ENABLED
                and int(target.production) <= LOW_PROD_NEUTRAL_SKIP_PROD
                and int(target.ships) >= LOW_PROD_NEUTRAL_SKIP_GARRISON):
            return False
    return True


def _update_neutral_watchlist(world):
    """V12.8c: rebuild the wounded-neutral set from this turn's deltas.
    A neutral that lost >= NEUTRAL_WATCHLIST_MIN_DROP ships since last
    turn is considered wounded — someone else attacked it, so it's now
    cheaper for us to take. _neutral_prev_ships is then refreshed.

    V13.3 F1: also track enemy planet ship-drops as 'recently launched'
    signal. A drop > FLEET_INTENT_MIN_DROP indicates the source committed
    a fleet outward; the source is in a brief vulnerable state."""
    _neutral_wounded.clear()
    if NEUTRAL_HARD_CAP_ENABLED:
        for p in world.neutral_planets:
            prev = _neutral_prev_ships.get(p.id)
            cur = int(p.ships)
            if prev is not None and (prev - cur) >= NEUTRAL_WATCHLIST_MIN_DROP:
                _neutral_wounded.add(p.id)
    _neutral_prev_ships.clear()
    for p in world.neutral_planets:
        _neutral_prev_ships[p.id] = int(p.ships)
    # V13.3 F1: enemy ship-drop tracking
    if FLEET_INTENT_ENABLED:
        _enemy_recently_launched.clear()
        for p in world.enemy_planets:
            prev = _enemy_prev_ships.get(p.id)
            cur = int(p.ships)
            if prev is not None:
                # Account for production growth between turns; require drop
                # over and above what production would have added.
                expected = prev + int(p.production)
                if expected - cur >= FLEET_INTENT_MIN_DROP:
                    _enemy_recently_launched.add(p.id)
        _enemy_prev_ships.clear()
        for p in world.enemy_planets:
            _enemy_prev_ships[p.id] = int(p.ships)
    # V13.3 R1: detect planets that flipped from us to enemy this turn —
    # they have tiny garrison (recently captured), easy recapture targets.
    # V14.2 (Phase 3.6, Idea 6a): also detect planets we just captured,
    # and track capture age for the fresh-capture mega-hammer threshold.
    if R1_RECAPTURE_PRIORITY_ENABLED:
        _freshly_lost_planets.clear()
        _freshly_captured_planets.clear()
        for p in world.planets:
            prev_owner = _planet_prev_owner.get(p.id)
            if prev_owner == world.player and p.owner != -1 and p.owner != world.player:
                _freshly_lost_planets.add(p.id)
            # V14.2 Idea 6a: detect newly-captured (was-not-ours -> ours).
            if (
                FRESH_CAPTURE_INHERITANCE_ENABLED
                and prev_owner is not None
                and prev_owner != world.player
                and p.owner == world.player
            ):
                _freshly_captured_planets.add(p.id)
                _planet_capture_age[p.id] = 0
        # Age existing captures + cleanup
        if FRESH_CAPTURE_INHERITANCE_ENABLED:
            for pid in list(_planet_capture_age.keys()):
                if pid in _freshly_captured_planets:
                    continue
                pp = world.planet_by_id.get(pid)
                if pp is None or pp.owner != world.player:
                    del _planet_capture_age[pid]
                else:
                    _planet_capture_age[pid] += 1
                    if _planet_capture_age[pid] > FRESH_CAPTURE_MAX_AGE:
                        del _planet_capture_age[pid]
        _planet_prev_owner.clear()
        for p in world.planets:
            _planet_prev_owner[p.id] = int(p.owner)


def _neutral_blocked_by_cap(world, target):
    """K-mode math-aware: skip neutral if its capture cost exceeds expected
    production payoff over remaining game (with margin).

    Math: skip if effective_ships > production * remaining_steps * RATIO,
    where RATIO=0.35 means we need 35% of expected payback in cost-equivalent.
    Falls back to legacy fixed cap when remaining_steps unknown.

    V12.9 cap55: ignore neutrals with high garrison. V13.3 N4: use
    effective_garrison_at_arrival projection (estimated 10-turn lookahead)."""
    if not NEUTRAL_HARD_CAP_ENABLED:
        return False
    if target.owner != -1:
        return False
    if NEUTRAL_CAP_USES_EFFECTIVE_GARRISON:
        eff_owner, eff_ships = effective_garrison_at_arrival(target, NEUTRAL_CAP_LOOKAHEAD, world)
        if eff_owner != -1:
            return False  # not a neutral anymore at arrival
        # K-mode math: payback = prod * remaining_steps; cost = eff_ships
        # Skip if cost > payback * MATH_THRESHOLD
        prod = max(1, int(target.production))
        payback = prod * max(0, world.remaining_steps)
        # 2P: stricter (less time advantage); 4P: looser (4P needs every neutral)
        math_threshold = 0.35 if world.is_2p else 0.6
        if eff_ships > payback * math_threshold:
            return True
        # Otherwise: not blocked. (No fallback to fixed cap — the math is the gate.)
        return False
    # Legacy fallback (kept for safety; usually unused since NEUTRAL_CAP_USES_EFFECTIVE_GARRISON=True)
    if world.is_2p:
        return int(target.ships) >= NEUTRAL_HARD_CAP_2P
    if int(target.ships) <= NEUTRAL_HARD_CAP_4P:
        return False
    return target.id not in _neutral_wounded


def _neutral_tempo_ok(world, target, ships, turns):
    """V12.8cq: skip neutral captures whose expected production gain over
    remaining turns doesn't beat the ship cost by NEUTRAL_TEMPO_THRESHOLD.
    4P-only (2P duels make every neutral worth it). Refuses captures that
    repay slowly even if technically positive (kovi-inspired patience)."""
    if not NEUTRAL_TEMPO_FILTER_ENABLED:
        return True
    if world.is_2p:
        return True
    if target.owner != -1:
        return True
    remaining_after = max(0, int(world.remaining_steps) - int(turns))
    net = float(target.production) * remaining_after - float(ships)
    return net >= NEUTRAL_TEMPO_THRESHOLD


def _ti1_extra_margin(world):
    """V13.3 TI1: returns extra margin to require on captures when we're
    trailing the leader in the late game. Tie counts as a win (engine reward=1
    for max-sum players); low-margin failed attacks drop our absolute sum but
    not our enemies' enough to help. Conserve when behind."""
    if not TI1_TIE_FOR_WIN_ENABLED:
        return 0
    if world.remaining_steps > TI1_HORIZON_TURNS:
        return 0
    my_sum = world.owner_strength.get(world.player, 0)
    leader_sum = my_sum
    for owner, ships in world.owner_strength.items():
        if owner == world.player or owner == -1:
            continue
        if ships > leader_sum:
            leader_sum = ships
    if leader_sum - my_sum < TI1_TRAILING_GAP_MIN:
        return 0  # we are leading, tied, or close enough
    return TI1_REQUIRED_EXTRA_MARGIN


def _endgame_roi_ok(world, target, ships, turns):
    """V12.8b: in the last ENDGAME_ROI_TURNS (4P only), refuse neutral captures
    whose expected production growth doesn't repay the ships spent. 4P-only
    because in 2P the differential-value of denying the neutral to the single
    opponent makes marginal late grabs still net-positive at this threshold;
    n=384 test of the un-gated version showed -38 wins 2P, +17pp 4P. Hostile
    targets always allowed. Returns True if firing is OK."""
    if not ENDGAME_ROI_ENABLED:
        return True
    if world.is_2p:
        return True
    if target.owner != -1:
        return True
    if world.step < TOTAL_STEPS - ENDGAME_ROI_TURNS:
        return True
    remaining_after = max(0, int(world.remaining_steps) - int(turns))
    expected_growth = float(target.production) * remaining_after
    # V13.3 E2: engine-correct threshold. Net ship change from capture-and-hold
    # = -G (garrison killed) + prod×T_held. Profitable iff prod×T_held > G.
    # Previous code used `ships` (which is need+1+margin ≈ G + margin + 1) —
    # over-conservative by `margin+1`. Compare against the actual garrison cost.
    threshold = float(target.ships) if E2_USE_GARRISON_THRESHOLD else float(ships)
    return expected_growth > threshold


def friendly_already_committed(world, target_id):
    """Patient ethos: ONE main fleet per target — UNLESS the target is enemy
    and our in-flight fleet undershoots its growing garrison.

    Neutrals don't grow, so a correctly-sized fleet wins or loses on arrival;
    a follow-up there is wasted ships (Bocsimacko/zvold canonical rule). For
    enemy targets, the planet grows by its production rate every turn the
    fleet is in flight, so a single source from long range can fail to
    capture; allow a sequenced follow-up only when no single pending fleet
    is sufficient at its own arrival turn.
    """
    target = world.planet_by_id.get(target_id)
    if target is None:
        return False
    pending = [c for c in _pending_commitments if c["target_id"] == target_id]
    if not pending:
        return False
    # Neutrals + own planets: any pending fleet locks the target.
    if target.owner == -1 or target.owner == world.player:
        return sum(c["ships"] for c in pending) > 0
    # Enemy target: block only if at least one pending fleet alone can capture
    # at its own ETA (factoring growth). If every pending fleet undershoots,
    # permit a follow-up (V12.2 R1a).
    for c in pending:
        eta = int(c["arrival_abs"]) - int(world.step)
        if eta <= 0:
            continue
        if int(c["ships"]) >= needed_to_capture(target, eta):
            return True
    return False


def _commit_fleet(world, moves, spent, target_locked,
                  src_id, target_id, angle, turns, ships):
    """Single point of truth for firing a fleet: appends move, charges spent,
    locks target this turn, and records the persistent commitment so future
    turns know we already engaged this target."""
    # V13.3 P3: stamp target's owner at commit time; pruner drops commitments
    # whose target ownership has changed.
    target_obj = world.planet_by_id.get(int(target_id))
    owner_at_commit = int(target_obj.owner) if target_obj is not None else -2

    # K6-A: replay-derived send-fraction governor. Applied here because
    # _commit_fleet is the single chokepoint for ALL launches (every handler
    # routes through this). Cap is target-type-aware: self-reinforce gets
    # the tightest cap (bovard winners 0.50 median vs ours 0.60), enemy
    # captures get a looser cap, neutrals in between. See K6 docs at top.
    if K6_REPLAY_SEND_CAP_ENABLED and target_obj is not None:
        src_planet = world.planet_by_id.get(int(src_id))
        if src_planet is not None:
            src_ships_now = int(src_planet.ships)
            if src_ships_now >= K6_REPLAY_SEND_CAP_MIN_SRC_SHIPS:
                if target_obj.owner == world.player:
                    tgt_type = "self"
                elif target_obj.owner == -1:
                    tgt_type = "neutral"
                else:
                    tgt_type = "enemy"
                cap_frac = k6_send_cap(int(world.step), tgt_type)
                max_send = int(src_ships_now * cap_frac)
                if max_send < K6_REPLAY_SEND_CAP_MIN_OUTPUT:
                    max_send = K6_REPLAY_SEND_CAP_MIN_OUTPUT
                if int(ships) > max_send:
                    ships = max_send

    moves.append([src_id, float(angle), int(ships)])
    spent[src_id] += int(ships)
    target_locked.add(target_id)
    _pending_commitments.append({
        "target_id": int(target_id),
        "ships": int(ships),
        "arrival_abs": int(world.step) + int(turns),
        "owner_at_commit": owner_at_commit,
    })
    if os.environ.get("ORBIT_TRACE"):
        try:
            with open(os.environ["ORBIT_TRACE"], "a") as fh:
                fh.write(
                    f"t={world.step} src={src_id} tgt={target_id} ships={ships} eta={turns}\n"
                )
        except Exception:
            pass


def plan_solo_capture(world, src, tgt, max_avail, max_travel):
    """Plan a single-fleet capture (angle, turns, ships) honoring all the
    fleet-quality rules. Returns None if no viable shot exists.

    Critical: aiming uses fleet_speed(ships), so a different ship count than
    we end up sending produces a wrong angle and the fleet wanders / hits the
    sun. We aim, decide ships, then RE-AIM with the exact ship count.
    """
    # V12.13 distance-conditional floor (2P-only): short hops don't need 8.
    # F3: three-bucket — SAFE short-hop neutral (5), HARD enemy w/ strong garrison (12), DEFAULT (8).
    raw_dist = dist(src.x, src.y, tgt.x, tgt.y)
    if F3_THREE_BUCKET_ENABLED:
        if tgt.owner == -1 and raw_dist < F3_SAFE_DIST:
            min_floor = F3_SAFE_FLOOR
        elif (tgt.owner != -1 and tgt.owner != world.player
              and int(tgt.ships) >= F3_HARD_GARRISON):
            min_floor = F3_HARD_FLOOR
        else:
            min_floor = MIN_DISPATCH_SHIPS
    else:
        min_floor = 5 if (world.is_2p and raw_dist < 12.0) else MIN_DISPATCH_SHIPS
    if max_avail < min_floor:
        return None
    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    need = effective_needed_to_capture(tgt, turns, world)  # V13.3 N1
    margin = EXPAND_MIN_MARGIN_4P if not world.is_2p else EXPAND_MIN_MARGIN
    # V13.3 X8b: preferred margin with graceful fallback. Try need+margin
    # first; if insufficient ships, fall back to need+0 (still flips planet).
    # Avoids X8 catastrophic regression where bumping margin globally rejected
    # captures when sources were tight.
    extra = X8B_2P_EXTRA if world.is_2p else 0
    # V13.3 TI1: when trailing in the endgame, require additional margin so
    # only near-certain captures fire. Stacks with X8b's graceful fallback —
    # if the higher preferred can't be paid, we still gracefully degrade to
    # need+margin (and finally need-only). Tie-for-win conservation only.
    extra += _ti1_extra_margin(world)
    preferred = max(min_floor, need + margin + extra)
    # V13.3 SP1: for long-distance captures, bigger fleets travel faster
    # (engine speed = 1+5*(log(ships)/log(1000))^1.5). Bump to SP1_LONG_DIST_SHIPS
    # if target is far. Graceful fallback if insufficient ships.
    if SP1_SPEED_AWARE_ENABLED:
        raw_dist = dist(src.x, src.y, tgt.x, tgt.y)
        if raw_dist >= SP1_LONG_DIST_THRESHOLD:
            preferred = max(preferred, min(SP1_LONG_DIST_SHIPS, max_avail))
    if preferred <= max_avail:
        ships = preferred
    else:
        ships = max(min_floor, need + margin)
        if ships > max_avail:
            ships = max(min_floor, need)  # last resort: just enough to flip
    if ships < min_floor or ships > max_avail:
        return None
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim2 is None:
        return None
    angle, turns = aim2
    if turns > max_travel:
        return None
    need2 = effective_needed_to_capture(tgt, turns, world)  # V13.3 N1
    if ships < need2 + margin:
        ships = need2 + margin
        if ships > max_avail:
            return None
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim3 is None:
            return None
        angle, turns = aim3
        if turns > max_travel:
            return None
    # V13.3 AS1 (anti-second-place 4P): engine combat among same-eta arrivals
    # is top-minus-second; the second-place owner loses ALL ships. If a larger
    # enemy fleet lands the same turn we do, our entire fleet is wiped. Skip
    # the launch. 4P only — 2P has at most one enemy to clash with and
    # existing N1/snipe checks cover it.
    if AS1_ANTI_SECOND_ENABLED and not world.is_2p:
        for eta, owner, e_ships in world.arrivals_by_planet.get(tgt.id, []):
            if int(eta) != int(turns):
                continue
            if owner == world.player or owner == -1:
                continue
            if int(e_ships) >= int(ships):
                return None  # we'd be #2 → wiped
    # V12.9 forward-sim filter: simulate +FWD_SIM_HORIZON turns with our
    # hypothetical capture. Veto only NEUTRAL captures where a known enemy
    # ends up owning our planet — this is the snipe risk our existing
    # _capture_holds_against_snipe is too narrow to catch (it doesn't consider
    # multi-arrival combat math). 4P only.
    if FWD_SIM_FILTER_ENABLED and not world.is_2p and tgt.owner == -1:
        proj = forward_project(
            world,
            our_capture_target=tgt.id,
            our_capture_turn=int(turns),
            our_capture_ships=int(ships),
            horizon=FWD_SIM_HORIZON,
            project_opponent_moves=True,
            opponent_emit_fraction=0.30,
        )
        end_owner, end_ships = proj.get(tgt.id, (-1, 0))
        # Only veto when projected enemy holds with substantial garrison —
        # strict cases. If they barely hold (few ships), our followup may
        # still flip; don't over-veto.
        if end_owner != world.player and end_owner != -1 and end_ships > 5:
            return None
    return angle, turns, int(ships)


# ============================================================
# Mode 2 — Defense
# ============================================================

def handle_defense(world, rescue_needs, available, spent, target_locked,
                   moves, mode_log):
    """Rescue siblings flagged by absorb. Single source preferred; 2-source
    coalition fallback. Each rescuer respects its own reserve and arrives by
    deadline. Locked rescue targets prevent over-rescue.

    V14.2 (Phase 3.8): preemptive doom-evac. When total incoming enemy
    ships overwhelm garrison+future_production, the planet is definitely
    doomed even with rescue. Skip rescue (which wastes ships) and evac
    directly. User-observed scenario: 40 garrison, 10+49 incoming → solo
    rescue would send a sub-need fleet and still lose; better to evac.
    """
    if not rescue_needs:
        return

    for victim_id, (deficit, deadline, victim) in rescue_needs.items():
        if victim_id in target_locked:
            continue
        need = deficit + DEFENSE_OVERSEND

        # V14.2 Phase 3.8/3.11: preemptive evac for clearly-doomed planets.
        # 2P: sum total incoming enemy (single enemy player → sum is real threat).
        # 4P: use LARGEST_SINGLE_ENEMY total — engine top-minus-second combat
        # means 3 enemies attacking same planet annihilate each other; the
        # single biggest owner's total is what actually overwhelms us.
        if PREEMPTIVE_DOOM_EVAC_ENABLED and (not PREEMPTIVE_DOOM_EVAC_2P_ONLY or world.is_2p):
            # K-mode math-aware doom check: use effective_garrison_at_arrival
            # projection (which already accounts for combat resolution) instead
            # of the hardcoded 1.20 ratio. If projected owner != us at deadline,
            # the planet is mathematically doomed → evac.
            # Keep a small safety buffer (1.08 rather than 1.20) to handle
            # uncertainty (we may have ships we haven't sent yet).
            # Math-aware window: if no explicit deadline, use the earliest
            # incoming enemy arrival eta (when threat actually lands), or fall
            # back to remaining game steps if no incoming.
            if deadline is not None:
                window = deadline
            else:
                enemy_etas = [eta for eta, owner, ships
                              in world.arrivals_by_planet.get(victim_id, [])
                              if owner != world.player and owner != -1]
                if enemy_etas:
                    window = max(3, min(enemy_etas))
                else:
                    window = min(world.remaining_steps, 15)
            try:
                proj_owner, proj_ships = effective_garrison_at_arrival(victim, window, world)
            except Exception:
                proj_owner, proj_ships = int(victim.owner), int(victim.ships)
            # Also compute classic threat for fallback / 2P consistency.
            enemy_arrivals = [
                (eta, owner, int(ships)) for eta, owner, ships
                in world.arrivals_by_planet.get(victim_id, [])
                if owner != world.player and owner != -1
            ]
            if world.is_2p or not PREEMPTIVE_EVAC_USE_LARGEST_SINGLE_ENEMY_4P:
                threat_metric = sum(ships for _eta, _owner, ships in enemy_arrivals)
            else:
                by_owner = defaultdict(int)
                for _eta, owner, ships in enemy_arrivals:
                    by_owner[owner] += ships
                threat_metric = max(by_owner.values()) if by_owner else 0
            garrison_at_deadline = int(victim.ships) + int(victim.production) * int(window)
            # Trigger evac if EITHER:
            # - Math projection says we lose the planet (proj_owner != us), OR
            # - Classic threat ratio exceeds 1.08 (tighter than the old 1.20)
            doomed_by_math = (proj_owner != world.player and proj_owner != -1)
            doomed_by_ratio = threat_metric > garrison_at_deadline * 1.08
            if doomed_by_math or doomed_by_ratio:
                if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                    continue
                # If evac fails too, fall through to normal rescue path —
                # might as well try (we'd lose either way).

        # Single-source candidates.
        solo = []
        for src in world.my_planets:
            if src.id == victim_id:
                continue
            avail = available[src.id] - spent[src.id]
            if avail < need:
                continue
            aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if deadline is not None and turns > deadline:
                continue
            solo.append((turns, src.id, src, angle, avail))

        if solo:
            solo.sort()  # closest first
            # V12.8ey: iterate sorted solo candidates — re-aim with exact ship
            # count can push turns past deadline, so try next-closest before
            # falling through to coalition.
            fired_solo = False
            last_fail = None
            for _t, src_id, src, _angle_est, avail in solo:
                send = min(avail, need)
                send = max(send, deficit + 1)
                if send < MIN_DISPATCH_SHIPS:
                    send = MIN_DISPATCH_SHIPS if avail >= MIN_DISPATCH_SHIPS else 0
                if send <= 0:
                    last_fail = "doomed-too-poor"
                    continue
                aim_final = aim_at_target(src, victim, send, world.initial_by_id, world.ang_vel, world=world)
                if aim_final is None:
                    last_fail = "doomed-aim-blocked"
                    continue
                angle, turns = aim_final
                if deadline is not None and turns > deadline:
                    last_fail = "doomed-too-slow"
                    continue
                # V12.9 fwd-sim defense check: project the rescue forward,
                # verify victim is still ours at +H. Else try next solo.
                if FWD_SIM_DEFENSE_CHECK and not world.is_2p:
                    proj = forward_project(
                        world,
                        our_capture_target=victim_id,
                        our_capture_turn=int(turns),
                        our_capture_ships=int(send),
                        horizon=FWD_SIM_HORIZON,
                        project_opponent_moves=True,
                        opponent_emit_fraction=0.30,
                    )
                    end_owner, _ = proj.get(victim_id, (-1, 0))
                    if end_owner != world.player:
                        last_fail = "fwd-sim-victim-still-lost"
                        continue
                _commit_fleet(world, moves, spent, target_locked,
                              src_id, victim_id, angle, turns, int(send))
                mode_log[victim_id] = "defended-by-solo"
                mode_log[src_id] = "defense"
                fired_solo = True
                break
            if fired_solo:
                continue
            if last_fail is not None:
                mode_log[victim_id] = last_fail
                # Fall through to coalition only if every solo failed cleanly.

        # 2-source coalition fallback.
        if not COALITION_ENABLED:
            # V14.1b (Phase 3.2): doom evac before logging "doomed"
            if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                continue
            mode_log[victim_id] = "doomed"
            continue
        coalition = _find_defense_coalition(
            world, victim, deadline, need, available, spent
        )
        if coalition is None:
            # V14.1b (Phase 3.2): doom evac before logging "doomed"
            if _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
                continue
            mode_log[victim_id] = "doomed"
            continue
        for src_id, src, angle, ships, turns in coalition:
            _commit_fleet(world, moves, spent, target_locked,
                          src_id, victim_id, angle, turns, int(ships))
            mode_log[src_id] = "defense-coalition"
        mode_log[victim_id] = "defended-by-coalition"


def _try_doom_evac(world, victim, available, spent, target_locked, moves, mode_log):
    """V14.1b (Phase 3.2 V2): doomed planet evacuation.

    When rescue attempts have failed and the planet is about to flip, send
    its garrison to our highest-production friendly within reach. Preserves
    ships that would otherwise be captured. Returns True if a fleet was
    committed.

    V14.2 (Phase 3.6, Idea 5): attack-fallback. If no friendly destination,
    try sending the garrison to a winnable enemy/neutral target instead of
    letting the ships die with the planet. Prioritizes enemy planets in
    _enemy_recently_launched (they just emptied → weakly defended).
    """
    if not DOOM_EVAC_ENABLED:
        return False
    garrison = available[victim.id] - spent[victim.id]
    if garrison < DOOM_EVAC_MIN_SHIPS:
        return False

    # 1. Find best evac destination: K-mode math-aware score.
    # value = garrison + production * 5  (existing hybrid)
    # MINUS incoming_threat_to_dst (avoid evacuating to a planet that ALSO
    # might flip — this is a math improvement over picking blind by garrison).
    friendly_candidates = []
    for dst in world.my_planets:
        if dst.id == victim.id:
            continue
        aim = aim_at_target(victim, dst, garrison, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > DOOM_EVAC_MAX_TRAVEL:
            continue
        # K-mode: subtract incoming hostile force at dst (so we don't evac
        # into another doomed planet).
        dst_incoming = sum(int(ships) for eta, owner, ships
                            in world.arrivals_by_planet.get(dst.id, [])
                            if owner != world.player and owner != -1
                            and eta < turns + 5)
        value = int(dst.ships) + int(dst.production) * 5 - dst_incoming
        friendly_candidates.append((-value, int(turns), dst, angle))
    if friendly_candidates:
        friendly_candidates.sort()
        _score, turns, dst, angle = friendly_candidates[0]
        _commit_fleet(world, moves, spent, target_locked,
                      victim.id, dst.id, angle, turns, int(garrison))
        mode_log[victim.id] = "doom-evac-launched"
        mode_log[dst.id] = "doom-evac-recipient"
        return True

    # 2. V14.2 attack-fallback: convert doomed garrison into a capture.
    if not DOOM_EVAC_ATTACK_FALLBACK_ENABLED:
        return False
    if DOOM_EVAC_ATTACK_FALLBACK_4P_ONLY and world.is_2p:
        return False
    attack_candidates = []
    for dst in world.planets:
        if dst.id == victim.id or dst.owner == world.player:
            continue
        if dst.id in target_locked:
            continue
        if not is_targetable(world, dst):
            continue
        aim = aim_at_target(victim, dst, garrison, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > DOOM_EVAC_MAX_TRAVEL:
            continue
        # Required ships at arrival:
        # - Enemy: current + production * turns (they produce in transit)
        # - Neutral: current (no production)
        is_enemy = dst.owner != -1
        prod = int(dst.production) if is_enemy else 0
        arrival_garrison = int(dst.ships) + prod * int(turns)
        required = arrival_garrison + DOOM_EVAC_ATTACK_OVERKILL
        if int(garrison) < required:
            continue
        # Tiebreak: prefer recently-launched-from enemies (weakened).
        recently_launched_bonus = (
            -DOOM_EVAC_ATTACK_PREFER_LAUNCHED_BONUS
            if (is_enemy and dst.id in _enemy_recently_launched) else 0
        )
        # K-mode: prefer focus_enemy_4p planets (or focus_enemy_2p) — doomed
        # garrison contributes to elimination instead of dying.
        focus_id = getattr(world, "focus_enemy_2p", None) if world.is_2p else \
                    getattr(world, "focus_enemy_4p", None)
        focus_bonus = -3.0 if (is_enemy and focus_id is not None
                                 and dst.owner == focus_id) else 0
        rank = (
            recently_launched_bonus + focus_bonus,
            -int(dst.production),
            int(turns),
            int(required),
        )
        attack_candidates.append((rank, dst, angle, turns))
    if not attack_candidates:
        return False
    attack_candidates.sort(key=lambda x: x[0])
    _rank, dst, angle, turns = attack_candidates[0]
    _commit_fleet(world, moves, spent, target_locked,
                  victim.id, dst.id, angle, turns, int(garrison))
    mode_log[victim.id] = "doom-evac-attack"
    mode_log[dst.id] = "doom-evac-attack-target"
    return True


def _find_defense_coalition(world, victim, deadline, need, available, spent):
    """Pick the closest pair of siblings whose combined ships meet `need`, both
    arrive by `deadline`, AND each contributes >= COALITION_MIN_PER_CONTRIBUTOR.
    Re-aims each contributor with its exact ship count.
    Returns [(src_id, src, angle, ships), ...] or None.
    """
    options = []
    for src in world.my_planets:
        if src.id == victim.id:
            continue
        avail = available[src.id] - spent[src.id]
        if avail < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            continue
        _angle_est, turns = aim
        if deadline is not None and turns > deadline:
            continue
        options.append((turns, src.id, src, avail))

    if len(options) < 2:
        return None
    options.sort()  # earlier-arriving first

    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            t_i, sid_i, s_i, a_i = options[i]
            t_j, sid_j, s_j, a_j = options[j]
            if a_i + a_j < need:
                continue
            ratio = a_i / float(a_i + a_j)
            ship_i = max(COALITION_MIN_PER_CONTRIBUTOR,
                         min(a_i, int(round(need * ratio))))
            ship_j = max(COALITION_MIN_PER_CONTRIBUTOR,
                         min(a_j, need - ship_i))
            while ship_i + ship_j < need:
                if ship_i < a_i:
                    ship_i += 1
                elif ship_j < a_j:
                    ship_j += 1
                else:
                    break
            if (ship_i + ship_j < need
                    or ship_i < COALITION_MIN_PER_CONTRIBUTOR
                    or ship_j < COALITION_MIN_PER_CONTRIBUTOR):
                continue
            # Re-aim each contributor with exact ships (speed differs).
            aim_i = aim_at_target(s_i, victim, ship_i, world.initial_by_id, world.ang_vel, world=world)
            aim_j = aim_at_target(s_j, victim, ship_j, world.initial_by_id, world.ang_vel, world=world)
            if aim_i is None or aim_j is None:
                continue
            ang_i, turns_i = aim_i
            ang_j, turns_j = aim_j
            if (deadline is not None
                    and (turns_i > deadline or turns_j > deadline)):
                continue
            return [
                (sid_i, s_i, ang_i, ship_i, turns_i),
                (sid_j, s_j, ang_j, ship_j, turns_j),
            ]
    return None


# ============================================================
# Mode 1b — Comet evacuation (dump-all before expiration)
# ============================================================

COMET_EVAC_REMAINING_TURNS = 3   # if a comet has <=3 path-steps left, evacuate
COMET_EVAC_MIN_SHIPS = 5          # don't bother for tiny garrisons

# V14.1b (Phase 3.2): doom-evacuation. When handle_defense fails to rescue a
# planet (both solo and coalition rescue exhausted) and the planet is about
# to flip to enemy, evacuate its garrison to our highest-prod friendly
# within reach. Preserves ships that would otherwise be captured by enemy.
# Variant V2 from RL/notes/phase3_doom_evac.md.
DOOM_EVAC_ENABLED = True
DOOM_EVAC_MIN_SHIPS = 5           # V14.1n: reverted CMA-ES 6 -> 5
DOOM_EVAC_MAX_TRAVEL = 40         # V14.1n: reverted CMA-ES 45 -> 40

# V14.2 (Phase 3.6, Idea 5): doom-evac attack-fallback.
# If no friendly destination is reachable, try sending garrison to a
# winnable enemy/neutral target instead of letting the planet die. Engine:
# arrival_garrison = current_ships + production * turns (enemies produce;
# neutrals don't). Required = arrival_garrison + overkill. Prioritize
# enemy planets in _enemy_recently_launched (just emptied → vulnerable).
DOOM_EVAC_ATTACK_FALLBACK_ENABLED = True
DOOM_EVAC_ATTACK_FALLBACK_4P_ONLY = True  # iter v1 hurt 2P (-8 wins); 4P 63.0% won
DOOM_EVAC_ATTACK_OVERKILL = 2     # need arrival_garrison + this many ships
DOOM_EVAC_ATTACK_PREFER_LAUNCHED_BONUS = 3  # tiebreak: prefer recently-launched-from

# V14.2 (Phase 3.8): preemptive doom-evac trigger inside handle_defense.
# When total_enemy_incoming > garrison_at_deadline * RATIO, skip rescue
# (it would waste ships on a lost cause) and evac directly. User-observed:
# we had 40 ships, enemy sent 10+49 (59 total) and we stayed/lost; should
# have evac'd to preserve garrison + rebuild attack from friendly.
PREEMPTIVE_DOOM_EVAC_ENABLED = True
PREEMPTIVE_DOOM_EVAC_2P_ONLY = False  # V14.2 Phase 3.11: now uses largest-single-enemy
# in 4P, correctly accounting for top-minus-second mutual destruction.
PREEMPTIVE_EVAC_DOOM_RATIO = 1.20  # incoming > garrison_at_deadline * 1.20 → doomed
PREEMPTIVE_EVAC_DEFAULT_WINDOW = 15  # if no deadline, assume 15-turn window
# V14.2 Phase 3.11: in 4P, use LARGEST_SINGLE_ENEMY_TOTAL instead of total.
# Engine combat = top-minus-second per turn per owner. If 3 enemies each
# send 50 (total 150), they fight each other → ~0 net damage to defender.
# But if ONE enemy sends 100 (single owner total = 100), that's the real
# threat. Use the per-owner top to predict overwhelm correctly in 4P.
PREEMPTIVE_EVAC_USE_LARGEST_SINGLE_ENEMY_4P = True


def handle_comet_evac(world, available, spent, target_locked, moves, mode_log):
    """K-mode math-aware comet evac. Fire when remaining-path turns equals
    shortest-travel-to-destination (we have JUST enough time). If less, dump
    anyway (last chance — ships would otherwise expire with the comet).

    No fixed COMET_EVAC_REMAINING_TURNS gate; the trigger is computed per
    comet from actual travel times to candidate destinations.
    """
    if not world.comet_remaining:
        return
    own_non_comet = [p for p in world.my_planets if p.id not in world.comet_ids]
    if not own_non_comet:
        own_non_comet = [p for p in world.planets
                         if p.owner == -1 and p.id not in world.comet_ids]
        if not own_non_comet:
            return
    for src in world.my_planets:
        rem = world.comet_remaining.get(src.id)
        if rem is None or rem > 20:
            continue  # only consider comets near end-of-life
        if src.id in mode_log:
            continue
        avail = max(0, available[src.id] - spent.get(src.id, 0))
        if avail < COMET_EVAC_MIN_SHIPS:
            continue
        # Pick destination with shortest aim-resolved travel.
        best = None
        best_turns = float("inf")
        best_angle = 0.0
        for dst in own_non_comet:
            if dst.id == src.id:
                continue
            aim = aim_at_target(src, dst, avail, world.initial_by_id,
                                world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns < best_turns:
                best_turns = turns
                best_angle = angle
                best = dst
        if best is None:
            continue
        # Math gate: fire if remaining path turns <= best_turns + 1 (safety).
        # Earlier than that, ships are still being produced — don't evac yet.
        # Note: rem == path turns until comet expires.
        if rem > best_turns + 1:
            continue
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, best.id, best_angle, best_turns, int(avail))
        mode_log[src.id] = "comet-evac"


# ============================================================
# Mode 3 — Expand (solo + coalition)
# ============================================================

def handle_cheap_pickup(world, available, spent, target_locked, moves, mode_log):
    """V12.4d (4P-only): each idle source fires on the cheapest reachable
    low-garrison neutral if it can solo it. Bypasses the K=1 mid-game
    starvation where small free planets sit ignored because the source's
    K=1 nearest is a higher-garrison target. 4P-only — see CHEAP_PICKUP_4P_ONLY.
    """
    if not CHEAP_PICKUP_ENABLED:
        return
    if CHEAP_PICKUP_4P_ONLY and world.is_2p:
        return
    # V12.8j: launch blackout in the final stretch — fleet won't arrive in
    # time to repay; hoard ships at home instead.
    if LAUNCH_BLACKOUT_ENABLED and world.step >= TOTAL_STEPS - LAUNCH_BLACKOUT_TURNS:
        return
    if world.is_opening:
        max_travel = world.mode_params.get("expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        max_travel = world.mode_params["expand_max_travel_mid"]

    cheap_neutrals = [
        p for p in world.neutral_planets
        if int(p.ships) <= CHEAP_PICKUP_MAX_GARRISON
        and p.id not in target_locked
        and is_targetable(world, p)
    ]
    if not cheap_neutrals:
        return
    # F32: skip prod=1 when prod>=2 options exist in cheap pool.
    if CHEAP_PICKUP_MIN_PROD >= 2 and any(int(p.production) >= CHEAP_PICKUP_MIN_PROD for p in cheap_neutrals):
        cheap_neutrals = [p for p in cheap_neutrals if int(p.production) >= CHEAP_PICKUP_MIN_PROD]

    sources = sorted(world.my_planets,
                     key=lambda s: -(available[s.id] - spent[s.id]))
    for src in sources:
        avail = available[src.id] - spent[src.id]
        if avail < MIN_DISPATCH_SHIPS:
            continue
        if mode_log.get(src.id):
            continue
        candidates = []
        for n in cheap_neutrals:
            if n.id in target_locked:
                continue
            if friendly_already_committed(world, n.id):
                continue
            cost = int(n.ships) + 1
            if cost > avail:
                continue
            raw = dist(src.x, src.y, n.x, n.y)
            if raw / MAX_SPEED > max_travel + 4:
                continue
            eff = _effective_target_dist(src, n, world)
            candidates.append((cost, eff, n))
        if not candidates:
            continue
        candidates.sort(key=lambda kv: (kv[0], kv[1]))
        for _cost, _eff, n in candidates:
            plan = plan_solo_capture(world, src, n, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            if RACE_ENABLED:
                enemy_eta = world.enemy_race_eta.get(n.id)
                if enemy_eta is not None and turns > enemy_eta:
                    continue
            if not _capture_holds_against_snipe(world, n, turns, int(ships)):
                continue
            if not _endgame_roi_ok(world, n, int(ships), turns):
                continue
            if not _neutral_tempo_ok(world, n, int(ships), turns):
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, n.id, angle, turns, int(ships))
            mode_log[src.id] = "cheap-pickup"
            break


def _is_cheap_neutral_pick(world, target):
    """V14.1f (Phase 3.5, Idea 4): cheap-pick predicate for combat-contact gate.

    Returns True if the neutral target has small garrison AND there exists
    one of our planets within COMBAT_CHEAP_DIST. Used to preserve free
    pickups while dropping expensive neutrals during active combat.
    """
    if target.owner != -1:
        return True  # not a neutral; bypass
    if int(target.ships) > COMBAT_CHEAP_GARRISON:
        return False
    for mp in world.my_planets:
        if dist(mp.x, mp.y, target.x, target.y) <= COMBAT_CHEAP_DIST:
            return True
    return False


def _handle_search_expand_4p(world, available, spent, target_locked, moves, mode_log):
    """V12.9 Melis search-based expansion (4P only). Generates candidate step
    actions via generate_step_actions, ranks by melis_evaluate gain, commits
    top SEARCH_MAX_ACTIONS_TO_PICK that don't conflict (different targets +
    sources). Returns list of committed source ids so caller can skip them.
    """
    actions = search_step_action(
        world, max_per_source=SEARCH_MAX_PER_SOURCE,
        max_actions_to_eval=12,
        use_depth2=SEARCH_DEPTH2_ENABLED,
    )
    committed_sources = set()
    committed_targets = set()
    for act in actions[:SEARCH_MAX_ACTIONS_TO_PICK * 2]:
        if act["score"] <= 0:
            continue
        src_id = act["source_id"]
        tgt_id = act["target_id"]
        if src_id in committed_sources or tgt_id in committed_targets:
            continue
        if tgt_id in target_locked:
            continue
        # B1 (one-brain): respect brain-reserved-lead — search must not drain it.
        src_status = mode_log.get(src_id)
        if src_status == "brain-reserved-lead":
            continue
        avail = available[src_id] - spent[src_id]
        if avail < act["ships"]:
            continue
        # V12.9 filter-leak fix: search picks bypassed the same vetoes that
        # legacy handle_expand enforces (snipe-hold, endgame-ROI, tempo).
        # Apply them here so search and legacy agree on what "skip" means.
        # Only neutral targets — enemy captures use Melis verification.
        tgt = world.planet_by_id.get(tgt_id)
        # V14.1a (Phase 3.1): drop neutral picks once 2P stop-expand gate opens.
        if (world.stop_expanding_2p or world.prod_lead_stop_expand_4p or world.turn_cutoff_stop_expand) and tgt is not None and tgt.owner == -1:
            continue
        # V14.1f (Phase 3.5, Idea 4): in combat, drop non-cheap neutral picks.
        if world.stop_expand_lax and tgt is not None and tgt.owner == -1:
            if not _is_cheap_neutral_pick(world, tgt):
                continue
        if tgt is not None and tgt.owner == -1:
            turns_act = int(act["arrival_turn"])
            ships_act = int(act["ships"])
            if not _capture_holds_against_snipe(world, tgt, turns_act, ships_act):
                continue
            if not _endgame_roi_ok(world, tgt, ships_act, turns_act):
                continue
            if not _neutral_tempo_ok(world, tgt, ships_act, turns_act):
                continue
        _commit_fleet(world, moves, spent, target_locked,
                      src_id, tgt_id, act["angle"], act["arrival_turn"], act["ships"])
        mode_log[src_id] = "search-expand"
        committed_sources.add(src_id)
        committed_targets.add(tgt_id)
        if len(committed_sources) >= SEARCH_MAX_ACTIONS_TO_PICK:
            break
    return committed_sources


def handle_expand(world, available, spent, target_locked, moves, mode_log):
    # V12.8j: launch blackout in the final stretch.
    if LAUNCH_BLACKOUT_ENABLED and world.step >= TOTAL_STEPS - LAUNCH_BLACKOUT_TURNS:
        return
    # V12.9 Melis search (4P-only): handle prioritized step-action picks first
    if (SEARCH_EXPAND_4P_ENABLED and not world.is_2p) or \
       (SEARCH_EXPAND_2P_ENABLED and world.is_2p):
        _handle_search_expand_4p(world, available, spent, target_locked, moves, mode_log)
        # fall through to existing greedy for remaining sources
    if world.is_opening:
        # V12.3b (2.2b): opening uses mode_params (with .get fallback to globals)
        # so 2P can widen K + travel cap without touching 4P behavior.
        K = world.mode_params.get("expand_k_opening", EXPAND_K_OPENING)
        max_travel = world.mode_params.get("expand_max_travel_opening", EXPAND_MAX_TRAVEL_OPENING)
    else:
        K = world.mode_params["expand_k_mid"]
        max_travel = world.mode_params["expand_max_travel_mid"]

    nonfriendly = [
        p for p in world.planets
        if p.owner != world.player and is_targetable(world, p)
    ]
    # V14.1a (Phase 3.1): drop neutrals when 2P stop-expand gate is open.
    # V14.1i iter 8h: also when 4P prod-lead gate triggers (strict).
    # V14.1j iter 8i: also past turn cutoff (strict).
    # K3-A: state-derived survival floor. If we own materially fewer planets
    # than our fair share (total_inhabitable / num_players), override the
    # stop-expand gates and ALWAYS allow neutrals. Threshold is 3/4 of fair
    # share — at half-share we're already in trouble. Floor scales with map
    # size and player count.
    if K3_ENABLED:
        inhabitable = sum(1 for p in world.planets if p.id not in world.comet_ids)
        num_p = max(2, _game_num_players or world.num_players)
        fair_share = inhabitable / num_p
        below_survival_floor = len(world.my_planets) <= (fair_share * 0.75)
    else:
        below_survival_floor = False
    drop_all_neutrals_gated = (
        (world.stop_expanding_2p or world.prod_lead_stop_expand_4p or world.turn_cutoff_stop_expand)
        and not below_survival_floor
    )
    drop_noncheap_neutrals_gated = (
        world.stop_expand_lax and not below_survival_floor
    )
    if drop_all_neutrals_gated:
        nonfriendly = [p for p in nonfriendly if p.owner != -1]
    # V14.1f (Phase 3.5, Idea 4): in combat, drop non-cheap neutrals.
    elif drop_noncheap_neutrals_gated:
        nonfriendly = [
            p for p in nonfriendly
            if p.owner != -1 or _is_cheap_neutral_pick(world, p)
        ]
    if not nonfriendly:
        return

    def frontier_key(src):
        return min(dist(src.x, src.y, t.x, t.y) for t in nonfriendly)

    sources = sorted(world.my_planets, key=frontier_key)

    for src in sources:
        # V14.1d iter g: hide production-tier reserve from routine spending
        avail = _routine_avail(world, src, available[src.id] - spent[src.id])
        if avail < MIN_DISPATCH_SHIPS:
            continue
        # V12.4d: allow main expand to fire after cheap-pickup pre-pass
        # (spent[src.id] already accounts for the pre-pass spend; we just
        # don't want the source's freebie to lock out a strategic capture).
        status = mode_log.get(src.id)
        if status and status != "cheap-pickup":
            continue  # already used in defense / absorb

        candidates = _nearest_targets(src, world, K, max_travel, target_locked)
        # K2: if this source is the closest defender to a threatened friendly
        # planet, don't drain it on NEUTRAL captures this turn. Enemy captures
        # (offensive value) are still allowed — they shift the strategic balance.
        src_is_reserved = (
            K2_DEFENSE_RESERVE_SOURCES_ENABLED
            and (not K2_DEFENSE_RESERVE_4P_ONLY or not world.is_2p)
            and src.id in getattr(world, "defense_reserve_sources", set())
        )
        fired_solo = False
        for tgt, _approx_dist in candidates:
            if friendly_already_committed(world, tgt.id):
                continue
            if src_is_reserved and tgt.owner == -1:
                continue
            plan = plan_solo_capture(world, src, tgt, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            if RACE_ENABLED and tgt.owner == -1:
                enemy_eta = world.enemy_race_eta.get(tgt.id)
                if enemy_eta is not None and turns > enemy_eta:
                    snipe = _plan_counter_snipe(world, src, tgt, avail, max_travel)
                    if snipe is None:
                        continue
                    angle, turns, ships = snipe
            if tgt.owner == -1 and not _capture_holds_against_snipe(world, tgt, turns, int(ships)):
                continue
            if not _endgame_roi_ok(world, tgt, int(ships), turns):
                continue
            if not _neutral_tempo_ok(world, tgt, int(ships), turns):
                continue
            # V12.8av (2P-only): forward-sim verify capture sticks. Now also
            # extends the check to neutrals (V12.8az), since the simulator's
            # multi-planet view catches snipes the single-planet heuristic misses.
            if (
                FWD_SIM_ENABLED
                and world.is_2p
                and tgt.owner != world.player
                and not _fwd_capture_holds_2p(world, src, tgt, angle, turns, int(ships), world.player)
            ):
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, tgt.id, angle, turns, int(ships))
            mode_log[src.id] = "expand-solo"
            fired_solo = True
            break

        if fired_solo:
            continue
        if not COALITION_ENABLED:
            continue

        # K2: if source is defense-reserved, skip coalition (coalition is
        # neutrals-only by default; same reasoning as solo neutral skip).
        if src_is_reserved and COALITION_NEUTRALS_ONLY:
            continue
        coalition_max_travel = max_travel + COALITION_MAX_TRAVEL_BONUS
        for tgt, _ in candidates:
            if tgt.id in target_locked:
                continue
            if COALITION_NEUTRALS_ONLY and tgt.owner != -1:
                continue
            if friendly_already_committed(world, tgt.id):
                continue
            ok = _try_coalition_expand(
                world, src, tgt, coalition_max_travel, available, spent,
                target_locked, moves, mode_log,
            )
            if ok:
                break


def _effective_target_dist(src, tgt, world):
    """V12.4a rotation-aware distance proxy for target prefilter ranking.

    Predicts target position at expected travel time and returns distance
    to that future position. Static planets unchanged. Orbital planets
    rotating toward us get a shorter effective distance (promote);
    rotating away get longer (demote). One-step approximation — cheap;
    real arrival is computed later by aim_at_target inside plan_solo_capture.
    Affects WHICH targets get inspected when K is small, not which fleets fly.
    """
    raw = dist(src.x, src.y, tgt.x, tgt.y)
    if not ROT_AWARE_RANK_ENABLED:
        return raw
    init = world.initial_by_id.get(tgt.id)
    if init is None:
        return raw
    if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
        return raw
    speed = fleet_speed(50)
    travel = max(1, int(math.ceil(raw / speed)))
    if travel > 60:
        return raw
    px, py = predict_planet_position(tgt, world.initial_by_id, world.ang_vel, travel)
    return dist(src.x, src.y, px, py)


def _counter_snipe_candidates(world, src, max_travel, target_locked):
    """V12.4c: neutrals where a known enemy fleet will capture before us, and
    we can re-flip cheaply on a short follow-up. Returns [(target, raw_dist)]
    sorted by re-flip cost ascending. 2P-only — see COUNTER_SNIPE_2P_ONLY note.
    """
    if not COUNTER_SNIPE_ENABLED:
        return []
    if COUNTER_SNIPE_2P_ONLY and not world.is_2p:
        return []
    out = []
    for n in world.neutral_planets:
        if n.id in target_locked:
            continue
        if not is_targetable(world, n):
            continue
        enemy_eta = None
        enemy_remaining = None
        needed = int(n.ships) + 1
        for eta, owner, ships in world.arrivals_by_planet.get(n.id, []):
            if owner == world.player or owner == -1:
                continue
            if ships < needed:
                continue
            if enemy_eta is None or eta < enemy_eta:
                enemy_eta = int(eta)
                enemy_remaining = ships - int(n.ships)
        if enemy_eta is None:
            continue
        d = dist(src.x, src.y, n.x, n.y)
        speed = fleet_speed(50)
        my_eta_est = max(1, int(math.ceil(d / speed)))
        if my_eta_est > max_travel + 4:
            continue
        delay = my_eta_est - enemy_eta
        if delay < COUNTER_SNIPE_MIN_DELAY or delay > COUNTER_SNIPE_MAX_DELAY:
            continue
        prod = max(0, int(n.production))
        defender_at_my_arrival = max(0, int(enemy_remaining)) + prod * delay
        flip_cost = defender_at_my_arrival + 1
        if flip_cost > COUNTER_SNIPE_MAX_COST:
            continue
        out.append((flip_cost, n, d))
    out.sort(key=lambda kv: kv[0])
    return [(n, d) for _cost, n, d in out]


def _plan_counter_snipe(world, src, tgt, max_avail, max_travel):
    """V12.4c: size a small fleet to re-flip a neutral AFTER a known enemy
    fleet captures it. Returns (angle, turns, ships) or None. 2P-only.
    """
    if not COUNTER_SNIPE_ENABLED or tgt.owner != -1:
        return None
    if COUNTER_SNIPE_2P_ONLY and not world.is_2p:
        return None
    if max_avail < MIN_DISPATCH_SHIPS:
        return None
    enemy_eta = None
    enemy_remaining = None
    needed_to_take = int(tgt.ships) + 1
    for eta, owner, ships in world.arrivals_by_planet.get(tgt.id, []):
        if owner == world.player or owner == -1:
            continue
        if ships < needed_to_take:
            continue
        if enemy_eta is None or eta < enemy_eta:
            enemy_eta = int(eta)
            enemy_remaining = ships - int(tgt.ships)
    if enemy_eta is None:
        return None

    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    delay = turns - enemy_eta
    if delay < COUNTER_SNIPE_MIN_DELAY or delay > COUNTER_SNIPE_MAX_DELAY:
        return None
    prod = max(0, int(tgt.production))
    defender = max(0, int(enemy_remaining)) + prod * delay
    ships = max(MIN_DISPATCH_SHIPS, defender + 1)
    if ships > max_avail or ships > COUNTER_SNIPE_MAX_COST:
        return None
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim2 is None:
        return None
    angle, turns = aim2
    if turns > max_travel:
        return None
    delay2 = turns - enemy_eta
    if delay2 < COUNTER_SNIPE_MIN_DELAY or delay2 > COUNTER_SNIPE_MAX_DELAY:
        return None
    defender2 = max(0, int(enemy_remaining)) + prod * delay2
    if ships < defender2 + 1:
        ships = defender2 + 1
        if ships > max_avail or ships > COUNTER_SNIPE_MAX_COST:
            return None
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim3 is None:
            return None
        angle, turns = aim3
        if turns > max_travel:
            return None
    return angle, turns, int(ships)


def _capture_holds_against_snipe(world, target, arrival_turn, ships_sent):
    """V12.4b: returns True if our post-capture garrison stays >0 through every
    KNOWN enemy fleet arriving within ANTI_SNIPE_HORIZON. Walks surplus +
    production growth between events; subtracts each enemy fleet at its eta;
    refuses if balance ever drops <=0. Friendly follow-ups credited.

    Gated to 2P only (ANTI_SNIPE_2P_ONLY): in 4P with 3 enemies the veto
    fires too often, starving expansion (192-game test: 55 third-place
    finishes vs 12_4a's 4). 2P has only one snipe source so the veto
    targets actual snipe traps without paralyzing expansion.
    """
    if not ANTI_SNIPE_ENABLED:
        return True
    if ANTI_SNIPE_2P_ONLY and not world.is_2p:
        return True
    if target.owner != -1:
        return True
    arrivals = world.arrivals_by_planet.get(target.id, [])
    enemy_after = []
    friendly_after = []
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        if eta <= arrival_turn:
            continue
        if eta - arrival_turn > ANTI_SNIPE_HORIZON:
            continue
        if owner == world.player:
            friendly_after.append((eta, ships))
        elif owner != -1:
            enemy_after.append((eta, ships))

    # V13.3 R8 (reactive-snipe projection): the existing check only saw fleets
    # already in flight. But enemy can REACTIVELY launch after seeing our move
    # — that's the exact bug user observed (we sent 33, captured leaving 1 ship,
    # enemy then sent 6 to snipe). Project each enemy planet as potentially
    # launching REACTIVE_EMIT_FRAC of its ships toward our target.
    if REACTIVE_SNIPE_PROJECTION_ENABLED:
        # K4-C: state-derived min-ships threshold. A planet too small to matter
        # depends on OUR production scale — what's "negligible" for us is the
        # right yardstick. Keep the original 5-ship floor as the minimum so
        # tiny-board parity is preserved.
        min_enemy_ships = (max(REACTIVE_MIN_ENEMY_SHIPS, world.my_prod * 2)
                           if K3_ENABLED else REACTIVE_MIN_ENEMY_SHIPS)
        for enemy_p in world.enemy_planets:
            e_ships = int(enemy_p.ships)
            if e_ships < min_enemy_ships:
                continue
            # V13.3 S2 (4P-only): if the straight-line path from enemy to
            # target is sun-blocked, enemy can't launch at target this turn.
            # Skip projecting a snipe from this enemy. 4P-gated: ungated
            # showed -4.8pp 2P (the single enemy in 2P rotates out of shadow
            # quickly so the filter is over-lenient).
            if SUN_SHADOW_REACTIVE_FILTER and not world.is_2p and segment_hits_sun(
                enemy_p.x, enemy_p.y, target.x, target.y
            ):
                continue
            d = dist(enemy_p.x, enemy_p.y, target.x, target.y)
            # K4-A: per-enemy reactive emit. The base fraction is the tuned
            # REACTIVE_EMIT_FRAC (≈0.50). When THIS enemy has heavily committed
            # ships (in flight against us OR against other enemies), they have
            # less home capacity to react — use commit_ratio to scale down the
            # emit. Realistic, not pessimistic: idle enemy keeps full fraction;
            # only committed ones get the discount. Bounded so we never go
            # higher than the original tuned value (no risk added).
            if K3_ENABLED:
                e_owner = int(enemy_p.owner)
                e_strength = max(1, world.owner_strength.get(e_owner, 1))
                e_committed = (
                    world.enemy_inflight_to_us.get(e_owner, 0)
                    + world.enemy_inflight_at_other_enemies.get(e_owner, 0)
                )
                e_commit_ratio = min(1.0, e_committed / e_strength)
                # Linear scale-down: at commit_ratio=0, full REACTIVE_EMIT_FRAC.
                # At commit_ratio=1.0, halved (every emit-able ship is already
                # in flight elsewhere). Lower bound 25% so we don't fully
                # discount any reactive — they still have production.
                emit_frac = max(0.25, REACTIVE_EMIT_FRAC * (1.0 - 0.5 * e_commit_ratio))
            else:
                emit_frac = REACTIVE_EMIT_FRAC
            projected_force = max(REACTIVE_MIN_PROJECTED, int(e_ships * emit_frac))
            speed = fleet_speed(projected_force)
            travel = max(1, int(math.ceil(d / speed)))
            # Enemy launches now (turn 0 relative); arrives at travel turns.
            snipe_eta = travel
            if snipe_eta <= arrival_turn:
                continue  # they'd arrive before us — different scenario
            if snipe_eta - arrival_turn > ANTI_SNIPE_HORIZON:
                continue
            enemy_after.append((snipe_eta, projected_force))

    if not enemy_after:
        return True

    # V13.3 N6: effective_garrison projection for pre_garrison
    if N6_USE_EFFECTIVE_PRE_GARRISON:
        _, pre_garrison = effective_garrison_at_arrival(target, arrival_turn, world)
    else:
        pre_garrison = garrison_at_arrival(target, arrival_turn)
    if ships_sent <= pre_garrison:
        return True
    surplus = ships_sent - pre_garrison
    prod = max(0, int(target.production))
    by_turn = defaultdict(int)
    for eta, ships in enemy_after:
        by_turn[eta] -= ships
    for eta, ships in friendly_after:
        by_turn[eta] += ships

    bal = surplus
    last_t = arrival_turn
    for eta in sorted(by_turn):
        bal += prod * (eta - last_t)
        bal += by_turn[eta]
        if bal <= 0:
            return False
        last_t = eta
    return True


def _tiebreak_hash(world, src_id, target_id):
    """Deterministic, replayable hash for breaking near-equal-distance ties.
    Salts on (player, step, src, target) so different turns / sources don't
    produce identical perturbations. Multiplicative mix instead of Python's
    hash() because PYTHONHASHSEED randomizes hash() across processes."""
    h = (int(world.player) * 2654435761) & 0xFFFFFFFF
    h ^= (int(world.step) * 1664525) & 0xFFFFFFFF
    h ^= (int(src_id) * 16777619) & 0xFFFFFFFF
    h ^= (int(target_id) * 2246822519) & 0xFFFFFFFF
    return h & 0xFFFF


def _nearest_targets(src, world, K, max_travel, target_locked):
    """Top-K nearest non-friendly, non-comet planets, plus any race-winnable
    contested neutrals appended at the FRONT regardless of K (V12.1a).

    Final travel-time and capture cost happen inside plan_solo_capture; the
    race-loss skip in handle_expand vetoes any target where we'd arrive after
    the enemy.

    V12.3c5 (2.5): in 2P, near-equal-distance candidates (within
    TIEBREAK_EPS_FRAC of best) are reordered by a deterministic
    (player, step, src, target) hash. Cracks symmetric-Nash mirror lock
    where two PATIENT bots otherwise pick the same target deterministically.
    Replayable via hash construction.
    """
    # F31: 2P production floor — skip prod=1 scraps when prod>=2 neutrals exist.
    # Applies both opening and mid-game: prod=1 at gar=1 costs same 8 ships as
    # prod=3 at gar=20, but the value difference compounds the whole game.
    _f31_has_better = (
        world.is_2p
        and EXPAND_MIN_PROD_2P >= 2
        and any(int(n.production) >= EXPAND_MIN_PROD_2P for n in world.neutral_planets
                if n.id not in target_locked)
    )
    candidates = []
    for t in world.planets:
        if t.owner == world.player:
            continue
        if t.id in target_locked:
            continue
        if not is_targetable(world, t):
            continue
        if _neutral_blocked_by_cap(world, t):
            continue
        # F31: skip prod=1 neutrals in 2P mid-game when better options exist.
        if _f31_has_better and t.owner == -1 and int(t.production) < EXPAND_MIN_PROD_2P:
            continue
        # V12.4a: keep raw-distance gate (don't exclude orbital targets that
        # are temporarily on the far side), but rank by rotation-aware
        # effective distance.
        raw = dist(src.x, src.y, t.x, t.y)
        if raw / MAX_SPEED > max_travel + 4:
            continue
        eff = _effective_target_dist(src, t, world)
        # V12.6a/b: subtract production*VALUE_WEIGHT from effective distance.
        # Format-split: 2P=4.0 (aggressive prod preference), 4P=2.0 (mild).
        weight = VALUE_WEIGHT_2P if world.is_2p else VALUE_WEIGHT_4P
        weighted = eff - max(0, int(t.production)) * weight
        # V13.3 F1b: recently-launched enemy planets are soft targets (just shed
        # ships, can't reinforce). Promote in expand ranking — same signal F1
        # uses for hammer, lower weight to avoid thrash on expand cadence.
        if F1B_EXPAND_BONUS_ENABLED and t.owner != world.player and t.owner != -1:
            if t.id in _enemy_recently_launched:
                weighted -= F1B_EXPAND_BONUS
        # K2/K4-B (4P): enemy planet currently being attacked by ANOTHER enemy
        # → its garrison will be reduced (or owner-flipped) before our arrival.
        # Bonus magnitude scales with attack-to-garrison ratio: a planet under
        # an attack equal to its garrison is far softer than one with a small
        # probe. State-derived — no fixed bonus magnitude. The threshold floor
        # is also state-derived: my_prod * 2 (small probes that won't matter).
        if ((not K2_ENEMY_VS_ENEMY_4P_ONLY or not world.is_2p)
                and t.owner != world.player and t.owner != -1):
            attackers = getattr(world, "enemy_planet_attackers", {}).get(t.id, {})
            if attackers:
                other_enemy_attack = sum(int(s) for o, s in attackers.items()
                                          if o != t.owner and o != world.player)
                min_attack = max(2, world.my_prod * 2)
                if other_enemy_attack >= min_attack:
                    # Bonus = base * min(1.5, attack / max(1, garrison)).
                    # At attack == garrison, full base. Cap at 1.5x to avoid
                    # over-promotion of already-flipped planets.
                    base = K2_ENEMY_VS_ENEMY_EXPAND_BONUS
                    ratio = other_enemy_attack / max(1, int(t.ships))
                    weighted -= base * min(1.5, ratio)
        # V13.3 SO1: static outer planets (r+radius >= ROTATION_RADIUS_LIMIT)
        # don't orbit. Permanent real estate, harder for enemy to sweep. Boost
        # their attractiveness (subtract from weighted distance).
        if SO1_STATIC_PREFERENCE_ENABLED:
            init_t = world.initial_by_id.get(t.id)
            if init_t is not None:
                r_t = dist(init_t.x, init_t.y, CENTER_X, CENTER_Y)
                if r_t + init_t.radius >= ROTATION_LIMIT:
                    weighted -= SO1_STATIC_BONUS
        # V12.8s: 4P leader-bashing — bonus for leader's planets when we're
        # behind. Mid-game onward.
        if (
            LEADER_BASH_ENABLED
            and not world.is_2p
            and world.contest_leader
            and world.step >= LEADER_BASH_MIN_STEP
            and world.leader_id is not None
            and t.owner == world.leader_id
        ):
            weighted -= LEADER_BASH_BONUS
        # V12.8et: punish-the-expander via opponent profile. If target's owner
        # has been emitting >40% of total ships as fleets (sustained over the
        # rolling window), they're committed to attacks elsewhere — their
        # planets are weakly defended. Apply small bonus to make them more
        # attractive targets. Requires >= 5 samples to avoid early-game noise.
        if (
            not world.is_2p
            and t.owner != -1
            and t.owner != world.player
            and world.opp_profile
            and t.owner in world.opp_profile
        ):
            prof = world.opp_profile[t.owner]
            if len(prof["emit"]) >= 5:
                avg_emit = sum(prof["emit"]) / len(prof["emit"])
                if avg_emit > 0.35:
                    weighted -= 5.0  # V12.8ev': was 4.0
        # V12.8d: 4P-only weakest-enemy bonus / don't-finish penalty. Active
        # only mid-game onward and in PRESSURE mode — opening / patient
        # neutral expansion shouldn't be disrupted.
        if (
            WEAKEST_TARGET_ENABLED
            and not world.is_2p
            and world.step >= WEAKEST_TARGET_MIN_STEP
            and world.mode == "pressure"
            and world.weakest_enemy is not None
            and t.owner == world.weakest_enemy
        ):
            if world.weakest_enemy_prod_share < WEAKEST_DONT_FINISH_SHARE:
                weighted += WEAKEST_DONT_FINISH_PENALTY
            else:
                weighted -= WEAKEST_TARGET_BONUS
        # 14_4a: 2P focus-enemy bonus (single-enemy concentration).
        if (
            F14_4A_2P_FOCUS_ENABLED
            and world.is_2p
            and world.focus_enemy_2p is not None
            and t.owner == world.focus_enemy_2p
        ):
            weighted -= F14_4A_2P_FOCUS_DIST_BONUS
        candidates.append((t, weighted, raw))
    if not candidates:
        return []
    candidates.sort(key=lambda kv: kv[1])
    # V12.9 1-ply lookahead — for top-N candidates in 4P, run forward sim
    # with hypothetical capture and re-rank by score gain. Only when bonus
    # is enabled (default 0 = disabled) since it's expensive and may not
    # exceed seed noise.
    if (FWD_SIM_RANK_BONUS_4P > 0 and not world.is_2p and len(candidates) > 1):
        baseline_proj = forward_project(
            world, horizon=FWD_SIM_HORIZON,
            project_opponent_moves=True, opponent_emit_fraction=0.30
        )
        baseline_score = forward_score(baseline_proj, world.player, 4, world)
        rerank = []
        topN = min(K + 2, len(candidates))
        for idx, (t, w, raw) in enumerate(candidates[:topN]):
            est_eta = max(1, int(math.ceil(raw / MAX_SPEED)))
            est_ships = needed_to_capture(t, est_eta) + 1
            proj = forward_project(
                world, our_capture_target=t.id, our_capture_turn=est_eta,
                our_capture_ships=est_ships, horizon=FWD_SIM_HORIZON,
                project_opponent_moves=True, opponent_emit_fraction=0.30
            )
            score_gain = forward_score(proj, world.player, 4, world) - baseline_score
            adjusted = w - FWD_SIM_RANK_BONUS_4P * score_gain
            rerank.append((t, adjusted, raw))
        candidates = rerank + candidates[topN:]
        candidates.sort(key=lambda kv: kv[1])
    if world.is_2p and TIEBREAK_ENABLED and len(candidates) > 1:
        best_d = candidates[0][1]
        eps = max(TIEBREAK_EPS_MIN, TIEBREAK_EPS_FRAC * best_d)
        def _k(kv):
            tgt, weighted_d, _raw = kv
            bucket = int(weighted_d / eps) if eps > 0 else 0
            return (bucket, _tiebreak_hash(world, src.id, tgt.id), weighted_d)
        candidates.sort(key=_k)

    counter_snipe = _counter_snipe_candidates(world, src, max_travel, target_locked)

    if not RACE_ENABLED or not world.enemy_race_eta:
        head = counter_snipe + [(t, raw) for t, _eff, raw in candidates[:K]]
        return _dedupe_targets(head)

    race_priority = []
    normal = []
    for t, _eff, raw in candidates:
        enemy_eta = world.enemy_race_eta.get(t.id)
        if enemy_eta is None or t.owner != -1:
            normal.append((t, raw))
            continue
        my_min = max(1, int(math.ceil(raw / fleet_speed(max(1, int(src.ships))))))
        if my_min <= enemy_eta:
            race_priority.append((t, raw))
        else:
            normal.append((t, raw))

    return _dedupe_targets(counter_snipe + race_priority + normal[:K])


def _dedupe_targets(seq):
    """V12.4c: preserve order, drop duplicates by target id (counter-snipe and
    race-priority can overlap with the K window)."""
    seen = set()
    out = []
    for tgt, d in seq:
        if tgt.id in seen:
            continue
        seen.add(tgt.id)
        out.append((tgt, d))
    return out


def _aim_partner(world, partner, tgt, ships, max_travel):
    """Aim a coalition partner with EXACT `ships` count. Returns (angle, turns) or None."""
    if ships < COALITION_MIN_PER_CONTRIBUTOR:
        return None
    aim = aim_at_target(partner, tgt, ships, world.initial_by_id, world.ang_vel, world=world)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    return angle, turns


def _try_coalition_expand(world, src, tgt, max_travel, available, spent,
                          target_locked, moves, mode_log):
    """src can't take tgt alone; find a partner whose combined ships flip it.
    Each contributor must send >= COALITION_MIN_PER_CONTRIBUTOR (no tiny
    pieces). For tiny targets we DON'T split — the patient ethos prefers
    waiting for a solo fleet over showering a small target with two halves.
    """
    src_avail = available[src.id] - spent[src.id]
    if src_avail < COALITION_MIN_PER_CONTRIBUTOR:
        return False
    # Don't split tiny targets into two fleets; let solo handle it once one
    # source has accumulated enough.
    if int(tgt.ships) < COALITION_MIN_TARGET_SHIPS:
        return False

    # Gather partners — skip self, anyone too poor, anyone whose path is blocked.
    # (B1.5 tested gating brain-reserved-lead from coalition partners; no
    # measurable impact at n=192 — coalition draft of the lead is rare in 4P
    # self-play. Reverted.)
    partners = []
    for p in world.my_planets:
        if p.id == src.id:
            continue
        avail = available[p.id] - spent[p.id]
        if avail < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        # Estimate-only aim with full avail to filter by travel.
        est = aim_at_target(p, tgt, avail, world.initial_by_id, world.ang_vel, world=world)
        if est is None:
            continue
        _, est_turns = est
        if est_turns > max_travel:
            continue
        partners.append((est_turns, p, avail))
    if not partners:
        return False
    partners.sort(key=lambda kv: kv[0])

    # Try src + best partner.
    for est_turns, p, p_avail in partners:
        combined = src_avail + p_avail
        # Both neutrals AND enemies grow during flight — must size against the
        # arrival-time garrison, not the current snapshot. Use the slowest
        # contributor's ETA as the worst-case growth horizon. (Earlier we
        # under-sized neutral coalitions by ignoring production growth, which
        # made small fleets land and fail to capture, then we'd re-attack with
        # equally small follow-ups every cycle — exactly the "many small
        # fleets at one target" pattern the user complained about.)
        est_src = aim_at_target(src, tgt, src_avail, world.initial_by_id, world.ang_vel, world=world)
        if est_src is None:
            continue
        worst = max(est_src[1], est_turns)
        total_needed = needed_to_capture(tgt, worst)
        if combined < total_needed:
            continue

        # Proportional split, but each ≥ COALITION_MIN_PER_CONTRIBUTOR.
        ratio = src_avail / float(combined)
        s_src = max(COALITION_MIN_PER_CONTRIBUTOR,
                    min(src_avail, int(round(total_needed * ratio))))
        s_p = max(COALITION_MIN_PER_CONTRIBUTOR,
                  min(p_avail, total_needed - s_src))
        # If sum fell short, bump one side that has headroom.
        while s_src + s_p < total_needed:
            if s_src < src_avail:
                s_src += 1
            elif s_p < p_avail:
                s_p += 1
            else:
                break
        if s_src + s_p < total_needed:
            continue
        if s_src < COALITION_MIN_PER_CONTRIBUTOR or s_p < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        if s_src > src_avail or s_p > p_avail:
            continue

        # Re-aim each contributor with their EXACT share (speed differs per ship count).
        aim_src = aim_at_target(src, tgt, s_src, world.initial_by_id, world.ang_vel, world=world)
        aim_p = aim_at_target(p, tgt, s_p, world.initial_by_id, world.ang_vel, world=world)
        if aim_src is None or aim_p is None:
            continue
        a_src, t_src = aim_src
        a_p, t_p = aim_p
        if t_src > max_travel or t_p > max_travel:
            continue

        # V12.7h: 2P-gated synchronized-arrival check for coalition (research §1.10).
        # If contributors land >1 turn apart, the faster one fights the defender
        # solo (and loses to a still-strong defender + production growth in
        # the gap), then the slower fleet arrives at a partner-or-enemy planet
        # that has been fed by the failed first assault. Abort the coalition
        # rather than fire two desynchronized fleets — the next turn we'll
        # try again with potentially better ship counts. 12_7g (ungated)
        # measured +48 wins 2P but -22pp 4P vs main: 4P relies on coalitions
        # to hit targets that K=1 expand can't solo, and aborting too many
        # cedes those targets entirely. 2P benefits from the strict gate.
        if world.is_2p and abs(t_src - t_p) > 1:
            continue

        # Re-validate against the true post-re-aim ETA. Re-aiming with smaller
        # ship counts can lengthen flight (slower fleets), and a longer flight
        # means a bigger garrison at arrival.
        post_eta = max(t_src, t_p)
        post_needed = needed_to_capture(tgt, post_eta)
        if s_src + s_p < post_needed:
            continue

        _commit_fleet(world, moves, spent, target_locked,
                      src.id, tgt.id, a_src, t_src, int(s_src))
        _commit_fleet(world, moves, spent, target_locked,
                      p.id, tgt.id, a_p, t_p, int(s_p))
        mode_log[src.id] = "expand-coalition"
        mode_log[p.id] = "expand-coalition"
        return True

    return False


# ============================================================
# Mode 4 — Hammer (persistent coordinated strike)
# ============================================================

def _routine_avail(world, planet, base_avail):
    """V14.1d iter g: production-tier reserve. Subtract a fraction of high-prod
    planet garrison from routine expand/hammer spending. The reserve grows
    naturally via production and is available to mega-hammer.
    """
    if not PROD_RESERVE_ENABLED:
        return base_avail
    if PROD_RESERVE_4P_ONLY and world.is_2p:
        return base_avail
    if world.step < PROD_RESERVE_TURN_MIN:
        return base_avail
    if int(planet.production) < PROD_RESERVE_MIN_PROD:
        return base_avail
    reserve = int(int(planet.ships) * PROD_RESERVE_FRAC)
    return max(0, base_avail - reserve)


def _brain_pick_lead(world, available, spent, mode_log, min_ships=None):
    """Shared lead-picker used by both _brain_reserve_lead (pre-pass) and
    handle_accumulator (post-defense). Returns Planet or None.

    Identical logic to handle_accumulator's original lead-selection so the
    reservation and the actual feeder-target agree. min_ships defaults to
    the accumulator's threshold; the brain pre-pass passes a higher value.

    B3b: when BRAIN_LEAD_PREFER_FRONTIER, score = avail - frontier_dist*weight
    so a frontier planet beats a deep-back-corner one even if the back has
    slightly more ships — a closer lead delivers strikes faster.
    """
    if min_ships is None:
        min_ships = ACCUMULATOR_LEAD_MIN_SHIPS
    enemies = world.enemy_planets
    candidates = []
    for p in world.my_planets:
        status = mode_log.get(p.id)
        # Accept reserved-lead sentinel (we set it ourselves); skip everything else.
        if status and status != "brain-reserved-lead":
            continue
        avail = available[p.id] - spent[p.id]
        if avail < min_ships:
            continue
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat >= avail * ACCUMULATOR_LEAD_THREAT_RATIO:
            continue
        if BRAIN_LEAD_PREFER_FRONTIER and enemies:
            frontier_dist = min(dist(p.x, p.y, e.x, e.y) for e in enemies)
            score = float(avail) - frontier_dist * BRAIN_LEAD_FRONTIER_WEIGHT
        else:
            score = float(avail)
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def _brain_reserve_lead(world, available, spent, mode_log):
    """B1 (one-brain pre-pass): mark the future accumulator-lead with a
    sentinel so handle_expand can't drain it into small-ship pickups before
    handle_accumulator / handle_mega_hammer run.

    Runs after defense (which doesn't gate on mode_log status of sources)
    and before the expand → accumulator → mega-hammer chain. If defense
    later commits the same planet, defense overwrites mode_log[p.id] = 'defense'
    and the chain naturally skips it — life beats lead."""
    if not BRAIN_LEAD_RESERVE_ENABLED:
        return
    if not ACCUMULATOR_ENABLED:
        return
    if BRAIN_LEAD_RESERVE_4P_ONLY and world.is_2p:
        return
    if ACCUMULATOR_4P_ONLY and world.is_2p:
        return
    if world.step < ACCUMULATOR_TURN_MIN:
        return
    lead = _brain_pick_lead(world, available, spent, mode_log,
                            min_ships=BRAIN_LEAD_RESERVE_MIN_SHIPS)
    if lead is None:
        return
    # B1 iter 3: only reserve if the lead has a real mega-hammer target
    # within reach. Otherwise reservation just freezes a productive planet.
    if BRAIN_LEAD_RESERVE_REQUIRE_TARGET:
        has_target = False
        for tgt in world.enemy_planets:
            if int(tgt.ships) > MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H:
                continue
            aim = aim_at_target(lead, tgt, available[lead.id] - spent[lead.id],
                                world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            _, turns = aim
            if turns > MEGA_HAMMER_MAX_TRAVEL:
                continue
            has_target = True
            break
        if not has_target:
            return
    mode_log[lead.id] = "brain-reserved-lead"


def handle_accumulator(world, available, spent, target_locked, moves, mode_log):
    """V14.2 (Phase 3.7, Idea 6c): accumulator — feed surplus from safe
    backline planets to the lead stockpile each turn.

    Engine: fleet speed = 1 + 5×(log(ships)/log(1000))^1.5. One big fleet
    (1000 ships, speed 6) arrives faster AND survives tied-combat better
    than 4 fleets of 250 ships. Concentration > spread.

    Strategy: each turn, identify our planet with the most ships ("lead").
    For other planets in the safe backline (no incoming enemy threat AND
    surplus above reserve), send their surplus TO the lead. Over multiple
    turns, the lead accumulates a massive stockpile and handle_mega_hammer
    fires it as one overwhelming strike.

    Runs BEFORE handle_mega_hammer so accumulated ships are visible to
    mega-hammer this turn (but in-flight feeds arrive on later turns).
    """
    if not ACCUMULATOR_ENABLED:
        return
    if ACCUMULATOR_4P_ONLY and world.is_2p:
        return
    if world.step < ACCUMULATOR_TURN_MIN:
        return

    # Lead = the my-planet with the most available ships that's not under
    # immediate threat. The lead should be near enemy frontier to launch
    # effective hammers; use simple "most ships" heuristic for now.
    # B1: a planet pre-marked "brain-reserved-lead" is eligible (it IS the
    # planet we want).
    lead_candidates = []
    for p in world.my_planets:
        status = mode_log.get(p.id)
        if status and status != "brain-reserved-lead":
            continue
        avail = available[p.id] - spent[p.id]
        if avail < ACCUMULATOR_LEAD_MIN_SHIPS:
            continue
        # Skip if under immediate threat
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat >= avail * ACCUMULATOR_LEAD_THREAT_RATIO:
            continue
        lead_candidates.append((avail, p))
    if not lead_candidates:
        return
    lead_candidates.sort(key=lambda x: -x[0])
    lead_avail, lead = lead_candidates[0]

    # Feeders: planets with surplus, NOT under threat, that can reach the lead.
    feeders = []
    for p in world.my_planets:
        if p.id == lead.id or p.id in mode_log:
            continue
        # Skip if under threat
        threat = sum(int(ships) for eta, owner, ships
                     in world.arrivals_by_planet.get(p.id, [])
                     if owner != world.player and owner != -1)
        if threat > 0:
            continue
        avail = available[p.id] - spent[p.id]
        surplus = avail - ACCUMULATOR_FEEDER_KEEP_RESERVE
        if surplus < ACCUMULATOR_FEEDER_MIN_SURPLUS:
            continue
        # Reachability
        aim = aim_at_target(p, lead, surplus, world.initial_by_id,
                            world.ang_vel, world=world)
        if aim is None:
            continue
        angle, turns = aim
        if turns > ACCUMULATOR_FEEDER_MAX_TRAVEL:
            continue
        feeders.append((turns, surplus, p, angle))

    if not feeders:
        return
    # Sort by closest, then largest surplus
    feeders.sort(key=lambda x: (x[0], -x[1]))
    fed_count = 0
    for turns, surplus, src, angle in feeders:
        if fed_count >= ACCUMULATOR_MAX_FEEDS_PER_TURN:
            break
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, lead.id, angle, turns, int(surplus))
        mode_log[src.id] = "accumulator-feeder"
        fed_count += 1
    if fed_count > 0:
        # Don't overwrite mode_log for lead if it already has a status
        if lead.id not in mode_log:
            mode_log[lead.id] = "accumulator-lead"


def handle_mega_hammer(world, available, spent, target_locked, moves, mode_log):
    """V14.1c (Phase 3.3): single-source overwhelming strike.

    For each of our planets with avail >= MEGA_HAMMER_SHIPS_MIN, find an
    enemy target whose garrison (after projected arrivals) is <=
    MEGA_HAMMER_TARGET_GARRISON_MAX and is within MEGA_HAMMER_MAX_TRAVEL
    turns. Launch the ENTIRE garrison as a single huge fleet — exploits
    the fleet-speed log formula (bigger = faster) and overwhelms
    reactive defense.

    Runs BEFORE handle_hammer so a successful mega strike isn't dissolved
    into the multi-stockpile coalition logic.
    """
    if not MEGA_HAMMER_ENABLED:
        return
    if MEGA_HAMMER_4P_ONLY and world.is_2p:
        return
    # Sort sources by available ships descending — biggest stockpiles fire first.
    # (B3a tested global target-ranking, regressed -1.5pp 4P / -13pp 2P noise; reverted.)
    sources = sorted(world.my_planets,
                     key=lambda p: -(available[p.id] - spent[p.id]))
    fired_targets = set()
    fired_count = 0
    for src in sources:
        # V14.2 Idea 6b: concentration cap.
        if MEGA_HAMMER_CONCENTRATE_ENABLED and fired_count >= MEGA_HAMMER_MAX_PER_TURN:
            break
        avail = available[src.id] - spent[src.id]
        # V14.1d iter h: per-production threshold. High-prod planets fire sooner.
        # V14.2 Idea 6a: fresh-capture inheritance — newly-captured planets
        # use a lower threshold so they chain attacks without accumulating.
        prod = int(src.production)
        if FRESH_CAPTURE_INHERITANCE_ENABLED and src.id in _planet_capture_age:
            threshold = MEGA_HAMMER_SHIPS_MIN_FRESH
        else:
            threshold = MEGA_HAMMER_THRESHOLD_BY_PROD.get(prod, MEGA_HAMMER_SHIPS_MIN)
        if avail < threshold:
            continue  # this source can't fire, but others may (different prod)
        # Already used in defense / absorb / etc.
        # B1: brain-reserved-lead is eligible (it's the planet we WANT to fire).
        status = mode_log.get(src.id)
        if status and status not in ("cheap-pickup", "brain-reserved-lead"):
            continue
        # Find best enemy target
        best = None
        for tgt in world.enemy_planets:
            if tgt.id in target_locked or tgt.id in fired_targets:
                continue
            if int(tgt.ships) > MEGA_HAMMER_TARGET_GARRISON_MAX_ITER_H:
                continue
            aim = aim_at_target(src, tgt, avail, world.initial_by_id,
                                world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > MEGA_HAMMER_MAX_TRAVEL:
                continue
            # Rank: prefer higher-production targets within reach
            # 14_4a: 2P focus bonus — prefer focus enemy's planets for mega-hammer.
            focus_bonus = 0
            if (F14_4A_2P_FOCUS_ENABLED and world.is_2p
                    and getattr(world, "focus_enemy_2p", None) is not None
                    and tgt.owner == world.focus_enemy_2p):
                focus_bonus = F14_4A_2P_FOCUS_MEGA_BONUS
            score = (int(tgt.production) + focus_bonus, -int(turns))
            if best is None or score > best[0]:
                best = (score, tgt, angle, turns)
        if best is None:
            continue
        _, tgt, angle, turns = best
        # B2: forward-sim verify (mirror HAMMER_MELIS_VERIFY). Mega-hammer
        # was firing without checking we hold the planet post-capture; a
        # 250-ship strike that flips back next turn is pure waste. Project
        # the strike forward and bail if end_owner != us.
        if MEGA_HAMMER_MELIS_VERIFY and turns > 0:
            proj = forward_project(
                world,
                our_capture_target=tgt.id,
                our_capture_turn=int(turns),
                our_capture_ships=int(avail),
                horizon=FWD_SIM_HORIZON + int(turns),
                project_opponent_moves=True,
                opponent_emit_fraction=MEGA_HAMMER_VERIFY_OPP_EMIT,
            )
            end_owner, _ = proj.get(tgt.id, (-1, 0))
            if end_owner != world.player:
                continue
        _commit_fleet(world, moves, spent, target_locked,
                      src.id, tgt.id, angle, turns, int(avail))
        mode_log[src.id] = "mega-hammer-launched"
        mode_log[tgt.id] = "mega-hammer-target"
        fired_targets.add(tgt.id)
        fired_count += 1


def handle_eliminate_focus(world, available, spent, target_locked, moves, mode_log):
    """K-mode FAST-KILL handler. Pick the focus enemy and explicitly try to
    capture their planets, sorted by easiest-to-take. Runs every turn after
    expand. For each focus-enemy planet (cheapest first), finds closest
    source that can solo-capture it and fires. Caps fires per turn.

    This is the load-bearing piece of main_k vs main: explicit elimination
    pressure that doesn't rely on existing scoring bonuses to align.
    """
    if not K_ELIMINATE_ENABLED:
        return
    # 2P is already covered by 14_4c's focus_enemy_2p expand/hammer/mega bias.
    if world.is_2p:
        return
    # Math turn-gate: skip while still in opening (state-derived flag).
    # No fixed turn — just "are we past opening expansion?".
    if world.is_opening:
        return
    # K3-B: dropped the hardcoded `my_planets < 3` floor. The per-source
    # `avail < required` check inside the loop is the only gate that needs
    # to be there — when we have no surplus, no source will pass the math,
    # and the handler harmlessly returns. When we DO have surplus on even
    # 1-2 planets, eliminate should fire (it's our path back).
    if not world.my_planets:
        return
    focus_id = getattr(world, "focus_enemy_4p", None)
    if focus_id is None:
        return
    focus_planets = [p for p in world.enemy_planets if p.owner == focus_id]
    if not focus_planets:
        return
    # K-mode 4P math: rank focus planets by EFFECTIVE garrison (subtract ships
    # already incoming from OTHER enemies — they'll do the work for us).
    # Then by production (higher prod = bigger denial value).
    def rank_key(p):
        attackers = world.enemy_planet_attackers.get(p.id, {})
        other_enemy_ships = sum(amt for owner, amt in attackers.items()
                                 if owner != focus_id and owner != world.player)
        effective_garrison = max(0, int(p.ships) - int(other_enemy_ships * 0.5))
        return (effective_garrison, -int(p.production))
    focus_planets.sort(key=rank_key)

    # Pre-compute the focus enemy's OWN planet ship list for reactive math.
    focus_other_planets = list(focus_planets)

    # Math: cap fires by available idle-source count. Preserve some sources
    # for next-turn flexibility — use ~1/3 of idle sources, min 1.
    # K3-C: exposure-aware bump. When the focus enemy is heavily committed
    # (ships in flight + incoming attacks from others vs their own strength),
    # their home defense is THIN — exploit the open window. Smooth curve:
    # at exposure=0  → max_fires = idle/3 (preserve flexibility)
    # at exposure=1  → max_fires = idle   (all-out while target is exposed)
    # State-derived; no magic threshold.
    idle_sources = sum(1 for p in world.my_planets
                       if mode_log.get(p.id) is None)
    base_fires = max(1, idle_sources // 3)
    if K3_ENABLED:
        focus_strength = max(1, world.owner_strength.get(focus_id, 1))
        focus_committed = (
            world.enemy_inflight_total.get(focus_id, 0)
            + world.enemy_under_attack_by_others.get(focus_id, 0)
        )
        exposure = min(1.0, max(0.0, focus_committed / focus_strength))
        # Linear interpolation: base_share (1/3) → full_share (1.0)
        share = (1.0 / 3.0) + exposure * (2.0 / 3.0)
        max_fires = max(base_fires, int(round(idle_sources * share)))
    else:
        max_fires = base_fires

    fires = 0
    for tgt in focus_planets:
        if fires >= max_fires:
            break
        if tgt.id in target_locked:
            continue
        # Find best source: prefer FRESH-captured planets first (kill cascade —
        # newly captured planets sit near focus_enemy's other planets so they
        # are the natural launchpad for the next strike), then by distance.
        # K2: de-rank defense-reserved sources (closest to a threatened friendly
        # planet) so eliminate uses them only when no other option exists.
        reserved = getattr(world, "defense_reserve_sources", set())
        def src_key(p):
            reserve_pen = (1 if (K2_ELIMINATE_DERANK_RESERVED_SOURCES
                                 and p.id in reserved) else 0)
            fresh = 0 if p.id in _planet_capture_age else 1
            return (reserve_pen, fresh, dist(p.x, p.y, tgt.x, tgt.y))
        srcs_by_dist = sorted(world.my_planets, key=src_key)
        fired = False
        for src in srcs_by_dist:
            # Only use genuinely idle sources (no mode_log entry).
            status = mode_log.get(src.id)
            if status is not None:
                continue
            avail = available[src.id] - spent[src.id]
            # No fixed minimum — the math check below (required ships) is
            # the only gate. If source has fewer than required, skip.
            if avail < MIN_DISPATCH_SHIPS:
                continue
            # Aim
            aim = aim_at_target(src, tgt, avail, world.initial_by_id,
                                 world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > K_ELIMINATE_MAX_TRAVEL:
                continue
            # MATH: ships needed = projected garrison + FOCUS ENEMY's reactive
            # emit (only counting their OTHER planets that can reach target in
            # time, NOT other enemies' planets). Plus 15% overkill buffer.
            try:
                proj_owner, proj_garrison = effective_garrison_at_arrival(tgt, turns, world)
            except Exception:
                proj_owner, proj_garrison = int(tgt.owner), int(tgt.ships) + int(tgt.production) * turns
            if proj_owner == world.player:
                continue
            # Math-aware emit fraction: when focus enemy has heavily committed
            # ships at us, they have LESS home reactive capacity. Use their
            # commitment ratio to scale down expected reactive emit.
            # emit_frac = 0.5 - commitment_ratio * 0.4, bounded [0.25, 0.65].
            # K2: include ships ALSO committed against other enemies — those
            # ships are equally unavailable for reactive defense of this target.
            focus_strength = max(1, world.owner_strength.get(focus_id, 1))
            if K2_ELIMINATE_REACTIVE_TIGHTER:
                already_committed = (
                    world.enemy_inflight_to_us.get(focus_id, 0)
                    + world.enemy_inflight_at_other_enemies.get(focus_id, 0)
                )
            else:
                already_committed = world.enemy_inflight_to_us.get(focus_id, 0)
            commit_ratio = already_committed / focus_strength
            emit_frac = max(0.25, min(0.65, 0.5 - commit_ratio * 0.4))

            reactive = 0
            for ep in focus_other_planets:
                if ep.id == tgt.id:
                    continue
                d = dist(ep.x, ep.y, tgt.x, tgt.y)
                eta_from_ep = max(1, int(math.ceil(d / 3.5)))
                if eta_from_ep <= turns + 3:
                    # K2: ep's ships that are being attacked by OTHER enemies
                    # will be needed for ep's own defense, not for reactive
                    # emit at our target. Subtract half the other-enemy attack
                    # from ep's effective ship pool. Realistic: ep won't burn
                    # 100% of its garrison defending, but won't emit 100% either.
                    ep_pool = int(ep.ships)
                    if K2_ELIMINATE_REACTIVE_TIGHTER:
                        ep_attackers = world.enemy_planet_attackers.get(ep.id, {})
                        other_attack = sum(int(s) for o, s in ep_attackers.items()
                                            if o != focus_id and o != world.player)
                        ep_pool = max(0, ep_pool - other_attack // 2)
                    reactive += int(ep_pool * emit_frac)
            required = int(math.ceil((proj_garrison + reactive) * 1.15)) + 1
            if avail < required:
                continue  # not enough force — would lose strike
            # Re-aim with exact ship count for accurate angle/turns.
            re_aim = aim_at_target(src, tgt, required, world.initial_by_id,
                                    world.ang_vel, world=world)
            if re_aim is None:
                continue
            r_angle, r_turns = re_aim
            if r_turns > K_ELIMINATE_MAX_TRAVEL:
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, tgt.id, r_angle, r_turns, int(required))
            mode_log[src.id] = "eliminate-focus"
            mode_log[tgt.id] = "eliminate-focus-target"
            fires += 1
            fired = True
            break
        if not fired:
            continue


def handle_hammer(world, available, spent, target_locked, moves, mode_log):
    """One persistent plan at a time. Plan picks a strong-production enemy
    target and a set of stockpiles whose combined fleet arriving simultaneously
    beats defender_at_arrival × overkill. Launches stagger so all fleets land
    on the same turn. Plan aborts if defender reinforces past committed strength.
    """
    global _hammer_plan
    if not HAMMER_ENABLED:
        return
    if not world.enemy_planets:
        _hammer_plan = None
        return

    if _hammer_plan is not None:
        # Validate ownership of the target and the participants are still ours.
        target = world.planet_by_id.get(_hammer_plan["target_id"])
        if target is None or target.owner == world.player:
            _hammer_plan = None
        else:
            # Recheck defender-at-arrival isn't beyond our committed strength.
            arrival_rel = _hammer_plan["target_arrival_abs"] - world.step
            if arrival_rel <= 0:
                _hammer_plan = None
            else:
                d_owner, d_ships = predict_defender_at_arrival(world, target, arrival_rel)
                if d_ships > _hammer_plan["committed_strength"] / HAMMER_ABORT_OVERRUN_RATIO:
                    _hammer_plan = None

    if _hammer_plan is None:
        # Decide whether to fire a new plan this turn.
        if not _hammer_should_fire(world):
            return
        plan = _build_hammer_plan(world, available, spent)
        if plan is None:
            return
        # V12.9: Melis verification of hammer plan. Project the combined
        # arrival as a single fleet at target_arrival_abs - world.step turns.
        # Skip the hammer if projection shows we don't end up owning target.
        if HAMMER_MELIS_VERIFY:
            target = world.planet_by_id.get(plan["target_id"])
            if target is not None:
                arrival_rel = plan["target_arrival_abs"] - world.step
                if arrival_rel > 0:
                    proj = forward_project(
                        world,
                        our_capture_target=plan["target_id"],
                        our_capture_turn=int(arrival_rel),
                        our_capture_ships=int(plan["committed_strength"]),
                        horizon=FWD_SIM_HORIZON + arrival_rel,
                        project_opponent_moves=True,
                        opponent_emit_fraction=0.30,
                    )
                    end_owner, _ = proj.get(plan["target_id"], (-1, 0))
                    if end_owner != world.player:
                        return  # don't commit hammer
        _hammer_plan = plan

    # Execute: any participant whose fire_turn_abs == world.step launches now.
    plan = _hammer_plan
    completed_launches = []
    for src_id, launch in list(plan["launches"].items()):
        if launch.get("fired"):
            continue
        if launch["fire_turn_abs"] > world.step:
            continue  # delay
        src = world.planet_by_id.get(src_id)
        if src is None or src.owner != world.player:
            completed_launches.append(src_id)
            continue
        ships = launch["ships"]
        if ships < HAMMER_MIN_PER_CONTRIBUTOR:
            completed_launches.append(src_id)
            continue
        avail = available[src_id] - spent[src_id]
        if avail < ships:
            completed_launches.append(src_id)
            continue
        target = world.planet_by_id[plan["target_id"]]
        # Re-aim with EXACT ship count; speed depends on log(ships).
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            completed_launches.append(src_id)
            continue
        angle, turns = aim
        _commit_fleet(world, moves, spent, target_locked,
                      src_id, plan["target_id"], angle, turns, int(ships))
        mode_log[src_id] = "hammer"
        launch["fired"] = True

    # Cleanup: drop fired-or-failed launches; abort plan if no launches remain.
    for sid in completed_launches:
        plan["launches"].pop(sid, None)
    if not plan["launches"] or all(l.get("fired") for l in plan["launches"].values()):
        _hammer_plan = None


def _hammer_should_fire(world):
    """Trigger condition: my prod share >= mode-specific threshold AND a strong
    enemy production target is reachable, OR we're in late-flush mode."""
    if world.is_late:
        return True
    threshold = world.mode_params["hammer_prod_share"]
    if world.my_prod_share < threshold:
        return False
    return True


def _build_hammer_plan(world, available, spent):
    """Pick best target + stockpile set. Stockpiles are planets with ships >= MIN
    or promoted-by-idle. Combined arrival fleet must beat defender × overkill.
    Returns plan dict or None."""
    # V12.3b (2.3): mode-aware stockpile floor. 2P duels churn ships through
    # expansion and rarely accumulate 50; lowered floor lets the lowered
    # prod-share trigger and lowered overkill ratio actually fire.
    stockpile_min = world.mode_params.get("hammer_stockpile_min", HAMMER_STOCKPILE_MIN)
    stockpiles = []
    for p in world.my_planets:
        # V14.1d iter g: hide production-tier reserve from routine hammer
        avail = _routine_avail(world, p, available[p.id] - spent[p.id])
        if avail < HAMMER_MIN_PER_CONTRIBUTOR:
            continue
        promoted = p.id in _promoted_stockpiles
        if avail < stockpile_min and not promoted:
            continue
        stockpiles.append((p, avail))
    if not stockpiles:
        return None

    overkill = LATE_FLUSH_OVERKILL_RATIO if world.is_late else world.mode_params["hammer_overkill"]

    targets = [
        p for p in world.enemy_planets
        if is_targetable(world, p) and p.production >= HAMMER_TARGET_PROD_MIN
    ]
    if not targets:
        if world.is_late:
            targets = [p for p in world.enemy_planets if is_targetable(world, p)]
        if not targets:
            return None

    best = None
    for tgt in targets:
        # Compute travel time per stockpile.
        per_src = []
        for src, avail in stockpiles:
            aim = aim_at_target(src, tgt, max(1, avail), world.initial_by_id, world.ang_vel, world=world)
            if aim is None:
                continue
            angle, turns = aim
            if turns > HAMMER_MAX_TRAVEL:
                continue
            per_src.append((turns, src, avail, angle))
        if not per_src:
            continue
        # Common arrival = max of participant travels (closer ones delay).
        per_src.sort()  # closest first
        target_arrival = per_src[-1][0]
        d_owner, d_ships = predict_defender_at_arrival(world, tgt, target_arrival)
        if d_owner == world.player:
            continue
        # K-mode math-aware per-target overkill: scale by target-owner's
        # reactive capacity. If the target's owner has more reachable ships
        # at other planets, expect heavier counter-attack — use higher overkill.
        # Math: dynamic_overkill = base_overkill + reactive_ratio * 0.3
        target_owner_reactive = 0
        if d_owner != -1 and d_owner != world.player:
            for ep in world.planets:
                if ep.owner != d_owner or ep.id == tgt.id:
                    continue
                d = dist(ep.x, ep.y, tgt.x, tgt.y)
                eta_from_ep = max(1, int(math.ceil(d / 3.5)))
                if eta_from_ep <= target_arrival + 3:
                    target_owner_reactive += int(int(ep.ships) * 0.4)
        reactive_ratio = target_owner_reactive / max(1, d_ships)
        dynamic_overkill = overkill + min(0.3, reactive_ratio * 0.3)
        required = int(math.ceil(d_ships * dynamic_overkill)) + 1

        # Greedily add participants until we cover required.
        accum = 0
        chosen = []
        for turns, src, avail, angle in per_src:
            chosen.append((turns, src, avail, angle))
            accum += avail
            if accum >= required:
                break
        if accum < required:
            continue

        # Trim last contributor to exact need (avoid blowing entire stockpile).
        # If trimming would push contributor below the per-contributor floor,
        # drop them entirely instead — better one fewer fleet than a tiny one.
        # V14.2 (Phase 3.8): when last source has NO incoming enemy threat,
        # SKIP TRIM — send full avail. Bigger fleets are faster (log-speed)
        # and harder to absorb (tied-combat math). User-observed: bot was
        # sending 22 of 58 garrison; should send full when safe.
        slack = accum - required
        if slack > 0 and chosen:
            last_turn, last_src, last_avail, last_angle = chosen[-1]
            # V14.2 (Phase 3.10): in 2P, ALWAYS send full avail (regardless
            # of threat). User-observed: split-fleet pattern persisted
            # because no-threat check rarely fires in mid-game. Bigger
            # fleets always better in 2P duel — log-speed + tied-combat.
            # If oversend leaves source empty, fresh-capture inheritance
            # (Idea 6a) chains from the captured planet next turn.
            oversend_active = (
                HAMMER_NO_THREAT_OVERSEND_ENABLED
                and (not HAMMER_NO_THREAT_OVERSEND_2P_ONLY or world.is_2p)
            )
            # Compute threat once.
            last_src_threat = sum(
                int(ships) for eta, owner, ships
                in world.arrivals_by_planet.get(last_src.id, [])
                if owner != world.player and owner != -1
            )
            # V14.2 (Phase 3.10 v2): safe-surplus oversend (both formats).
            # Skip trim ONLY if source has large surplus AND threat is small.
            safe_surplus_ok = (
                HAMMER_SAFE_SURPLUS_OVERSEND_ENABLED
                and last_avail >= required * HAMMER_SAFE_SURPLUS_RATIO
                and last_src_threat <= last_avail * HAMMER_OVERSEND_MAX_THREAT_RATIO
            )
            if safe_surplus_ok:
                # Send full avail (oversend) — surplus is large enough to not strip.
                pass
            elif oversend_active and HAMMER_ALWAYS_OVERSEND_2P and world.is_2p:
                # Always-oversend in 2P (currently disabled, kept for re-test).
                pass
            elif oversend_active and last_src_threat == 0:
                # No-threat path: skip trim (original V14.2 Phase 3.8 fix).
                pass
            else:
                trimmed = last_avail - slack
                if trimmed < HAMMER_MIN_PER_CONTRIBUTOR:
                    chosen.pop()
                    if not chosen or sum(c[2] for c in chosen) < required - last_avail:
                        chosen.append((last_turn, last_src, last_avail, last_angle))
                else:
                    chosen[-1] = (last_turn, last_src, trimmed, last_angle)

        score = required - target_arrival * 0.5  # cheaper + sooner = better
        # 14_4a: 2P focus bonus — concentrate hammers on the single enemy.
        if (F14_4A_2P_FOCUS_ENABLED and world.is_2p
                and getattr(world, "focus_enemy_2p", None) is not None
                and tgt.owner == world.focus_enemy_2p):
            score += F14_4A_2P_FOCUS_HAMMER_BONUS
        # K-mode 4P: hammer bonus for focus_enemy_4p ONLY when leader-bash is
        # active (math: leader > my_strength * 2). General 4P focus regressed
        # earlier; this gated version only fires when we MUST attack the leader.
        elif (K_ELIMINATE_ENABLED and not world.is_2p
                and getattr(world, "leader_4p", None) is not None
                and tgt.owner == world.leader_4p
                and world.enemy_strength_4p.get(world.leader_4p, 0) >
                    (world.owner_strength.get(world.player, 0)
                     + world.my_prod * 8
                     + world.owner_planet_count.get(world.player, 0) * 5) * 2.0):
            score += 12.0  # mild bonus — only fires in leader-bash mode
        # V13.3 F1: bonus for recently-launched enemy planets — they just shed
        # ships and can't reinforce in time. (Higher score = preferred target.)
        if FLEET_INTENT_ENABLED and tgt.id in _enemy_recently_launched:
            score += FLEET_INTENT_HAMMER_BONUS
        # V13.3 R1: even bigger bonus for planets enemy just captured FROM US.
        # Fresh enemy capture leaves tiny garrison; recapture is cheap and dual-
        # purpose (denies enemy production + restores our position).
        if R1_RECAPTURE_PRIORITY_ENABLED and tgt.id in _freshly_lost_planets:
            score += R1_RECAPTURE_HAMMER_BONUS
        # V12.7m: 4P kingmaker — penalize the global leader's planets only when
        # I'M NOT the leader. Attacking a stronger enemy provokes retaliation
        # while a weaker enemy grows uncontested; in FFA the rational play is
        # to hit the second-place opponent. When I'M leading, attacking the
        # strongest other enemy is correct consolidation, not provocation —
        # the gate prevents the penalty from ceding free attacks (research §4).
        # 2P has no FFA dynamic, leave score untouched.
        if not world.is_2p:
            my_strength = world.owner_strength.get(world.player, 0)
            enemy_strengths = [
                (world.owner_strength[o], o)
                for o in world.owner_strength
                if o not in (-1, world.player) and world.owner_strength[o] > 0
            ]
            if enemy_strengths:
                max_enemy_strength, max_enemy_owner = max(enemy_strengths)
                if max_enemy_strength > my_strength and tgt.owner == max_enemy_owner:
                    score = score - abs(score) * 0.3
        cand = {
            "target_id": tgt.id,
            "target_arrival_abs": world.step + target_arrival,
            "committed_strength": sum(c[2] for c in chosen),
            "score": score,
            "launches": {},
        }
        for turns, src, ships, angle in chosen:
            fire_turn_rel = target_arrival - turns
            cand["launches"][src.id] = {
                "fire_turn_abs": world.step + fire_turn_rel,
                "ships": int(ships),
                "angle": float(angle),
                "fired": False,
            }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


# ============================================================
# Mode 4b - Multi-prong forcing (V12.3c1)
# ============================================================

def handle_multiprong(world, available, spent, target_locked, moves, mode_log):
    """If a hammer is committed at target T and a credible enemy reinforcer E
    is pumping ships into T, open a same-turn second prong at E using surplus
    ships. Strict credibility gates: 2P only, real-reinforcement gate, post-
    launch garrison gate, prong-credibility gate.

    The picture-1 failure: bot fed all output into one stream against an
    actively-reinforced target. Two prongs force the opponent to choose:
    defend T -> we take E (no more reinforcements -> hammer lands clean);
    defend E -> they pull ships off T (hammer lands clean).
    """
    if not MULTIPRONG_ENABLED:
        return
    if MULTIPRONG_2P_ONLY and not world.is_2p:
        return
    if _hammer_plan is None:
        return

    target_id = _hammer_plan.get("target_id")
    target = world.planet_by_id.get(target_id)
    if target is None or target.owner == world.player or target.owner == -1:
        return
    arrival_rel = _hammer_plan.get("target_arrival_abs", world.step) - world.step
    if arrival_rel <= 0:
        return
    committed = int(_hammer_plan.get("committed_strength", 0))
    if committed <= 0:
        return

    # Identify reinforcers: enemy planets with in-flight fleets aimed at T.
    # Sum ships per source. Skip non-enemy and -1 owners.
    reinforcer_ships = defaultdict(int)
    for f in world.fleets:
        if int(f.ships) <= 0:
            continue
        if f.owner == world.player or f.owner == -1:
            continue
        ftarget, _eta = fleet_target_planet(
            f, world.planets, world.initial_by_id, world.ang_vel
        )
        if ftarget is None or ftarget.id != target_id:
            continue
        reinforcer_ships[int(f.from_planet_id)] += int(f.ships)
    if not reinforcer_ships:
        return

    # T's defender at our hammer's arrival, factoring all in-flight fleets.
    _, defender_at_arrival = predict_defender_at_arrival(world, target, arrival_rel)
    needed_t = int(math.ceil(defender_at_arrival)) + 1
    deficit = max(0, needed_t - committed)

    # If our hammer already covers needed(T), the reinforcement isn't actually
    # decisive. We still want to consider multi-prong if reinforcement size is
    # large enough to mean opponent cares about T (signal of valuable target).
    # But scale the gate: require at least deficit OR a meaningful absolute floor.
    min_reinforce = max(1, int(math.ceil(deficit * MULTIPRONG_REINFORCER_MIN_RATIO)))

    # Find the strongest credible reinforcer.
    candidates = []
    for src_id, ship_count in reinforcer_ships.items():
        src = world.planet_by_id.get(src_id)
        if src is None:
            continue
        if src.owner == world.player or src.owner == -1:
            continue
        if ship_count < min_reinforce:
            continue
        candidates.append((src, ship_count))
    if not candidates:
        return
    # Prefer reinforcer with most ships in flight (most committed to T).
    candidates.sort(key=lambda kv: kv[1], reverse=True)

    # Try each candidate in order. Stop on first feasible second prong.
    for reinforcer, in_flight in candidates:
        if reinforcer.id in target_locked:
            continue
        if not is_targetable(world, reinforcer):
            continue
        # Build a multi-source attack on E.
        prong = _build_multiprong_attack(
            world, reinforcer, available, spent, target_locked
        )
        if prong is None:
            continue
        prong_strength, prong_arrival, prong_landings, e_at_arrival = prong

        # Credibility gate: post-launch home garrison < what we land with.
        # We use predict_defender_at_arrival output (e_at_arrival) which already
        # accounts for in-flight fleets aimed at E, so this is the more honest
        # check than reinforcer.ships - in_flight.
        if prong_strength <= e_at_arrival * MULTIPRONG_E_OVERKILL:
            continue
        # Prong-credibility: total committed offense >= needed(T) + needed(E)*0.6.
        needed_e = int(math.ceil(e_at_arrival)) + 1
        if committed + prong_strength < needed_t + int(round(needed_e * MULTIPRONG_CREDIBILITY_FACTOR)):
            continue

        # Commit each landing in the prong.
        for src_id, src, angle, ships, turns in prong_landings:
            _commit_fleet(
                world, moves, spent, target_locked,
                src_id, reinforcer.id, angle, turns, int(ships),
            )
            mode_log[src_id] = "multiprong"
        mode_log[reinforcer.id] = "multiprong-target"
        return  # only one second prong per turn


def _build_multiprong_attack(world, target, available, spent, target_locked):
    """Plan a 1-3 source attack on `target` from surplus ships (post-hammer,
    post-expand, post-defense). Returns (strength, arrival_turn, landings, e_at_arrival) or None.

    Each landing: (src_id, src, angle, ships, turns).
    """
    sources = []
    for src in world.my_planets:
        avail = available[src.id] - spent[src.id]
        if avail < MULTIPRONG_MIN_PER_CONTRIBUTOR:
            continue
        # Estimate aim with full avail to filter by travel.
        aim = aim_at_target(src, target, max(MULTIPRONG_MIN_PER_CONTRIBUTOR, avail), world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            continue
        _angle, est_turns = aim
        if est_turns > MULTIPRONG_MAX_TRAVEL:
            continue
        sources.append((est_turns, src, avail))
    if not sources:
        return None
    sources.sort(key=lambda kv: kv[0])  # nearest (fastest) first

    # Common arrival = max(participant travels). Add sources until we beat
    # E_at_arrival * MULTIPRONG_E_OVERKILL.
    chosen = []
    for est_turns, src, avail in sources[:MULTIPRONG_MAX_PARTICIPANTS]:
        chosen.append((est_turns, src, avail))
        common_arrival = max(t for t, _, _ in chosen)
        _, e_at_arrival = predict_defender_at_arrival(world, target, common_arrival)
        total_avail = sum(a for _, _, a in chosen)
        required = int(math.ceil(e_at_arrival * MULTIPRONG_E_OVERKILL)) + 1
        if total_avail >= required:
            break
    common_arrival = max(t for t, _, _ in chosen)
    _, e_at_arrival = predict_defender_at_arrival(world, target, common_arrival)
    required = int(math.ceil(e_at_arrival * MULTIPRONG_E_OVERKILL)) + 1
    total_avail = sum(a for _, _, a in chosen)
    if total_avail < required:
        return None

    # Trim last contributor to exact need.
    slack = total_avail - required
    if slack > 0 and chosen:
        last_turn, last_src, last_avail = chosen[-1]
        trimmed = last_avail - slack
        if trimmed >= MULTIPRONG_MIN_PER_CONTRIBUTOR:
            chosen[-1] = (last_turn, last_src, trimmed)

    # Re-aim each contributor with EXACT ship counts.
    landings = []
    final_strength = 0
    for est_turns, src, ships in chosen:
        if ships < MULTIPRONG_MIN_PER_CONTRIBUTOR:
            return None
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel, world=world)
        if aim is None:
            return None
        angle, turns = aim
        if turns > MULTIPRONG_MAX_TRAVEL:
            return None
        landings.append((src.id, src, angle, int(ships), int(turns)))
        final_strength += int(ships)

    # Re-validate: defender at the post-re-aim worst-case arrival.
    final_arrival = max(turns for _, _, _, _, turns in landings)
    _, final_defender = predict_defender_at_arrival(world, target, final_arrival)
    final_required = int(math.ceil(final_defender * MULTIPRONG_E_OVERKILL)) + 1
    if final_strength < final_required:
        return None

    return final_strength, final_arrival, landings, final_defender


# ============================================================
# Top-level plan
# ============================================================

def plan_moves(world, deadline=None):
    global _planet_idle_counts, _promoted_stockpiles, _pending_commitments

    # Prune the persistent commitment ledger: drop entries whose fleets
    # should already have arrived. Also drop entries pointing at targets
    # we now own (capture succeeded — no need to keep blocking ourselves).
    # V13.3 P3: ALSO drop commitments where target ownership changed since
    # commit (zvold-style failTolerant cleanup — situation is different from
    # what we planned for, free main to re-evaluate).
    def _commitment_viable(c):
        if c["arrival_abs"] <= world.step:
            return False
        target = world.planet_by_id.get(c["target_id"])
        if target is None:
            return False
        if target.owner == world.player:
            return False
        if FAILTOLERANT_ENABLED:
            owner_at_commit = c.get("owner_at_commit")
            if owner_at_commit is not None and int(target.owner) != int(owner_at_commit):
                return False
        return True
    _pending_commitments[:] = [c for c in _pending_commitments if _commitment_viable(c)]

    # V12.8c: refresh wounded-neutral watchlist from per-turn ship deltas.
    _update_neutral_watchlist(world)

    moves = []
    spent = defaultdict(int)
    target_locked = set()
    mode_log = {}

    # Mode 1 — Absorb / reserve walk for every owned planet.
    rescue_needs = {}
    available = {}
    for p in world.my_planets:
        arrivals = world.arrivals_by_planet.get(p.id, [])
        reserve, holds, deficit, dline = compute_planet_reserve(
            p, arrivals, world.player
        )
        # K-mode 4P: anticipatory reserve. If a primary attacker is identified
        # for this planet, hold back a small extra for likely follow-up next
        # turn. Math: reserve += min(8, primary_threat_amount * 0.15). Bounded
        # so we don't over-defend.
        if not world.is_2p:
            ppt = getattr(world, "our_planet_primary_threat", {}).get(p.id)
            if ppt is not None:
                _attacker, threat_amt = ppt
                extra = min(8, int(threat_amt * 0.15))
                reserve = min(int(p.ships), reserve + extra)
        available[p.id] = max(0, int(p.ships) - reserve)
        if not holds:
            rescue_needs[p.id] = (deficit, dline, p)
            mode_log[p.id] = "absorb-need-rescue"
        elif arrivals:
            mode_log[p.id] = "absorb"

    # V12.8ex: deadline gate between modes — defense always runs (must not
    # skip), but expand/hammer/multiprong are skipped if we've blown past
    # the soft deadline.
    def _over_budget():
        return deadline is not None and time.perf_counter() >= deadline

    # Mode 1b — Comet evacuation. Run BEFORE defense so soon-to-expire comets
    # dump their ships before they're conscripted into reserves. Ships left
    # on a comet that exits the system are lost outright.
    handle_comet_evac(world, available, spent, target_locked, moves, mode_log)

    # Mode 2 — Defense.
    handle_defense(world, rescue_needs, available, spent, target_locked,
                   moves, mode_log)

    # B1 (one-brain pre-pass): reserve the accumulator-lead so handle_expand
    # can't drain it before mega-hammer / accumulator run. Runs AFTER defense
    # so life-saving still preempts; runs BEFORE cheap-pickup/expand so the
    # lead is protected from small-ship spending. 4P-only initially.
    _brain_reserve_lead(world, available, spent, mode_log)

    # Mode 2b — V12.4d cheap-pickup pre-pass (4P-only).
    # V12.9: skip when Melis search will run (it handles same targets).
    if not _over_budget():
        if not (SEARCH_EXPAND_4P_ENABLED and not world.is_2p
                and SEARCH_DISABLES_CHEAP_PICKUP):
            handle_cheap_pickup(world, available, spent, target_locked, moves, mode_log)

    # Mode 3 — Expand (solo + coalition).
    if not _over_budget():
        handle_expand(world, available, spent, target_locked, moves, mode_log)

    # Mode 4-prep — Accumulator (V14.2, Phase 3.7): feed surplus from safe
    # backline planets to the lead stockpile. Runs BEFORE mega-hammer so
    # the lead's accumulated stockpile is visible to mega-hammer this turn.
    # In-flight feeds arrive on future turns to grow the lead bigger.
    if not _over_budget():
        handle_accumulator(world, available, spent, target_locked, moves, mode_log)

    # Mode 4-prelude — Mega-hammer (V14.1c): single-source overwhelming strike
    # fires before handle_hammer so big stockpiles aren't split into coalitions.
    if not _over_budget():
        handle_mega_hammer(world, available, spent, target_locked, moves, mode_log)

    # K-MODE: elimination strike on focus enemy. Runs BEFORE hammer so we
    # claim sources for focus_enemy capture before hammer disperses them
    # to other targets. Only uses idle sources (no mode_log entry) — won't
    # steal from expand/accumulator/mega.
    if not _over_budget():
        handle_eliminate_focus(world, available, spent, target_locked, moves, mode_log)

    # Mode 4 — Hammer (persistent coordinated strike).
    if not _over_budget():
        handle_hammer(world, available, spent, target_locked, moves, mode_log)

    # Mode 4b - Multi-prong forcing (V12.3c1, 2P only).
    if not _over_budget():
        handle_multiprong(world, available, spent, target_locked, moves, mode_log)

    # Mode 5 — Grow (implicit: planets without an entry in mode_log just sit).

    # Update per-planet idle counters and stockpile promotion.
    for p in world.my_planets:
        if mode_log.get(p.id) and "absorb" not in mode_log[p.id]:
            _planet_idle_counts[p.id] = 0
        else:
            _planet_idle_counts[p.id] = _planet_idle_counts.get(p.id, 0) + 1
            if _planet_idle_counts[p.id] >= HAMMER_SURROUNDED_PROMOTE_TURNS:
                _promoted_stockpiles.add(p.id)

    return moves


# ============================================================
# Agent entry
# ============================================================

def agent(obs, config=None):
    global _agent_step, _hammer_plan, _planet_idle_counts, _promoted_stockpiles, _pending_commitments
    global _game_num_players, _2p_patient_streak, _2p_prod_share_history

    global _opp_profile  # V12.8et
    obs_step = _read(obs, "step", 0) or 0
    if obs_step == 0:
        _agent_step = 0
        _hammer_plan = None
        _planet_idle_counts = {}
        _promoted_stockpiles = set()
        _pending_commitments = []
        _game_num_players = None
        _2p_patient_streak = 0
        _2p_prod_share_history = []
        _neutral_prev_ships.clear()
        _neutral_wounded.clear()
        _enemy_prev_ships.clear()
        _enemy_recently_launched.clear()
        _planet_prev_owner.clear()
        _freshly_lost_planets.clear()
        _opp_profile = {}
    _agent_step += 1

    start = time.perf_counter()
    world = World(obs, inferred_step=_agent_step - 1)
    if not world.my_planets:
        return []

    # V12.8et: 4P opponent profiling — call gated to skip 2P entirely.
    if not world.is_2p:
        _update_opp_profile_4p(world)

    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = max(0.5, act_timeout * SOFT_DEADLINE_FRACTION)
    deadline = start + soft_budget

    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "Planet", "Fleet"]