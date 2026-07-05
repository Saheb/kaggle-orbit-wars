import math
import os
import time
from collections import defaultdict, namedtuple

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

# ============================================================
# Strategy knobs (V11.6 patient design)
# ============================================================

PSM_OPENING_TURN = 20

# Mode 1 — Absorb / reservation walk
ABSORB_MIN_THREAT = 3            # incoming hostile fleets <this many ships are noise
ABSORB_PROJECTION_MARGIN = 1     # running balance must stay >= this to "survive"

# Mode 2 — Defense
DEFENSE_OVERSEND = 2             # send deficit + this for safety
DEFENSE_COALITION_MAX = 2        # rescue coalition cap

# Global fleet-quality floor — the patient ethos says ONE main fleet at the
# target, not many drips. Fleets below this size fly slowly (speed≈1.0), miss
# orbital targets, and waste tempo. Skip the dispatch entirely if the right-
# sized fleet would be smaller than this.
MIN_DISPATCH_SHIPS = 5

# Mode 3 — Expand
EXPAND_K_OPENING = 2             # turns 0..PSM_OPENING_TURN: examine 2 nearest
EXPAND_K_MID = 1                 # mid-game: examine ONLY the absolute nearest
EXPAND_MAX_TRAVEL_OPENING = 20
EXPAND_MAX_TRAVEL_MID = 14
EXPAND_MIN_MARGIN = 0            # exact-+1 capture (needed_to_capture already adds the +1 to flip)
EXPAND_MIN_SHIPS = MIN_DISPATCH_SHIPS

# V12.3c5 (2.5) hash-entropy tiebreak. In 2P, near-equal-distance candidates
# get reordered by a deterministic hash so two mirrored PATIENT bots don't
# always pick the same target. Replayable since the hash is salted on (player,
# step, src, target) only.
TIEBREAK_ENABLED = True
TIEBREAK_EPS_FRAC = 0.005   # 0.5% of best distance defines the tie bucket
TIEBREAK_EPS_MIN = 0.5      # absolute floor so very-near sources still tie

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
VALUE_WEIGHT_2P = float(os.environ.get("V126_VALUE_WEIGHT_2P", "4.0"))
VALUE_WEIGHT_4P = float(os.environ.get("V126_VALUE_WEIGHT_4P", "2.0"))

# V12.4b — Anti-snipe veto (2P-only). Before firing on a NEUTRAL, simulate
# post-capture surplus + production growth vs known incoming enemy fleets;
# refuse if balance ever drops <=0. 2P-only because the 4P 192-game test
# showed -1.9pp 1st rate AND a structural regression (55 third-place
# finishes vs 12_4a's 4) — with 3 enemies, "some enemy fleet incoming"
# is too easy to trigger and the bot starves itself of expansion.
ANTI_SNIPE_ENABLED = os.environ.get("V124_ANTI_SNIPE", "1") != "0"
ANTI_SNIPE_HORIZON = 25
ANTI_SNIPE_2P_ONLY = True

# V12.4c — Counter-snipe priority (2P-only). Prepend neutrals where a known
# enemy fleet WILL capture before us, with cheap re-flip plans (size = post-
# enemy-capture defender + production*delay + 1). 2P-only because 4P 192-game
# test showed -14pp 1st rate vs noise floor — too many "enemy-committing"
# opportunities in 4P starve main expansion. Toggle V124_COUNTER_SNIPE=0.
COUNTER_SNIPE_ENABLED = os.environ.get("V124_COUNTER_SNIPE", "1") != "0"
COUNTER_SNIPE_2P_ONLY = True
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

# Mode 3b — Coalition expand (neutrals only; enemies route through hammer).
# Patient ethos: prefer ONE big solo fleet over two coalition fleets. Coalition
# only fires when a target genuinely can't be soloed by any one source AND the
# target is large enough that splitting is unavoidable.
COALITION_ENABLED = True
COALITION_MAX_PARTICIPANTS = 3   # solo + 2 partners maximum
COALITION_NEUTRALS_ONLY = False  # V12.2 R1b: allow enemy-target coalitions
COALITION_MAX_TRAVEL_BONUS = 4   # partner can be slightly further than solo cap
COALITION_MIN_PER_CONTRIBUTOR = 10   # no tiny 5-ship "halves" — minimum substantive piece
COALITION_MIN_TARGET_SHIPS = 20      # V12.6d: was 20 — allow 2-source coalitions on medium neutrals

# Mode 4 — Hammer
HAMMER_ENABLED = True
HAMMER_STOCKPILE_MIN = 50
HAMMER_TARGET_PROD_MIN = 2
HAMMER_PROD_SHARE_TRIGGER = 0.40
HAMMER_OVERKILL_RATIO = 1.30
HAMMER_SURROUNDED_PROMOTE_TURNS = 10  # idle this many turns => permanent stockpile
HAMMER_MAX_TRAVEL = 40                # hammers reach further than expansion
HAMMER_ABORT_OVERRUN_RATIO = 1.05     # if defender exceeds committed x this, abort
HAMMER_PLAN_REVALIDATE_INTERVAL = 1   # re-check defender every turn
HAMMER_MIN_PER_CONTRIBUTOR = 8        # drop tiny stockpile contributions

# Mode 4b - Multi-prong forcing (V12.3c1, 2P only).
# When a hammer plan is active against target T and an enemy reinforcer E is
# pumping ships into T, open a second prong at E using surplus ships. Strict
# credibility gates keep this from splitting offense into two underweight prongs.
MULTIPRONG_ENABLED = True
MULTIPRONG_2P_ONLY = True
# Reinforcer must have at least this much in flight toward T (relative to T's
# arrival deficit) to count as a real reinforcer (vs a probe).
MULTIPRONG_REINFORCER_MIN_RATIO = 1.0
# Second prong must land with > E_home * this to satisfy credibility (E really
# takeable post-launch).
MULTIPRONG_E_OVERKILL = 1.05
# We must be prong-credible: committed_T + planned_E >= needed(T) + needed(E) * this.
MULTIPRONG_CREDIBILITY_FACTOR = 0.6
MULTIPRONG_MAX_TRAVEL = 35
MULTIPRONG_MIN_PER_CONTRIBUTOR = 8
MULTIPRONG_MAX_PARTICIPANTS = 3

# Late-game flush (only when patient farming has saturated and time is short)
LATE_FLUSH_REMAINING_TURNS = 30
LATE_FLUSH_OVERKILL_RATIO = 1.05      # tolerate thinner margins under time pressure

# Per-turn budget guard
SOFT_DEADLINE_FRACTION = 0.82

# Race-to-neutral (V12.1a)
RACE_ENABLED = True
RACE_HORIZON_TURNS = 25          # only consider enemies that could capture within this window
RACE_MAX_NEUTRAL_DIST = 50.0     # don't bother computing race for unreachable neutrals
RACE_TIE_GOES_TO_LARGER = True   # we still race when arrivals tie, since combat resolves by ship count

# Adaptive personality (V12.1b)
PERSONALITY_ENABLED = True
PERSONALITY_AGG_HIGH = 0.30      # enemy_ships_in_flight / total_enemy_ships above this => PRESSURE
PERSONALITY_AGG_LOW = 0.10       # below this => OPPORTUNISTIC
PERSONALITY_MIN_SAMPLE = 50      # below this many enemy ships, signal too weak — stay PATIENT

MODE_PARAMS = {
    "patient": {
        "expand_k_opening": 2,            # V12.3b 2.2b: explicit opening overrides
        "expand_max_travel_opening": 20,
        "expand_k_mid": 1,
        "expand_max_travel_mid": 14,
        "hammer_prod_share": 0.40,
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,       # V12.3b 2.3: explicit (was global)
    },
    "opportunistic": {
        "expand_k_opening": 2,
        "expand_max_travel_opening": 20,
        "expand_k_mid": 2,                # examine 2 nearest mid-game (vs 1)
        "expand_max_travel_mid": 18,      # +4 reach
        "hammer_prod_share": 0.35,        # slightly more eager to hammer
        "hammer_overkill": 1.30,
        "hammer_stockpile_min": 50,
    },
    "pressure": {
        "expand_k_opening": 2,
        "expand_max_travel_opening": 20,
        "expand_k_mid": 1,
        "expand_max_travel_mid": 16,      # slight reach increase to grab contested
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
        "expand_max_travel_opening": 30,  # V12.3b 2.2b: was 20 — engage cross-map
        "expand_k_mid": 4,                # was 1 — duels need broader target search
        "expand_max_travel_mid": 28,      # was 14 — engage past midline of 100x100 map
        "hammer_prod_share": 0.30,        # was 0.40 — symmetric duels rarely hit 40%
        "hammer_overkill": 1.15,          # was 1.30 — only one sniper to defend against
        "hammer_stockpile_min": 25,       # V12.3b 2.3: was 50 — duel planets churn ships
    },
    "opportunistic": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 30,
        "expand_k_mid": 6,
        "expand_max_travel_mid": 30,
        "hammer_prod_share": 0.28,
        "hammer_overkill": 1.15,
        "hammer_stockpile_min": 25,
    },
    "pressure": {
        "expand_k_opening": 5,
        "expand_max_travel_opening": 30,
        "expand_k_mid": 4,
        "expand_max_travel_mid": 35,      # V12.5d: was 30 — extend cross-map mid-game reach
        "hammer_prod_share": 0.20,        # V12.5b: was 0.25 — fire ~5 turns earlier than V12.4 expects
        "hammer_overkill": 1.10,
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


AIM_MAX_ITERS = 12          # was 5 — orbital targets sometimes need more
AIM_CONVERGE_TURNS = 1
AIM_CONVERGE_DIST = 0.4


def aim_at_target(src, target, ships, initial_by_id, ang_vel):
    """Returns (angle, turns) for sending `ships` from src to hit target.
    Iterates orbital prediction. Returns None if the path is blocked by the
    sun OR if convergence isn't reached — better to skip a target than fire
    a fleet that wanders past it because our aim didn't settle."""
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        return None
    init = initial_by_id.get(target.id)
    if init is None:
        return est
    if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
        return est

    angle, turns = est
    tx, ty = target.x, target.y
    for _ in range(AIM_MAX_ITERS):
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
    if initial_by_id is not None:
        max_t = int(math.ceil(min(best_t, float(SIM_HORIZON))))
        for t in range(1, max_t + 1):
            fx = fleet.x + dx_dir * speed * t
            fy = fleet.y + dy_dir * speed * t
            for p in planets:
                if not _is_orbital(p):
                    continue
                px, py = predict_planet_position(p, initial_by_id, ang_vel, t)
                rr = p.radius * p.radius
                if (fx - px) ** 2 + (fy - py) ** 2 < rr:
                    if t < best_t:
                        best_t, best_p = float(t), p
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
    skip_in_window_hostiles = hostile_in_window < ABSORB_MIN_THREAT

    events = defaultdict(int)
    for eta, owner, ships in arrivals:
        if ships <= 0:
            continue
        if owner == player:
            events[eta] += ships              # friendly reinforce
        elif owner == -1:
            continue                          # not a real combat scenario
        else:
            if skip_in_window_hostiles and int(eta) <= absorb_window:
                continue                      # aggregate-noise within window
            events[eta] -= ships              # hostile threat

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
# World snapshot
# ============================================================

class World:
    def __init__(self, obs, inferred_step=None):
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

        # Adaptive personality (V12.1b): pick a mode based on opponent activity.
        # Opening always stays PATIENT — initial expansions look like aggression
        # but aren't. V12.2 R2: 2P uses MODE_PARAMS_2P (broader search, lower
        # hammer threshold) and a forced-pressure timeout if PATIENT stalls.
        self.mode = _detect_mode(self) if PERSONALITY_ENABLED else "patient"
        params_table = MODE_PARAMS_2P if self.is_2p else MODE_PARAMS
        self.mode_params = params_table[self.mode]

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

# Persistent dispatch ledger. fleet_target_planet can't reliably attribute
# in-flight fleets to orbital targets (the target has rotated since launch),
# so we track our own commitments here. Each entry: {target_id, ships,
# arrival_abs}. Pruned each turn once the fleet should have arrived. Reset
# on a new game.
_pending_commitments = []


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
# Tempo helpers — avoid double-firing & comet/transient targeting
# ============================================================

def is_targetable(world, target):
    """Comets travel along non-orbital elliptical paths that aim_at_target can't
    predict. Aiming at them produces fleets that wander and often hit the sun.
    Skip them entirely as expansion / hammer targets."""
    return target.id not in world.comet_ids


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
    moves.append([src_id, float(angle), int(ships)])
    spent[src_id] += int(ships)
    target_locked.add(target_id)
    _pending_commitments.append({
        "target_id": int(target_id),
        "ships": int(ships),
        "arrival_abs": int(world.step) + int(turns),
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
    if max_avail < MIN_DISPATCH_SHIPS:
        return None
    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel)
    if aim is None:
        return None
    angle, turns = aim
    if turns > max_travel:
        return None
    need = needed_to_capture(tgt, turns)
    ships = max(MIN_DISPATCH_SHIPS, need + EXPAND_MIN_MARGIN)
    if ships < MIN_DISPATCH_SHIPS or ships > max_avail:
        return None
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel)
    if aim2 is None:
        return None
    angle, turns = aim2
    if turns > max_travel:
        return None
    need2 = needed_to_capture(tgt, turns)
    if ships < need2 + EXPAND_MIN_MARGIN:
        ships = need2 + EXPAND_MIN_MARGIN
        if ships > max_avail:
            return None
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel)
        if aim3 is None:
            return None
        angle, turns = aim3
        if turns > max_travel:
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
    """
    if not rescue_needs:
        return

    for victim_id, (deficit, deadline, victim) in rescue_needs.items():
        if victim_id in target_locked:
            continue
        need = deficit + DEFENSE_OVERSEND

        # Single-source candidates.
        solo = []
        for src in world.my_planets:
            if src.id == victim_id:
                continue
            avail = available[src.id] - spent[src.id]
            if avail < need:
                continue
            aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel)
            if aim is None:
                continue
            angle, turns = aim
            if deadline is not None and turns > deadline:
                continue
            solo.append((turns, src.id, src, angle, avail))

        if solo:
            solo.sort()  # closest first
            _t, src_id, src, _angle_est, avail = solo[0]
            send = min(avail, need)
            send = max(send, deficit + 1)
            # Patient ethos: if even the rescue is below the dispatch floor,
            # don't fire a tiny fleet that flies at speed 1 and may not arrive.
            if send < MIN_DISPATCH_SHIPS:
                send = MIN_DISPATCH_SHIPS if avail >= MIN_DISPATCH_SHIPS else 0
            if send <= 0:
                mode_log[victim_id] = "doomed-too-poor"
                continue
            # Re-aim with the EXACT ship count we'll send (speed depends on ships).
            aim_final = aim_at_target(src, victim, send, world.initial_by_id, world.ang_vel)
            if aim_final is None:
                mode_log[victim_id] = "doomed-aim-blocked"
                continue
            angle, turns = aim_final
            if deadline is not None and turns > deadline:
                mode_log[victim_id] = "doomed-too-slow"
                continue
            _commit_fleet(world, moves, spent, target_locked,
                          src_id, victim_id, angle, turns, int(send))
            mode_log[victim_id] = "defended-by-solo"
            mode_log[src_id] = "defense"
            continue

        # 2-source coalition fallback.
        if not COALITION_ENABLED:
            mode_log[victim_id] = "doomed"
            continue
        coalition = _find_defense_coalition(
            world, victim, deadline, need, available, spent
        )
        if coalition is None:
            mode_log[victim_id] = "doomed"
            continue
        for src_id, src, angle, ships, turns in coalition:
            _commit_fleet(world, moves, spent, target_locked,
                          src_id, victim_id, angle, turns, int(ships))
            mode_log[src_id] = "defense-coalition"
        mode_log[victim_id] = "defended-by-coalition"


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
        aim = aim_at_target(src, victim, avail, world.initial_by_id, world.ang_vel)
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
            aim_i = aim_at_target(s_i, victim, ship_i, world.initial_by_id, world.ang_vel)
            aim_j = aim_at_target(s_j, victim, ship_j, world.initial_by_id, world.ang_vel)
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
            _commit_fleet(world, moves, spent, target_locked,
                          src.id, n.id, angle, turns, int(ships))
            mode_log[src.id] = "cheap-pickup"
            break


def handle_expand(world, available, spent, target_locked, moves, mode_log):
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
    if not nonfriendly:
        return

    def frontier_key(src):
        return min(dist(src.x, src.y, t.x, t.y) for t in nonfriendly)

    sources = sorted(world.my_planets, key=frontier_key)

    for src in sources:
        avail = available[src.id] - spent[src.id]
        if avail < MIN_DISPATCH_SHIPS:
            continue
        # V12.4d: allow main expand to fire after cheap-pickup pre-pass
        # (spent[src.id] already accounts for the pre-pass spend; we just
        # don't want the source's freebie to lock out a strategic capture).
        status = mode_log.get(src.id)
        if status and status != "cheap-pickup":
            continue  # already used in defense / absorb

        candidates = _nearest_targets(src, world, K, max_travel, target_locked)
        fired_solo = False
        for tgt, _approx_dist in candidates:
            if friendly_already_committed(world, tgt.id):
                continue
            plan = plan_solo_capture(world, src, tgt, avail, max_travel)
            if plan is None:
                continue
            angle, turns, ships = plan
            # V12.1a race-loss skip: a strict ETA check after orbital aim. If
            # our actual planned arrival is later than the enemy's earliest
            # capture turn, don't fire — better to stockpile and grab an
            # uncontested target next turn than waste a fleet on a lost race.
            if RACE_ENABLED and tgt.owner == -1:
                enemy_eta = world.enemy_race_eta.get(tgt.id)
                if enemy_eta is not None and turns > enemy_eta:
                    # V12.4c: try counter-snipe (re-flip after enemy capture).
                    snipe = _plan_counter_snipe(world, src, tgt, avail, max_travel)
                    if snipe is None:
                        continue
                    angle, turns, ships = snipe
            # V12.4b: veto neutral captures that would be sniped post-capture.
            if tgt.owner == -1 and not _capture_holds_against_snipe(world, tgt, turns, int(ships)):
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

    aim = aim_at_target(src, tgt, max_avail, world.initial_by_id, world.ang_vel)
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
    aim2 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel)
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
        aim3 = aim_at_target(src, tgt, ships, world.initial_by_id, world.ang_vel)
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
    if not enemy_after:
        return True

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
    candidates = []
    for t in world.planets:
        if t.owner == world.player:
            continue
        if t.id in target_locked:
            continue
        if not is_targetable(world, t):
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
        candidates.append((t, weighted, raw))
    if not candidates:
        return []
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
    aim = aim_at_target(partner, tgt, ships, world.initial_by_id, world.ang_vel)
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
    partners = []
    for p in world.my_planets:
        if p.id == src.id:
            continue
        avail = available[p.id] - spent[p.id]
        if avail < COALITION_MIN_PER_CONTRIBUTOR:
            continue
        # Estimate-only aim with full avail to filter by travel.
        est = aim_at_target(p, tgt, avail, world.initial_by_id, world.ang_vel)
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
        est_src = aim_at_target(src, tgt, src_avail, world.initial_by_id, world.ang_vel)
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
        aim_src = aim_at_target(src, tgt, s_src, world.initial_by_id, world.ang_vel)
        aim_p = aim_at_target(p, tgt, s_p, world.initial_by_id, world.ang_vel)
        if aim_src is None or aim_p is None:
            continue
        a_src, t_src = aim_src
        a_p, t_p = aim_p
        if t_src > max_travel or t_p > max_travel:
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
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel)
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
        avail = available[p.id] - spent[p.id]
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
            aim = aim_at_target(src, tgt, max(1, avail), world.initial_by_id, world.ang_vel)
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
        required = int(math.ceil(d_ships * overkill)) + 1

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
        slack = accum - required
        if slack > 0 and chosen:
            last_turn, last_src, last_avail, last_angle = chosen[-1]
            trimmed = last_avail - slack
            if trimmed < HAMMER_MIN_PER_CONTRIBUTOR:
                chosen.pop()
                if not chosen or sum(c[2] for c in chosen) < required - last_avail:
                    chosen.append((last_turn, last_src, last_avail, last_angle))
            else:
                chosen[-1] = (last_turn, last_src, trimmed, last_angle)

        score = required - target_arrival * 0.5  # cheaper + sooner = better
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
        aim = aim_at_target(src, target, max(MULTIPRONG_MIN_PER_CONTRIBUTOR, avail), world.initial_by_id, world.ang_vel)
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
        aim = aim_at_target(src, target, ships, world.initial_by_id, world.ang_vel)
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
    _pending_commitments[:] = [
        c for c in _pending_commitments
        if c["arrival_abs"] > world.step
        and not (
            world.planet_by_id.get(c["target_id"]) is not None
            and world.planet_by_id[c["target_id"]].owner == world.player
        )
    ]

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
        available[p.id] = max(0, int(p.ships) - reserve)
        if not holds:
            rescue_needs[p.id] = (deficit, dline, p)
            mode_log[p.id] = "absorb-need-rescue"
        elif arrivals:
            mode_log[p.id] = "absorb"

    # Mode 2 — Defense.
    handle_defense(world, rescue_needs, available, spent, target_locked,
                   moves, mode_log)

    # Mode 2b — V12.4d cheap-pickup pre-pass (4P-only).
    handle_cheap_pickup(world, available, spent, target_locked, moves, mode_log)

    # Mode 3 — Expand (solo + coalition).
    handle_expand(world, available, spent, target_locked, moves, mode_log)

    # Mode 4 — Hammer (persistent coordinated strike).
    handle_hammer(world, available, spent, target_locked, moves, mode_log)

    # Mode 4b - Multi-prong forcing (V12.3c1, 2P only).
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
    _agent_step += 1

    start = time.perf_counter()
    world = World(obs, inferred_step=_agent_step - 1)
    if not world.my_planets:
        return []

    act_timeout = _read(config, "actTimeout", 1.0) if config is not None else 1.0
    soft_budget = max(0.5, act_timeout * SOFT_DEADLINE_FRACTION)
    deadline = start + soft_budget

    return plan_moves(world, deadline=deadline)


__all__ = ["agent", "Planet", "Fleet"]
