# Why the out-mass wall resists reward-side fixes (limits doc)

Durable record of *why* the decisive-mass / force-concentration reward levers keep failing to move
`out-massed%`, grounded in the actual code. Companion to `docs/conclusions.md` (closed directions) and
`docs/next-steps.md` (live work). Written 2026-06-16 while decmass-v2 (the properly-tested re-run) is live.

**The wall, restated:** vs strong forward-projecting planners (deb/Ajay, and the h-ladder) we lose ~95-98%
of contested captures by being **out-massed** — the enemy concentrates more force at the contested moment
than we do. Probe A ([[project_aggregation_probe]]) showed we *already multi-source aggregate at winner rates*
(aggTurn ≈ rank1's 0.082, `sameSrc=0`) → the wall is **aggregation SUFFICIENCY (size-to-floor) + retention,
NOT presence**, and **not the action grammar**. So the lever has to make our aggregated attacks *sufficient*
against the projected defense. decmass is the reward that tries to price exactly that.

---

## Part 1 — How the decmass reward decouples from the real contest (code review)

The credit fires when `mass ≥ floor` (`torch_env._decisive_mass_bonus`, line 842), where `mass`/`floor` come
from `_decisive_mass_fields` (763). The problem: **neither `mass` nor `floor` is what actually decides the
fight**, so `decis`/`cross` can rise without `out-massed%` moving. Ranked by likely impact:

1. **`mass` is the un-windowed SUM of all inflight fleets aimed at `t` (line 811); combat resolves
   per-arrival.** `m = scatter_add(all our fleets → t)` over *every* ETA; `eta` is taken separately as the
   *max* (816). The max-ETA fix correctly inflates the **floor** for staggered arrivals (co-arriving the same
   mass crosses a *smaller* floor — good incentive), but does nothing to the **mass** side: fleets launched at
   t and t+40 both count, even though the early wave is defeated *alone* before the late wave arrives at a
   re-defended planet. A staggered pile that loses **in detail** in the real game still crosses the modeled
   floor and is credited. *The max-ETA fix hardened the floor, not the mass.*

2. **The floor is a SNAPSHOT that under-models a REACTIVE opponent (819-829).** `inbound` counts only enemy
   fleets *already* en route now; the defense the opponent will *launch as we approach* is a static estimate
   `β·rho(eta)·reachable_enemy_mass`. And `rho = clamp((eta−3)/12, 0, 1)` (826) → for fast strikes (eta<3)
   **rho=0, reactive defense is ignored entirely**, and `enemy_mass` decays with distance (789). So in exactly
   the close/contested cases the floor *under*-counts what a strong planner actually scrambles → we cross the
   modeled bar, the real opponent out-reacts, we lose, and the +0.2 already fired. **Crossing the floor is
   decoupled from winning the contest by construction.** This is the decmass1 signature baked into the proxy,
   not a win-starvation artifact.

   *1 + 2 together:* `cross` measures "did our cumulative tonnage beat an optimistic snapshot," which can move
   independently of "did we stop being out-massed."

3. **Per-crossing credit + re-arm = farmable by oscillation (842-846).** Credit is the rising edge
   `suff & ~prev_suff`; `prev` re-arms when mass drops back below floor. Building floor-crossing waves on a
   target, letting them lapse, and rebuilding pays +0.2 each time — **with no requirement to ever capture it.**
   A policy can maximize `decis` by keeping targets *flickering* above the floor (perpetual threatening /
   over-launch — the ping-pong pathology) instead of committing decisively. Most natural explanation for
   decmass1's `decis 13→19` with nothing structural moving.

4. **Flat +0.2 per crossing → gradient flows to EASY crossings, not the wall (845).** The credit is
   difficulty-blind. The under-massed losses are the *hard* contests (high floor); crossing a weak target's
   floor is cheap. PPO maximizes total reward by farming cheap crossings, with ~zero pressure on the lost
   contests that *are* the wall. Probe A shows we already aggregate at winner rates on games we win → the
   reward largely reinforces existing easy aggregation and barely touches the residual.

5. **Loose geometric target attribution (`_fleet_target_idx`, 739-761).** A fleet is assigned to the nearest
   alive planet *ahead along its heading* (perp < r+2). A fleet passing near a planet en route elsewhere is
   counted on the near planet → `mass`/`inbound` over/mis-counted → false crossings. Faithful to the sim's own
   targeting but loose enough to inflate `decis`/`cross` without real convergence.

**Lower-order:** floor is **producer_v2-calibrated** (β2.2, horizon18) but the training ladder is producer
**h10/h12/h14** (shorter horizons, defend with less) → crossing the v2-floor may not map to beating h12;
and the ~20% self/pool-self games still apply the reward symmetrically (farmable, no opponent-relative
meaning — diluted from decmass1's 40% but nonzero).

**Read rule for decmass-v2:** `cross↑`/`decis↑` with **out-massed% flat** ⇒ it's the reward-design decoupling
above (1-4), not win-starvation → fix the *reward*, not the pool. `cross↑` **and** out-massed↓ together ⇒ the
decouplings weren't fatal at this magnitude and the lever works.

---

## Part 2 — The deeper trap: proxy rewards decouple, outcome rewards get gamed

The decmass failures (Part 1) are all "proxy ≠ outcome." The obvious fix is **outcome-tied credit** (only
reward a crossing that actually leads to a capture/hold). But the history says outcome rewards have their own
failure class in this game:

- **rev49 — production-weighted capture (outcome reward) → carpet-bomb** (`docs/training.md`). Rewarding total
  `production_delta` captured is maximized by firing at *everything* simultaneously, not selective decisive
  strikes. Clean (bad) Nash: clip_frac 0.88, srcs_multi 5-6, fire[0] 0.6. **Outcome = quantity → spray.**
- **caputil1 — capture follow-through (outcome-ish) → no wall movement** (2026-06-16, this session). Rewarding
  a captured planet for being used/held within 30 steps couldn't grip: median-hold is 13st (captures die
  before the window) and triage showed losses are 98% *hopeless* (out-massed at capture), 0% cheap-save-missed
  → the reward was conditioned on surviving a contest we were already losing. [[project_capture_utility]]
- **win_margin (sparse outcome)** gives no *per-action concentration* signal — it can't tell which launch
  should have been bigger.

**So both ends fail:** a board-grounded *proxy* (decmass) decouples from the real contest; an *outcome* reward
is either gamed (production → spray) or too sparse/late (win, capture-survival) to teach which attack to size
up. This tension — not any single knob — is why the wall has survived every reward lever. A reward that works
likely needs to be **outcome-tied AND difficulty/selectivity-aware AND co-arrival-grounded** simultaneously
(e.g. credit only crossings that *capture*, weighted by floor difficulty, counting only co-arriving mass) —
which is a much narrower target than any single term we've tried, and may still be gameable.

---

## Part 3 — Per-planet value head (VDN): rigorous but under-scoped for THIS question

`docs/conclusions.md` shelved per-planet credit assignment (per-slot PPO + VDN value head) on 2026-06-09.
**Assessment: the execution was sound, but the scope does not close the question we'd ask now.**

- **Sound:** 4 variants one-delta apart (joint / per-slot / hybrid / VDN), a proper confound control
  (`--reinit-critic` proved the collapse is the *method*, not the resume/cold-critic), clean failure taxonomy
  (under-commit → over-fire → hoard). Not half-hearted mechanically.
- **Under-scoped:** it ran on the **old rev53b lineage**, was judged on **general Ajay-WR** (joint 9.4% vs VDN
  8.0%), and was tested **before** the force-concentration wall was diagnosed and before `out-massed%`/`dm`/
  aggregation metrics existed. It was **never paired with a concentration signal** — the doc itself flags it'd
  be worth revisiting "**with a firing floor**." So the trustworthy conclusion is *"per-planet credit ties/loses
  to joint on general conversion,"* **NOT** *"per-planet credit can't sharpen a concentration/sufficiency signal."*
- **Open ambiguity:** the doc's strongest reason (#1: "decoupling → the only fix is coupling = joint loss") is a
  real principle, but canonical VDN *does* couple (summed per-planet values, global TD target). If the variant
  used purely *local* per-planet advantages, the failure may be that implementation choice, not "per-planet
  value is doomed." Unresolved from the writeup.
- **Why it's relevant here:** Part 1 #4 (flat credit → gradient flows to easy crossings) is exactly a *spatial
  credit-assignment* gap. A per-planet value/advantage **paired with the decmass crossing signal** is a
  not-yet-tried combination that targets that gap directly. VDN code is preserved on branch
  `claude/strange-khorana-6f6141`. Reason #3 in conclusions.md (no edge over joint) still says "don't, absent a
  new reason to expect an edge" — Part 1 #4 is arguably that new reason, but it's a *speculative* revisit, not a
  refutation of the shelving.

---

## Part 3.5 — Over-extension REFUTED: the wall is defensive concentration on DEFENDABLE captures

We tested whether the out-mass losses are **over-extensions** — captures we physically can't defend
(planet sits closer to enemy support than ours), i.e. a **target-selection** failure rather than a
concentration one. Probe: `orbit_wars_rl/probe_overextension.py` walks each game's ownership timeline on
the Probe-A replay corpus; each capture is classified at capture time as **over-extended** (nearest enemy
planet closer than our nearest other owned planet) or **supported**, then followed to loss.

**Result (revedge1 4.72M, 128 games each):** over-extended captures are lost only **+9-11pp** more (88% vs
**77-79%** supported), consistent across Ajay and deb. The median capture is well-positioned (enemy ~1.5×
farther). **We lose 77-79% of POSITIONALLY-SUPPORTED captures, and 61-62% of all losses were supported at
capture.**

**Verdict — over-extension is REAL but SECONDARY; NOT the wall (rules out the target-selection lever).** Most
losses are planets we had the *position* to defend and lost anyway. So the wall is **defensive deployment**:
even when our support is closer than the enemy's, the enemy **out-concentrates the peel** (assembles a bigger
arriving fleet than the defense we route from nearby planets; hold-floor age-0-5 = 86% under-defended). We have
the position; we don't **concentrate the defensive response** to meet the inbound. [[project_overextension_probe]]

Combined with Probe A (grammar isn't the wall), the session has now ruled out **action grammar** and **target
selection**, leaving: **the wall is concentrating *defense* (and *opening offense*, Part 1's `<50` cross 0.23)
on contests we are already positioned for — and reward proxies (Part 1) don't reach it.**

## Part 4 — Implications

1. decmass-v2 is still the right *next* experiment (it fixes decmass1's *testing* confounds: ladder win-gradient
   + max-ETA + the `dm` leading metrics). But Part 1 says **even a clean run can fail for reward-design reasons**,
   and the `cross`-vs-`out-massed%` split is what discriminates that from win-starvation.
2. If decmass-v2 shows the decoupling (`cross↑`, out-massed flat), the staged reward fixes are: **co-arrival
   window on `mass`** (count only fleets arriving within ~k steps of the max), **capture-tied credit** (kills the
   farm + snapshot-decouple at once — but watch the rev49 spray trap), **difficulty-weighted credit** (point the
   gradient at the wall, not easy crossings). One delta at a time.
3. Per-planet value (VDN) + a concentration signal is a *legitimately open* revisit, not a closed door — but
   lower priority than the reward-grounding fixes, and only with a concrete edge hypothesis (Part 3).
4. The honest meta-point (Part 2): no pure reward term — proxy or outcome — has moved the wall. Keep a live
   hypothesis that the fix is **outcome-tied + selectivity-aware + co-arrival-grounded** *together*, or that the
   wall is ultimately an opponent-relative / curriculum problem (beat a forward-projector in training) rather
   than a shapeable one. [[project_force_concentration_wall]] [[project_decisive_mass_lever]] [[project_aggregation_probe]]
5. **The two diagnostics this session (Probe A + over-extension, Part 3.5) have narrowed the problem statement:**
   the wall is NOT action grammar (we aggregate at winner rates) and NOT target selection (we lose 78% of
   *positionally-supported* captures). It is **concentrating *defense* on defendable captures + *offense* in the
   opening** (`<50` cross 0.23) — contests we are already positioned for. Since reward proxies (Part 1) don't
   reach it and a defensive hold-floor reward would be the decmass mirror (same decoupling), the indicated next
   bet is **curriculum / opponent-relative: make holding-via-concentration NECESSARY to win** (boards where a
   capture is only holdable if you concentrate the defensive response), not another reward term.
   [[project_overextension_probe]]
