# Architectural Decision Records — Orbit Wars RL

## ADR-001: Discretized Angle Actions — 72 bins (5° resolution)

**Date:** 2025-05-19  
**Status:** Accepted

### Decision

Use 72 discrete angle bins (each 5° / π/36 radians wide) for fleet launch direction, not continuous parameterization.

### Context

The Orbit Wars action space per owned planet requires: (1) should I launch? (2) which direction? (3) how many ships? The direction choice is continuous in the real game (any angle in [0, 2π)), but we need a representation that PPO can train stably.

### Reasoning

1. **Trainability.** Discretized actions map to categorical cross-entropy in PPO. Continuous angles require von Mises losses, Gaussian mixtures, or autoregressive factorization — all significantly harder to stabilize. The competition discussion's #1 lesson was training stability; every unnecessary parametric complexity is a liability.

2. **5° resolution is sufficient.** At distance 30 (median inter-planet distance), a 5° angular error translates to ~2.6 units of positional error. Planet radii range from 1.0 to ~2.6 (radius = 1 + ln(production)), and collision detection uses point-segment intersection, not just endpoints. A fleet aimed within 5° of a target's center will still collide with it in most cases.

3. **Reasonable action space size.** Each owned-planet entity produces 3 head outputs: `fire_prob` (binary), `angle` (72-way categorical), and `ships_frac` (16 log-scale bins). That's ~89 logits per planet entity. With max ~8 owned planets producing actions, the total policy dimension is manageable for a ~300K param transformer.

4. **Action masking is trivial.** We zero out logits for angles that would cross the sun or go out of bounds — exact geometry computed once, applied as a legal action mask. With continuous actions, you'd need penalty terms or constrained projections instead.

5. **Escape hatch.** If 72 bins proves too coarse late-game, we can add a continuous refinement head later (coarse-to-fine), similar to AlphaStar's approach. But start simple.

### Consequences

- Simpler PPO implementation (no distributional complexity)
- Free action masking for sun/OOB constraints
- Slightly coarser actions than continuous, but 5° is practically sufficient
- Can add fine-grained refinement later without restructuring

---

## ADR-002: Shared Backbone with Mode Conditioning (2p/4p)

**Date:** 2025-05-19  
**Status:** Accepted

### Decision

Use a single shared transformer backbone with a learned mode token (`[2p]` or `[4p]`) prepended to the entity sequence. Do not train separate policies.

### Context

Orbit Wars supports both 2-player and 4-player modes. The game mechanics (planet conquest, fleet movement, combat, sun collision, orbit rotation) are identical regardless of player count. We could train separate policies for each mode, or share a backbone.

### Reasoning

1. **Core mechanics are identical.** Planet conquest, fleet movement, combat resolution, sun collision, orbit mechanics — these are the same regardless of player count. The transformer's early attention layers learn to compute distances, pressures, and threats identically in both modes. Wasting parameters on duplicating this is inefficient at our budget (~300K params).

2. **Data efficiency.** Training on both 2p and 4p games simultaneously doubles the self-play throughput. A shared backbone means every gradient update, regardless of mode, improves the feature extraction.

3. **The mode token allows divergence.** The learned mode embedding propagates through all attention layers, allowing the policy head to produce mode-appropriate behavior (e.g., more conservative expansion in 4p due to multi-front threats, or more aggressive in 2p where there's a single opponent).

4. **The discussion warns about instability.** Training two separate policies doubles the hyperparameter search, doubles the debugging surface, and doubles the GPU time needed to find stable configurations. One policy, one set of hyperparameters, one clip_frac trajectory to monitor.

5. **If needed later.** If 2p and 4p truly require different behaviors that the mode token can't mediate, we can split into two policy heads sharing the same backbone. But we should only do this if we see mode interference in training (e.g., clip_frac diverges in one mode but not the other).

### Implementation

The mode is encoded in the `global_features` vector (indices 8 and 9: `mode_2p` and `mode_4p`). The global token is projected separately via `mode_proj` and added to `global_proj`, which becomes the first entity in the sequence. No special token is needed — the information is already in the feature vector.

### Consequences

- Single training config, single monitoring dashboard
- Doubled data from self-play (2p + 4p games)
- Small risk of mode interference (mitigated by mode embedding)
- Can split heads later without restructuring

---

## ADR-003: Bake in Geometric Features; Discover Strategy

**Date:** 2025-05-19  
**Status:** Accepted

### Decision

Provide all exact geometric computations as input features. Let the network learn all strategic value judgments (which planet to attack, when to defend, how many ships to send).

### Context

The competition discussion's key lesson: *"Put as many inductive biases as possible"* because *"we don't have scale."* The question is where to draw the line between facts about the game (which the network shouldn't need to re-learn) and opinions about strategy (which RL should discover).

### Baked-in Features (Explicit, Computed Exactly)

| Feature | Why | File |
|---------|-----|------|
| `distance` (to each other entity) | Permutation-invariant attention can't directly compute Euclidean distance from raw x,y without wasting capacity | `features.py` |
| `travel_time` (source→target, given ships) | Depends on the exact log-speed formula `1 + 5*(log(ships)/log(1000))^1.5`; wrong approximation wastes episodes | `features.py` |
| `fleet_speed` (from ships) | Exact deterministic formula — no reason to learn this | `features.py` |
| `sun_crossing_mask` (per angle bin from source) | Critical hard constraint; fleets that cross the sun are destroyed. Learning this via negative reward takes hundreds of episodes | `action_mask.py` |
| `out_of_bounds_mask` (per angle bin from source) | Same as above — hard constraint, free action mask | `action_mask.py` |
| `incoming_friendly_pressure` (ships heading toward this planet) | Requires geometric ray-shooting across all fleets — expensive for a network to redundantly compute | `features.py` |
| `incoming_enemy_pressure` (same, enemy fleets) | Same geometric computation, high value for defense decisions | `features.py` |
| `orbit_future_position` (predicted x,y at t+5 for orbiting planets) | Requires angular_velocity and initial position — exact formula, not something to learn | `features.py` |
| `is_orbiting` (boolean per planet) | Determined by `orbital_radius + planet_radius < 50` — exact rule | `features.py` |
| `is_comet` (boolean per planet) | Available from observation, no reason to discover | `features.py` |

### Features the Network Should Discover (NOT Baked In)

| Strategic Decision | Why Let RL Discover It |
|---|---|
| Contest risk estimation | Value judgment about opponent psychology, not geometry |
| Attack value scoring (production × horizon − cost) | The network should learn its own value function via PPO reward |
| Defense reserve sizing | Strategic trade-off between expansion and defense |
| Opening vs midgame tempo | The network can learn phase-appropriate behavior from reward shaping |
| Wait gate decisions (when to delay launching) | Timing optimization is exactly what RL should discover |
| Ship allocation across targets | The whole point of training — no heuristic captures the optimal split |

### Key Principle

The geometric features are inductive biases that reduce the policy search space without constraining the policy's strategic flexibility. They're facts about the game, not opinions about strategy. The sun-crossing mask alone likely saves days of training instability — the network never needs to learn "don't send fleets into the sun" through negative reward.

### Consequences

- Features computed in `features.py` and `action_mask.py` — easy to add/remove
- Action masks eliminate illegal actions entirely (set logits to −∞)
- The policy can focus entirely on strategic decisions
- Low reversal cost: just modify the feature extraction, retrain