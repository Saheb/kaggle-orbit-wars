# NOOP Jensen gap

The NOOP Jensen gap measures how much the binary policy's decision to launch
depends on its latent target choice. It also measures the error in the old PPO
likelihood, which treated a sampled target as meaningful even when the executed
action was simply **NOOP**.

It is a diagnostic, not a score to maximize. A larger gap does not by itself
mean a stronger policy.

## Why NOOP needs marginalization

For one owned source planet, the binary policy has these executed actions:

- `NOOP`: launch nothing from this source.
- `COMMIT(t)`: launch the resolved binary commitment at target `t`.

The model is evaluated as target first, then target-conditioned fire:

1. Choose a latent target `T` from `pi(t)`.
2. Given that target, choose NOOP with probability `q_t` or COMMIT with
   probability `1 - q_t`.

Suppressing the state and source indices, the executed-action probabilities are

```text
P(COMMIT(t)) = pi(t) * (1 - q_t)
P(NOOP)      = sum_t pi(t) * q_t
```

A committed launch exposes its target, so `COMMIT(t)` has an ordinary joint
probability. A NOOP does not expose a target. All target choices that would have
produced NOOP are therefore the same environment action and must be summed.

The exact implementation is `binary_action_log_probs()` in
`orbit_wars_rl/binary_policy.py`.

### What the old PPO likelihood did

Before exact marginalization, rollout sampling still selected a target before
the fire decision. On NOOP, the target log-probability was omitted because the
target was not executed, but PPO retained the conditional fire log-probability
`log q_T` for the sampled hidden target.

Across targets, that surrogate averages to

```text
E_{T ~ pi}[log q_T]
```

The observable NOOP action actually has log-probability

```text
log E_{T ~ pi}[q_T]
```

These are not equal. The old expression is the expectation of a log; the
correct expression is the log of an expectation. Consequently, the old PPO
ratio was not the likelihood ratio of the action the environment observed. Its
gradient could depend on an arbitrary hidden target sampled on a turn where no
target was executed.

Exact marginalization fixes both rollout sampling and PPO updates by treating
the policy as one categorical distribution over `{NOOP, COMMIT(t)}`.

## Definition

The NOOP Jensen gap is the difference between the correct and old expected
NOOP log-likelihoods:

```text
J_NOOP = log E_{T ~ pi}[q_T] - E_{T ~ pi}[log q_T]
```

The logarithm is natural, so the unit is **nats**. One nat is approximately
1.443 bits.

Because `log` is concave, Jensen's inequality guarantees

```text
J_NOOP >= 0
```

The gap is zero exactly when the NOOP probability is constant over all targets
with policy mass. In that case, the target does not affect the fire decision and
marginalizing it changes nothing numerically.

The gap grows when different plausible targets produce meaningfully different
NOOP probabilities.

## A small example

Suppose there are two equally likely targets:

```text
pi = [0.5, 0.5]
q  = [0.9, 0.1]
```

The correct NOOP probability is `0.5`, so its log-probability is
`log(0.5) = -0.6931`. The old expected conditional log-probability is

```text
0.5 * log(0.9) + 0.5 * log(0.1) = -1.2040
```

Therefore

```text
J_NOOP = -0.6931 - (-1.2040) = 0.5108 nats
```

Although both targets disappear behind the same executed NOOP, the old PPO
likelihood differed sharply according to which hidden target happened to be
sampled.

If both targets instead had `q = 0.5`, the two expressions would match and the
gap would be zero.

## Equivalent KL interpretation

After observing a NOOP, Bayes' rule gives the posterior latent-target
distribution

```text
pi(t | NOOP) = pi(t) * q_t / P(NOOP)
```

Substituting this posterior shows

```text
J_NOOP = KL(pi(T) || pi(T | NOOP))
```

Thus the gap answers this question:

> How much would observing NOOP change our belief about which target the model
> had internally considered?

This is a reverse KL from the pre-action target distribution to the posterior.
It is not mutual information, and it should not be interpreted as target quality.

## Why it complements the target-flip metric

`fire_target_flip_prob` asks whether target conditioning crosses the hard
`P(COMMIT) = 0.5` decision boundary. It is useful for deterministic decode, but
it discards most of the probability distribution.

For example, two targets might have NOOP probabilities `0.70` and `0.90`. Both
remain on the NOOP side of the threshold, so there is no flip, but the Jensen
gap is positive because target choice still changes the action probability.

Read the metrics together:

- **Flip probability:** how often target conditioning changes the deterministic
  NOOP/COMMIT decision.
- **Jensen gap:** how strongly the full stochastic NOOP probability depends on
  target, including changes that do not cross 0.5.
- **Straddle rate:** how many sources have at least one target on each side of
  the 0.5 boundary, regardless of target-policy mass.

## Aggregating across a corpus

Each actionable source produces one Jensen gap. Two summaries are useful:

1. **Mean gap:** average over actionable sources. This describes the model's
   target dependence wherever it could launch.
2. **NOOP-weighted mean:** weight each source by its marginal `P(NOOP)`. This
   emphasizes sources where a NOOP is likely to be executed and discounts
   extreme gaps on sources that almost certainly commit.

Always retain distribution quantiles. A rising mean can mean that every source
became moderately target-dependent, or that a small tail became extremely
target-dependent; those are different policy behaviors.

## Fixed-corpus audit: binary marginalization experiment

The July 14, 2026 audit used a checkpoint-independent corpus so every checkpoint
saw the same states:

- 16 diverse panel boards generated by Ajay versus Ajay.
- Both seats sampled every 10 environment steps.
- 1,068 states and 6,586 actionable source planets.
- Checkpoint target legality, reinforcement empire gate, and binary commitment
  feasibility were applied.
- Reverse-edge cooldown was not applied because Ajay-versus-Ajay trajectories
  do not contain the model's action history. This is the main audit limitation.

The fixed corpus is intentionally off-policy. Its absolute values can differ
from on-policy W&B metrics, but it provides the cleaner checkpoint comparison.

| Run and checkpoint | mean gap | NOOP-weighted | median | p90 | p99 | flip probability |
|---|---:|---:|---:|---:|---:|---:|
| Old factorized 5M | 0.0264 | 0.0112 | 0.000504 | 0.0416 | 0.662 | 10.44% |
| Old factorized 10M | 0.0652 | 0.0204 | 0.000069 | 0.0786 | 1.593 | 10.73% |
| Exact marginalization 5M | 0.0391 | 0.0133 | 0.000338 | 0.0563 | 0.833 | 8.76% |
| Exact marginalization 10M | 0.0620 | 0.0165 | 0.000005 | 0.0285 | 1.649 | 8.85% |

Within the exact-marginalization run from 5M to 10M:

- Mean gap increased by 59%.
- NOOP-weighted mean increased by 24%.
- Median, p90, and the fraction of sources above 0.01 fell.
- p99 approximately doubled.
- Fixed-corpus flip probability stayed essentially flat.

Target dependence therefore became **sparser but stronger in the tail**. The
unchanged flip probability also demonstrates why the Jensen gap is
complementary: substantial probability changes need not create more threshold
crossings.

At 10M the old control had a slightly higher mean and NOOP-weighted gap than the
exact-marginalization run, despite a lower Ajay win rate (`9.0%` versus `16.8%`).
This is important evidence against treating the gap as a performance target.
The architectural fix is valuable because it gives PPO the correct executed
action likelihood, not because it necessarily makes the gap larger.

## How to read future runs

- **Gap near zero:** fire is nearly target-independent. This may indicate that
  target conditioning is unused, but can also be correct in states where every
  target should lead to the same NOOP/COMMIT decision.
- **Mean and quantiles rise together:** target dependence is becoming broadly
  stronger.
- **Mean or p99 rises while median/p90 fall:** dependence is concentrating in a
  small tail, as in the 10M marginalized checkpoint.
- **Mean rises but NOOP-weighted mean does not:** the strongest dependence is on
  sources that rarely execute NOOP.
- **Flip rises without better conversion or retention:** the policy is changing
  hard decisions more often, but not necessarily making better ones.

For model selection, pair this diagnostic with the outcome-grounded north-star
metrics: opening capture/attack, planets at step 50, peel rate, end territory,
and loss depth. Jensen gap explains policy structure; those metrics determine
whether the structure produces stronger play.
