[www.kaggle.com /competitions/orbit-wars/discussion/717226 https://www.kaggle.com/competitions/orbit-wars/discussion/717226](https://www.kaggle.com/competitions/orbit-wars/discussion/717226)

Orbit Wars
==========

10-13 minutes

* * *

Quick note: I had Codex help draft and edit more of this post than I normally would because I am recovering from surgery. The implementation details and project described here are mine; the writeup has more agent-assisted polish than usual.

Thanks to Kaggle and the Orbit Wars organizers for hosting this competition. This was my first serious machine-learning project. I finished with a best public score of `1267.8`, public rank `#99`, and a system that was much more interesting than the final rank: a Rust simulator, compact observation contract, JAX PPO actor-critic, opponent banks, local eval gates, Kaggle package audits, and a late 4-player wrapper.

This was my first dive into ML. I was so far from ML in the past that setup itself was part of the learning curve. I apparently made a poor choice last year when I upgraded my PC to an AMD `7900 GRE`, so the Windows path was awkward and I eventually dual-booted Linux to get a usable ROCm training setup.

I deliberately did not use imitation learning for my main model. I wanted to see how far I could get with self-play PPO and debugging the RL loop directly. I did use leaderboard games as a diagnostic/value-learning reference point, but not as a deployed imitation-learning path.

High-Level System
-----------------

The final useful version split the work into two parts:

*   Rust handled game stepping, movement, combat, observation construction, legality masks, and compact binary frames.
*   Python/JAX handled policy/value inference, rollout collection, PPO updates, checkpoints, diagnostics, and packaging.

This split mattered because the simulator was part of the learning algorithm. If movement, target legality, or observation features were wrong, PPO would optimize the wrong game very confidently.

The final policy was a compact entity-transformer actor-critic with roughly `642k` learned parameters. Each player view was encoded independently. The model saw one global token plus up to `64` planet tokens, then produced target and fraction decisions for a sparse list of legal source planets.

Observation Features
--------------------

The 2-player planet feature contract had `32` base features. They included:

*   active/ownership flags: self, enemy, neutral
*   geometry: `x`, `y`, radius
*   planet state: ships, production, comet flag, rotating flag
*   incoming own/enemy fleet totals
*   own/enemy ETA features
*   incoming fleet buckets: `1-10`, `11-30`, `31-50`, `51+`
*   no-action owner/margin features
*   possible own/enemy arrival windows
*   centroid and rotation features, including distance to own/enemy centroid and whether an orbiting planet was moving toward my control area

For 4-player, I kept the first `32` features for warm-start compatibility and appended `8` relative-player lanes:

`owner_rel_0_self`, `owner_rel_1_next`, `owner_rel_2_across`, `owner_rel_3_prev`, `incoming_rel_0_self`, `incoming_rel_1_next`, `incoming_rel_2_across`, `incoming_rel_3_prev`.

This was a late but important fix. Absolute player IDs are not what the policy should learn. Each seat needs to see the board from its own perspective.

Source-Target Features
----------------------

The model also received pair features for each candidate source-target relation. This ended up being one of the most important parts of the system.

The final pair feature vector included:

*   source legality and target active flags
*   distance
*   ETA for each send fraction
*   full-send speed and ETA
*   whether a sent fleet should be enough to capture
*   projected target ships at full-send ETA
*   projected target owner at full-send ETA
*   remaining production after arrival
*   capture margin
*   whether the route would hit an intermediate planet first
*   projected intermediate garrison and intermediate-plus-target garrison

The key lesson was that "target quality" was not just distance or production. A target was only meaningful relative to arrival time, projected garrison, route collision behavior, and whether another planet blocked the shot.

This was also where some of my biggest bugs lived. At one point I masked sun-blocked targets, but the check used an infinite ray and did not correctly ask whether the fleet would hit another planet before the sun. Later projection and first-contact fixes showed up clearly in diagnostics and on the leaderboard.

Action Space
------------

I factorized the action space in two stages:

1.  For each legal source slot, choose one of `64` target planets or no-op.
2.  Conditioned on that selected target, choose a send fraction or no-op.

The late fraction bins were:

`0.05`, `0.10`, `0.15`, `0.25`, `0.40`, `0.60`, `0.85`, `1.00`.

There was also a fraction no-op. If the target was no-op, the runtime forced the fraction to no-op. If a source slot was dead, both target and fraction were forced to no-op.

The final path used `40` sparse source slots. This avoided scoring a full dense action tensor of roughly `64 sources * 65 targets * 9 fractions` per player. The target head still considered all physical planets as targets; only sources were sparsified.

I also added same-turn proposal context for the fraction head. If several sources selected the same target, the model received features like proposal count, total proposed ships, this source's share of proposed ships, min/mean/max ETA, and ETA rank. This helped the fraction head reason about multi-source attacks instead of treating every launch independently.

Architecture
------------

The encoder was a small transformer:

*   planet projection: `planet_feature_dim -> 128`
*   global projection: `9 -> 128`
*   one learned global token
*   `3` pre-norm transformer blocks
*   hidden size `128`
*   `4` attention heads
*   transformer MLP hidden size `384`

The target head combined:

*   encoded source token
*   encoded target token
*   a dot-product source/target score
*   an MLP score over source token, target token, and the `22` source-target tactical features
*   a separate no-op logit

The fraction head ran after target selection. Its input included:

*   encoded source token
*   encoded selected-target token
*   selected source-target tactical features
*   same-turn proposal context

The value head shared the encoder global token and added a small value-only economy correction over ship delta, production delta, planet-count delta, and projected economy score delta.

Training Setup
--------------

The late 2-player PPO setup used:

*   rollout length: `128`
*   source slots: `40`
*   worker count: `20`
*   worker batch: `32`
*   service batch: `640`
*   gamma: `0.99`
*   GAE lambda: `0.95`
*   clip epsilon: `0.2`
*   PPO epochs: `1`
*   value coefficient: `0.3`
*   learning rate: `5e-5`
*   weight decay: `1e-4`
*   grad clip: `1.0`
*   target/fraction entropy coefficient: `0.0`
*   target/fraction sampling temperature: `0.85`
*   economy reward shaping coefficient: usually `0.2`

The opponent mix late in training was roughly:

*   self-play: `0.5`
*   active checkpoint: `0.2`
*   heuristic opponent: `0.2`
*   opponent bank: `0.1`

Across Orbit Wars 3.0 artifacts, I have about `32k` PPO summary files, `3.5B` recorded player-step samples, `3.7k` checkpoint files, and `1.2M` local evaluation games. Most of the late useful training ran on Vast instances; local Windows/AMD setup was useful for iteration, but Linux/ROCm was the real training path.

Reward And Value Debugging
--------------------------

The biggest turning point was realizing that my critic was not learning a useful signal.

One failure run, `run_000063...strong_rollback`, had `32`\-step rollouts and top-level value explained variance of `-0.905`. That is not just noisy. It means the value function was worse than predicting the mean.

The fix was a combination of:

*   longer `128`\-step rollouts
*   economy-shaped reward deltas
*   lower/more stable value coefficient choices
*   action diagnostics instead of only loss curves

The economy shaping was simple: estimate position quality from ships plus production over a horizon, then reward the delta. Later shaped runs moved overall explained variance into roughly `0.89-0.92`. One useful late run had median overall EV `0.920` and early `0-31` bucket EV `0.637`.

This changed the project. Before that, I often could not tell whether a policy had degraded, whether the reward was too sparse, or whether my eval was lying. After value diagnostics became usable, policy changes became much less mysterious.

What Actually Moved The Needle
------------------------------

The most visible diagnostic was win rate against the heuristic opponent inside PPO summaries. This was not a formal leaderboard eval, but it was useful because it was cheap and frequent.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F34539132%2Ff5f547b7c25e52b4b985ccabf4d6f93a%2Fheuristic-win-rate-timeline.png?generation=1782925514177335&alt=media)

The filtered rolling median moved from `22.2%` to `89.3%`, peaking around `91.3%`.

The clearest jump came around the projection/legal-action work. Around `run_000029_20260620_183410_phase12g_projectionfix_temp085_bank2_from_run28u312`, the event-window median moved from `49.3%` before to `61.6%` after. Inside the run, raw heuristic win rate moved from `56.3%` to `68.0%`.

The public leaderboard told a similar story:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F34539132%2F582490c878f05abfe4d0e6be5fa24eea%2Fkaggle-score-rank-timeline.png?generation=1782925558565566&alt=media)

Some useful score anchors:

*   early complete submission: `506.6`
*   Rust PPO wrapper fix: `775.8`
*   low-entropy / early PPO package: `925.0`
*   temp/no-op run28 package: `1133.9`
*   projection-fix run29 package: `1178.8`
*   best public package: `1267.8`, estimated public rank `#99`

I would not claim any single commit caused a single leaderboard jump. The improvements were usually bundles: reward/value stabilization, temperature/no-op behavior, projection legality, wrapper fixes, and 2p/4p packaging all interacted.

4-Player
--------

4-player was a late pivot and never became as mature as the 2-player loop.

The 4p model used the same basic architecture but a wider observation contract: `64 x 40` planet features instead of `64 x 32`. The relative-owner lanes helped, and a dedicated 4p checkpoint improved the final package, but 4p training remained much more volatile.

If I could restart, I would make 4p first-class much earlier:

*   relative player features from the beginning
*   separate 4p eval banks
*   a wider opponent league
*   more early-game coverage
*   less dependence on late wrapper fixes

What Did Not Work
-----------------

My first approach was inspired by AlphaZero/MCTS, but it turned into a messy search/regressor system rather than a clean RL system. It still taught useful lessons: action labels need attribution, candidate generation is part of the model, and physics bugs become training-data bugs.

I did not run clean ablations for every improvement. Many changes happened close together because this was a competition sprint: reward shaping, opponent mix, sampling temperature, masks, projection fixes, and package changes were often entangled.

Final Thoughts
--------------

The biggest lesson for me was that RL debugging is systems debugging. The model was only one component. The simulator, action masks, observation contract, reward, rollout horizon, opponent bank, eval gate, and Kaggle wrapper were all part of the learning algorithm.

Thanks again to Kaggle and everyone who shared writeups during the competition. In particular, [Lin Myat Ko's RL writeup](https://www.kaggle.com/competitions/orbit-wars/discussion/697725) was one of the posts that helped me understand what a serious PPO system for this competition could look like.