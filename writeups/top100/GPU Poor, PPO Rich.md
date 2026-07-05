[www.kaggle.com /competitions/orbit-wars/discussion/714226 https://www.kaggle.com/competitions/orbit-wars/discussion/714226](https://www.kaggle.com/competitions/orbit-wars/discussion/714226)

A Sample Rule-Based Agent Iono's Deck
=====================================

kiyotah 22-28 minutes 6/17/2026

* * *

Introduction
------------

First of all, many thanks to the team working behind the scenes for creating such an interesting and enjoyable competition. Whenever an issue or question appeared, it was handled quickly. I would also like to thank all the competitors who shared ideas and helped create such an active and competitive environment.

My main goal was to learn more about reinforcement learning while combining different neural-network ideas if suitable during the process. RL seemed to have a realistic chance of producing reasonably strong gameplay, so the objective felt nicely aligned: **learn as much as possible while still having a chance to compete.**

This is my **second time doing RL; in my first used SARSA for a university dice game competition**. One early lesson was how difficult it can be to make RL work properly in this kind of more complex states.

I am looking forward to reading the solutions from more experienced competitors and seeing how they approached the same problems. Previous competition writeups helped me a lot, so here it's my own!

Minimal Game Description
------------------------

Orbit Wars is a 2P or 4P game where players fight for control of planets by launching ships from planets they own. Everyone acts simultaneously.

A game ends when only one player has ships remaining or when the 500-step limit is reached. At the limit, the winner is the player with the largest total number of ships, counting both garrisoned ships and fleets in flight.

A surprising amount of complexity comes from map variation, planetary production, static and orbiting planets, changing route availability, fleet-size-dependent speed, and everyone acting at once.

Final Submission at a Glance
----------------------------

The final system was built around a few main ideas:

*   **Simplified Action Space:** Each planet decided whether to commit all of its ships to a target or no-op.
*   **Projected future state:** Each planet token included a 23-step no-new-actions projection.
*   **A graph-biased transformer, or something close enough:** Planet tokens, a strategic token, and a value token were processed by a compact transformer biased by a compact set of pair features.
*   **A relational action head:** Each source-target score came from their interaction.
*   **PPO with pure self-play in JAX:** This was the setup I found easiest to train, debug, and trust.

**Most development was done on my local RTX 3070 Ti with 8 GB.** For the final runs, I rented two Vast.ai RTX 5070 Ti instances for about USD 25 total. They reached roughly 750M and 350M steps (one run was heavier than the other), although the submitted checkpoint was from around 412M. More on that slightly painful story later.

Environment Optimization
------------------------

To get high throughput, I used the fact that player actions do not change the planets’ underlying movement. Planet positions and route geometry could therefore be calculated ahead of time for all 500 steps.

For each map, I precomputed legality information indexed approximately as:

    [source, target, step, speed_bin]
    

This made route checks, masking, and several route features much cheaper, at the cost of cache memory for having precomputed diverse maps.

It was a trade I was happy to make. My GPU was not enormous, but at least it spent more time training and less time rediscovering where the planets would be, route trajectories, collisions, etc.

The environment, rollout collection, inference, and PPO updates were implemented end to end in JAX.

Action Representation
---------------------

Per-source Decisions
--------------------

Each of the 44 regular planet slots acted as a potential decision unit.

For every source slot, the policy produced logits for:

    44 target-slot indices + 1 no-op = 45 output choices
    

The network output had shape:

    [B, 44, 45]
    

This was **not** one categorical action space containing `44 × 45` choices. The policy sampled one categorical action for each valid source, so the final action was a vector of parallel decisions.

A source could also never have all 45 choices available. Its own target index was always illegal, giving at most:

    43 other targets + 1 no-op = 44 legal choices
    

Usually there were fewer because some planet slots were inactive, some targets were unreachable, and some routes were invalid for the current all-in fleet size.

Launch Behavior
---------------

Selecting a target launched the source’s full available fleet. Selecting no-op meant that source did nothing.

A target action was legal only when:

*   source and target slots were active (basically planets that exist in that map);
*   the source belonged to the current player and the target was not the source (logically);
*   the source had at least six ships (decided 6 because smallest spawning neutral planet had 5 and also speed gains at low ships counts are massive);
*   a valid route existed for the current all-in fleet size.

Inactive and invalid actions were masked with a very negative logit before sampling or greedy inference.

The all-in formulation was intentionally simple. It was not the most expressive action space I tried, but it was the one I could actually train and debug with some confidence.

Comet Handling
--------------

Comets were excluded from the learned action space. Otherwise they would have added roughly 20 more controllable or targetable bodies during the steps when they appeared.

They were not removed from the environment, though. They still existed in the simulation and affected route geometry.

Because some geometry calculations traded a little precision for speed, a comet could occasionally be reached or captured even though the policy was not meant to control it. When that happened, the named by me **un/lucky-comet rule** took over: on the comet’s final actionable turn, it launched its ships toward the nearest planet target. This handling is far from perfect, but it is better than letting ships disappear with the comet and in LB is does introduce some un/lucky situations!

**Why comets are not important unless you are competing at elite?** We could argue that given comets are around only for 30 turns, you should only "purchase" them if its cost is below 30 without taking into account ETA to them, the cost of not using those ships for something less, and that it can be contested if you marginally capture it (like by 1/2 ships). The speed of it and its trajectory to use it as a "bus" to get somewhere else your ships is another thing. So yes, it should not be harmful not to use them.

Feature Engineering
-------------------

Planet Features
---------------

Each regular planet slot was represented as one token.

Base features covered ownership, production, ship count, static or orbiting status, centered position, distance to the center or sun, and approximate fleet speed.

These described the planet now. The more important addition was what was already going to happen to it.

Projected Planet Timeline
-------------------------

For every planet, I simulated its known future state assuming nobody made any new launches. The projection included production, fleets already in flight, future arrivals, combat resolution, and ownership changes caused by those known events.

For each future step from 1 to 23, the model received:

*   `is_self`, `is_enemy`, and `is_neutral`;
*   self, enemy, and neutral ship counts.

Before this, I tried simpler summaries such as incoming-fleet buckets. The explicit timeline was one of the biggest improvements I found.

It made timing explicit: a planet becoming enemy-owned in three steps is not the same as one changing in twenty, even if both look identical now.

The model was mostly blind beyond this horizon, although shorter projections may be more stable when new actions can quickly invalidate distant predictions.

Pair and Edge Features
----------------------

My initial mental model was closer to a graph neural network: planets as nodes and source-target relationships as edges.

The submitted model did not perform explicit pairwise message passing. Instead, compact directional pair features were used:

1.  as learned attention biases;
2.  as direct inputs to the action head.

The final features were distance, route existence, short-term change in distance, all-in ETA, and whether the all-in action was legal.

This kept the useful part of the graph idea: geometry, direction, route awareness, and travel time, without paying the full memory cost of storing a large pair tensor throughout PPO.

A nice side effect was that **I could print the learned bias weights for each attention head and try to guess what the heads were doing.** Some were strongly influenced by distance, while others learned a more mixed set of weights closer to zero.

I will not pretend this suddenly made the transformer interpretable, but it was fun to inspect and occasionally reassuring to see that the model had noticed at least some of the features I spent time building.

Map and Global Features
-----------------------

The map profile summarized planet counts, static and orbiting composition, production, route density, and all-in ETA statistics.

Some of this could be inferred from the tokens, but the same local state can mean different things on different maps. A production-3 planet is not equally valuable on a dense high-production map and a sparse low-production one.

The map profile was projected into an embedding and added to each planet token through a learned residual scale initialized near `0.1`. The idea was to start with almost no effect and let the gradients decide whether the information was useful, or quietly push it back toward zero.

Printing these scales was another small diagnostic I enjoyed. It was a way of asking the model, “Did you actually use the feature I spent all afternoon engineering?”

Global features summarized the step, projected production and score, and their differences. They fed the strategic token and critic.

Architecture
------------

Here comes the fun part, or the not-so-fun part when it refuses to learn anything.

The final policy was a compact graph-biased transformer.

I am still not completely sure whether I am allowed to call it a **graph transformer**. There was no explicit edge-message-passing layer, and pair information entered the transformer only through attention biases. On the other hand, attention was relation-aware and the action head was strongly pair-based. “Graph-biased transformer” feels safer, but perhaps I can be forgiven either way.

At a high level, the model processed:

    planet_tokens + strategic_global_token + learned_value_token
    

The submitted configuration was:

*   `d_model = 128`;
*   three transformer layers;
*   four attention heads;
*   32 dimensions per head;
*   **around 750K parameters**.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8540276%2F772eb7bc6447dc77967c221e586d2a7b%2FScreenshot%202026-06-25%20at%205.00.16PM.png?generation=1782417665142901&alt=media)

My apologies for the slightly overloaded diagram. I tried quite a few interactions with image-generation models before arriving to this. The final result is mostly well aligned with the model ;). (I will probably upload an upgraded version soon!)

Feel free to ask if you have any doubts when looking at it. There are some double headed arrows that need correcting. Either way, I think the most important ideas are there and explained below.

Input Encoders and Special Tokens
---------------------------------

Each planet token came from two streams. A dense layer encoded the standard planet features, while small 1D convolutions encoded the projected timeline. The two embeddings were added and passed through GELU.

I originally used a simple MLP for the timeline. My intuition was that Conv1D might smooth nearby projected states while still preserving when events occurred. I cannot claim this came from a perfect ablation study. It seemed reasonable, trained well enough, and survived until the final model, which in this project was already quite a strong selection criterion.

The map profile was softly added to planet tokens, while the global timeline used small temporal convolutions.

The transformer also included a **strategic token**, shifted by global features through a learned gate, and a **value token** for the critic learn from zero. I preferred a dedicated value token over hoping that a simple average contained exactly the summary it needed. This for obvious reasons it's quite limited.

Graph-Biased Transformer Layers
-------------------------------

The model used standard multi-head self-attention with an extra learned bias per head derived from the directional pair features.

This gave each head hints about distance, routes, ETA, and legality. Planet tokens attended to one another and the strategic token, while the value token collected global information.

It was my compromise between a pure transformer and a full graph network.

**Would explicit pair messages have been better? Possibly. Would they also have made PPO storage much larger and my GPU much less happy? Almost certainly.**

Action Head
-----------

I genuinely think this was where I struggled the most.

I was fairly confident in the observation. The harder problem was letting planets decide in parallel while considering other likely decisions and avoiding several sources independently choosing the same excellent-looking target.

For every source-target pair, the model projected the transformer outputs into an as-source embedding and an as-target embedding.

The final interaction combined:

*   source and target projections;
*   `source × target`;
*   `source - target`;
*   `abs(source - target)`;
*   pair features;
*   source and target ship features;
*   the target’s projected state at estimated arrival time.

That last ETA-selected feature is admittedly a fairly strong inductive bias. At that point, however, I was very happy to give the network a hint if the hint worked.

\*\*A compact pair MLP produced one logit per source-target action, while a separate per-source head produced the no-op logit. \*\*Together they formed `[B, 44, 45]`.

This worked noticeably better than a simpler dot-product score. It still did not fully solve coordination: sources communicated before sampling, but their actions were sampled independently.

Value Head
----------

The value head used the value token, strategic token, map embedding, current global features, global projected-timeline embedding, and current step. A small MLP produced the scalar value.

I do not think the critic used as much information from the transformer as it could have. Later in training, explained variance around step 50 dropped quite aggressively.

I am not completely sure how to interpret that. It may have meant missing information or insufficient capacity. But stronger, more symmetric self-play also produced longer games where early positions genuinely stayed close to 50% win probability.

So, as with several things in this project, the honest answer is probably: a little of both.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8540276%2F25175b625350da9a671658f68f61e6bd%2FScreenshot%202026-06-25%20at%203.53.07PM.png?generation=1782413706184443&alt=media)

Training, Self-Play, and Diagnostics
------------------------------------

The final setup used PPO with pure self-play.

Config was:

*   fixed pool size: 384 Maps
*   rollout steps x envs: 32 x 1024
*   minibatch size: 256
*   ppo epochs: 1
*   learning rate: 1e-4 -> 1e-5
*   entropy coef: 0.01 -> 0.001
*   gamma / gae-lambda: 0.993 and 0.95
*   clip 0.2
*   grad 0.5.
*   rewards were purely sparse -1 / 0 / +1 (tried early shaping with production, but did not see results)

Training was always canonicalized to the player 0 perspective. Regardless of the original seat, ownership and player-relative features were remapped so the policy always saw the acting player as player 0 and everyone else as opponents. This kept the observation and value conventions consistent across rollouts.

I kept the five most recent checkpoints and evaluated the current policy against them. Since self-play win rate naturally stays near 50%, I also watched production, score, launch rate, entropy, game length, explained variance, and parameter or gradient norms.

Production reaching a certain level was informative of learning. But then I could not really assess if a +5%/-5% actually revealed more or less strength. Launch rates being so high for sure showed uncertainty and some actions collapse, but they "always" converged to the expected (given stats calculated from leaderbord) to "desired" 0.15.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8540276%2Fbcfcea70a68065f5557ec378c7f160b1%2FScreenshot%202026-06-25%20at%203.54.06PM.png?generation=1782413942071656&alt=media) ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F8540276%2F548ba98626dc5a7366e0d1959ec6d345%2FScreenshot%202026-06-25%20at%203.53.54PM.png?generation=1782413936193103&alt=media)

For model selection, I maintained a reference checkpoint. A new model replaced it after winning at least 80% of 128 games against that reference.

This rule was simple and cheap, which was exactly why I used it. It was also not nearly robust enough, which became clear later.

Still, one result I was genuinely very happy about was managing to win some head-to-head matches against several top-40 players. I am sure at least some of those wins came from favorable or lucky matchups, but seeing my agent occasionally compete with such strong submissions made me proud and gave me confidence that the overall approach had real potential.

Lessons Learned, Things That Did Not Work, and What I Would Try Next
--------------------------------------------------------------------

Everything below is the collection of failed experiments, unresolved problems, and ideas I would still like to revisit.

Factorized Launch, Target, and Fraction Decisions
-------------------------------------------------

Early in the project, I tried a factorized action representation with:

1.  launch or no-launch;
2.  a conditional target;
3.  a conditional launch fraction.

This was much more expressive than the final all-in action space. It was also much harder to train.

Exploration and credit assignment became more difficult, and the policy struggled to beat simple hand-built agents. With several conditional heads, it was also harder to tell whether the problem was the representation, masking, PPO, exploration, or a small bug hiding between them.

I eventually restarted with target-or-no-op and full-fleet launches. That version was easier to debug and finally produced useful learning curves when looking at production at step 50 for example.

Still, I do not think factorization was the wrong idea. Adding launch fractions is probably the clearest next step for expressiveness. I would try it again from a stronger baseline, perhaps with better initialization, autoregressive conditioning, auxiliary losses, or staged training.

Coordination Between Source Planets
-----------------------------------

The main unresolved problem was coordination between parallel decisions.

The transformer let planets share information before action selection, but each source still sampled independently and did not know what the others had selected. This made synchronized attacks, defensive transfers, and avoiding overcommitment difficult.

Autoregressive decoding, pointer-style heads, or a shared latent plan could help, although probably at a throughput cost. Elegant architectures become less exciting when they make training five times slower and the time to see actual progress grows as well.

PFSP and Heuristic Opponents
----------------------------

I experimented with PFSP and a small group of heuristic opponents to increase diversity and reduce overfitting.

In practice, they added more moving parts and made improvements harder to interpret. As the learned policy improved, some heuristics also became too weak, so spending 10–20% of rollouts against them may have slowed training.

I returned to pure self-play because it was simpler and continued to work. A proper adaptive league would still be worth trying, but my small mixed pool was not quite that.

Four-player Fine-tuning
-----------------------

I briefly tried fine-tuning the same model for 4P, grouping all enemies into one channel. It required at least roughly twice the compute, so I quickly returned to improving the 2P policy.

**At least below top-level play, getting better at 2P also seemed to improve 4P results, which made that choice easier.** The main inference adaptation was how to summarize multiple opponents. Using the **mean score and production over active opponents** worked better than using the maximum, which I also tried.

Local Evaluation and Model Selection
------------------------------------

**My local evaluation setup was not robust enough.**

The checkpoint selected internally as strongest was trained for approximately 668 million steps. However, a checkpoint from around 412 million performed noticeably better on the leaderboard. One final run continued to around 750 million steps, so more training clearly did not translate cleanly into a better submission.

Possible reasons include too few games, limited map and seed diversity, local overfitting, non-transitive policies, or leaderboard noise. Whatever the explanation, my evaluation was not reliably finding the best checkpoint.

Next time, a league or rating system would also beat replacing one reference after a single head-to-head threshold.

Experiment Duration
-------------------

I often compared alternatives after roughly 50 million or 150 million steps.

Those runs already felt long, but the final training showed that learning curves could change substantially later. Some ideas that looked weaker early might have had better long-term potential, while others improved quickly and then stopped.

**In retrospect, I would run fewer experiments for longer and compare improvement rate and checkpoint stability, not only current strength.**

Other Directions
----------------

One inference idea I especially wanted to try was a lightweight lookahead. I would sample a few top moves from both the policy and the opponent, evaluate the resulting paths with the value function, and choose the action leading to the best estimated outcome. This might add some planning and coordination without changing the whole policy architecture.

I would also revisit sparse graph messaging, compact orbital encodings, a multiscale projected horizon, occasional learned comet control, and a critic that pools transformer outputs more effectively. Explicit 2D positional encoding, or even 3D encoding with time since planet positions are known, also seems worth exploring.

All of these competed with throughput, rollout memory, implementation complexity, and the simple fact that I had too many things I wanted to try near the end. The un/lucky-comet rule did its job, but I would not describe it as peak comet strategy.

Given that the model **only has 750K parameters**, I think the same general approach could have gone considerably further with a larger model, a stronger and more diverse training league, launch-fraction control, and longer training; potentially on the order of one billion environment steps or more.

Conclusion
----------

To conclude, I am quite happy with the results. All things considered, I ended up learning a lot about RL, best practices and also about patience in this kind of experiments.

A lesson for me was that more expressiveness is not automatically better in reinforcement learning. Even if you keep action space limited and in the long term it could be the difference between top models. Expresiveness in my case made me doubt about many things that I was doing right, but seemed not to work just because of learning signal being very far for example.

There are still several directions I would genuinely like to continue exploring, especially coordinated action decoding and launch-fraction control. Or some autoregressive capabilities.

The factorized launch-target-fraction setup did not work the first time. But I still think it is probably part of the next step toward a more expressive policy. Hopefully with a stronger baseline, fewer hidden bugs, and slightly longer runs I could make it work!

I would also like to know how far with very basic feature encoding and a more capable model or for example autoregressive policy a model can go. I do not like that my solution has so many inductive biases, but may it's the way it's supposed to be…

Sources and Thanks
------------------

I spent a considerable amount of time reading solutions from previous competitions, **especially Lux AI and Kore 2022**, as well as following discussions throughout this competition. Many ideas and small bits of reassurance came from the community, so many thanks to:

*   [@lightmk](https://www.kaggle.com/lightmk): Your RL post, including your experience starting out in a previous competition, was very helpful and encouraging.
    
*   [@pressman1](https://www.kaggle.com/pressman1) and [@ferdinandlimburg](https://www.kaggle.com/ferdinandlimburg) : Whenever I doubted my approach or saw my agent do something questionable. Such as sending 20 ships to capture a neutral planet with exactly 20 ships. I went through your replays (which I almost certain new they were RL approaches) to check whether even very strong and carefully prepared agents made similar mistakes sometimes.
    

**Seeing an all-in-only RL agent near the top also made me think that perhaps a relatively simple approach still had a chance.** That genuinely helped keep me going and first make work something "simple".

*   [@slawekbiel](https://www.kaggle.com/slawekbiel): Understanding the heuristic agents you proposed helped me not only evaluate my own policy, but also develop a better understanding of the game dynamics. That understanding directly influenced several of my engineered features.
    
*   [@bovard](https://www.kaggle.com/bovard) and all the organizers: Thank you again for making this competition possible, handling issues quickly, and creating such an enjoyable environment.