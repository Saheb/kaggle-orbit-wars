[www.kaggle.com /competitions/orbit-wars/discussion/713354 https://www.kaggle.com/competitions/orbit-wars/discussion/713354](https://www.kaggle.com/competitions/orbit-wars/discussion/713354)

Orbit Wars
==========

7-8 minutes

* * *

Hi everyone,

Some of you may be interested in a relatively strong 100% rule-based agent: no reinforcement learning, no neural networks, and no imitation learning.

My agent reached roughly 1300 Elo and should finish somewhere between the Top 50 and Top 100.

While RL clearly dominated the leaderboard, I was impressed by how far a carefully engineered planning agent could go. Even though it was a lot of fun to build, I would probably choose an RL approach if I were starting over today!

Here are some of the key ideas behind it.

* * *

1\. Environment Optimization
----------------------------

The first challenge was performance.

Orbit Wars is a game where evaluating future consequences is extremely valuable, but simulation is expensive. A large part of the work therefore consisted of making computations as cheap as possible.

Most computations are heavily cached, including:

*   future planet positions
*   collision paths
*   travel time approximations
*   trajectory evaluations
*   planet pair information

I also take advantage of map symmetries whenever possible to avoid recomputing equivalent situations.

* * *

2\. Early Game Optimization
---------------------------

The early game is extremely important because production compounds.

Rather than greedily capturing the closest or most profitable planets, I implemented a dedicated search that maximizes the total number of ships produced over a future horizon (generally between 30 and 60 steps).

The horizon is dynamically adapted according to the map geometry, planet costs, available production and other characteristics of the map.

The search explores different capture sequences and evaluates them according to future production rather than immediate gain.

Many games are essentially won or lost during these first decisions.

Running on a single CPU, this search can be expensive and can take up to 25 seconds for complex maps, although it usually completes in less than 5 seconds.

* * *

3\. Planet Forecasting
----------------------

A large part of the code is dedicated to predicting the future.

For every planet and up to 20 to 30 future steps ahead, the agent computes:

*   future positions
*   future ownership
*   future ship counts
*   incoming fleets
*   how many ships can realistically arrive from each neighbouring planet
*   how many ships can safely be sent away

This allows the agent to reason about situations such as:

> "If I attack this planet now, what will it look like 15 turns later?"

instead of only looking at the current state.

* * *

4\. Scenario Simulation
-----------------------

My agent continuously builds future scenarios and evaluates how good they are.

One design choice that worked well was to think target-wise rather than source-wise.

Instead of asking:

> "What should this planet do?"

the agent asks:

> "What can I do with this target?"

This naturally allows multiple planets to cooperate toward the same objective.

For each candidate target, the agent searches for six categories of plans:

*   neutral captures
*   defenses
*   reinforcements
*   strong attacks (very likely capture)
*   weak attacks (no capture)
*   sniper opportunities (capture a planet right after an ennemy paid the cost of it)

While strong attacks guarantee a capture, weak attacks do not necessarily capture a planet immediately. Their purpose is often to weaken an enemy position, delay an expansion, or create future opportunities.

Overall, this usually results in anywhere between 0 and 100 candidate plans to evaluate every step.

* * *

5\. Evaluation Function
-----------------------

For each scenario, the agent evaluates both:

*   the value gained by acting on the target
*   the value lost by weakening the source planets

The evaluation is strongly production-driven.

Rather than focusing only on ships gained or planets captured, the agent tries to estimate the long-term economic impact of a move.

Among the factors considered are:

*   production gained or denied
*   vulnerability of the target and source planets
*   nearby allies and enemies
*   future reinforcement potential
*   travel distance
*   execution delay
*   ownership stability

The objective is not to maximize immediate profit, but to maximize future strategic value.

This often leads to counter-intuitive decisions such as delaying a capture, reinforcing an ally first, or refusing an apparently profitable attack because it would create a weakness elsewhere.

That being said, the evaluation function is certainly the weakest part of the agent. When reviewing replays, it is often easy to identify situations where the wrong plan was selected because it received an incorrect score. Most of my improvements throughout the competition came from analyzing these mistakes and refining the scoring function accordingly. However, this process is endless and therefore time-consuming ! Because there is always another edge case to fix, another scoring rule to tweak. At some point, the approach was reaching its limits against strongest RL-based agents.

* * *

6\. Delayed Execution
---------------------

One aspect of the agent is its ability to reason about actions that should be executed in the future rather than immediately.

At every turn, future plans compete with immediate plans. If a future opportunity is estimated to generate significantly more value, the agent may deliberately postpone its action.

In practice, this means the agent may voluntarily wait, not because no good move exists, but because it expects a significantly better opportunity to appear a few turns later.

For example, it may wait for:

*   additional production to accumulate
*   reinforcements to arrive
*   an enemy planet to become vulnerable
*   a trajectory to become available
*   a coordinated multi-planet attack to become feasible

This planning mechanism helps avoid many short-sighted decisions and often produces more efficient expansions and attacks.

* * *

7\. Reinforcement Chains
------------------------

One feature that helped was multi-step reinforcement planning.

Instead of only considering:

Planet A → Planet B

the agent also searches for opportunities such as:

Planet A → Planet B → Planet C

This often allows productive planets to be captured significantly earlier than with direct attacks.

Since production compounds over time, arriving even a few turns earlier can create a substantial snowball effect.

* * *

What I Think Made the Difference
--------------------------------

If I had to identify the biggest contributors to performance:

1.  Early-game optimization.
2.  Strong future-state simulation.
3.  Production-driven decision making.
4.  Safety constraints preventing overextension.
5.  Delayed execution and long-term planning.
6.  Reinforcement-chain planning.

The final result was an agent that often looks less spectacular than top RL agents, but is difficult to exploit because most of its decisions are grounded in long-term economic value.

It was a fun competition, and building a strong agent without machine learning taught me a lot about planning, simulation, and game strategy.

Next time, though, I'll definitely try reinforcement learning !

Thanks to the organizers, and congratulations to everyone who participated !