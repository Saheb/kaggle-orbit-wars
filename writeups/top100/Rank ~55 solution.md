[www.kaggle.com /competitions/orbit-wars/discussion/715147 https://www.kaggle.com/competitions/orbit-wars/discussion/715147](https://www.kaggle.com/competitions/orbit-wars/discussion/715147)

Orbit Wars
==========

8-11 minutes

* * *

_I'm going to skip the stuff every other writeup already nailed, like the obs design, all-in actions, self-play, leagues, the env rewrite. These are just the parts I didn't see elsewhere, the things I tried, and what I actually learned._

### First, some homage

Before anything, a tip of the hat to the OG Planet Wars and its #1, bocsim. Reading his and the other post-mortems from that era was a huge source of ideas and inspiration (here's [his writeup](https://quotenil.com/Planet-Wars-Post-Mortem.html)). It's wild how far the bar has moved. A few thousand lines of clever heuristics was SOTA back then. Today even the mid-ranks are running custom architectures, PPO, leagues, imitation learning, custom transformers. So if your final rank wasn't sky high, just know you were competing against the best, in one of the toughest sim competitions, in the most demanding era it's ever had. I'm partly saying this to console myself.

> **My one real takeaway**, something I already knew but fully felt this time. Let the agents do the execution, but never delegate the thinking. Not yet, anyway. It caused me a lot of grief watching it confidently come up with very dumb ideas. Honestly, maybe we should be happy they can't replicate our thinking just yet.

### Why I even did this

My goal was simple. Have some fun, write a bit of code in my free time. I'd never even hand-coded a transformer before this. But I got completely addicted to building a little brain that teaches itself to play a tiny space game. Watching the brain grow up, I kept finding parallels to life. A bit philosophical, but I found it quite elegant. I still think self-play RL is one of the most incredible ideas in ML. It's got me wondering what SPS I'm living at, and what raising the entropy in my own life would do.

**One-liner:** another self-play RL solution that tried a lot of things and failed at a bunch of them.

### My ladder journey

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F76a817e6e24b4cab68bf34dc1252b42c%2FScreenshot%202026-06-27%20at%2022.24.09.png?generation=1782595468011470&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2Fe578c267556ff374c1a642742e7de17b%2FScreenshot%202026-06-27%20at%2022.24.49.png?generation=1782595503001706&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2Ff97a55d4dde9ef142d13887aa17cdae3%2FScreenshot%202026-06-27%20at%2022.25.18.png?generation=1782595527604286&alt=media) _This is basically the whole plot of the story. Solid in 2p, but 4p was clearly my weak spot, under the 25% first-place rate you'd get from chance._

### My arch

4p arm was trained on an L4 for 3 days since I don't have a GPU. Probably $20 or so for final training run, but I spent maybe $100-200 for experiments. Final 2p arm uses BC.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F6bfd0eee38552ef5e0d83188f811d4ea%2FScreenshot%202026-06-27%20at%2022.26.01.png?generation=1782595573680472&alt=media) _The 2p brain. Each planet plus a global token goes through a small transformer, then an autoregressive head builds the turn one launch at a time, and a value head reads off the win probability._

### Agentic dev, to the stars

I spent the first month trying to "mathematically" solve the game, much like [@vincentschuler](https://www.kaggle.com/vincentschuler) did. But the coding kept turning into a pure SWE grind, and training a brain just sounded more fun. It also didn't help that I got thoroughly demotivated the day Producer dropped.

The funny part is that for roughly the last month I was travelling, and nearly all my "coding" happened through Claude on my phone. Once the env, tests, and codebase were set up, it was genuinely unreal. I'd talk to my phone, launch three experiments, go swim in the sea, and check the results two hours later. I leaned hard into fully agentic dev with a remote L4 on Google Cloud and my laptop (spot instances are super cheap!).

I experimented with fully autonomous runs too. I'd set up a strict scaffold, like directory structure, sprint logs, and decision rules, then let Claude experiment on ideas and params on its own for 12 hours or so. It mostly reinforced my takeaway. Brilliant at execution, genuinely dumb at ideas. It surfaced a handful of usable threads that I then had to build out and validate properly myself.

Where agents really shined was debugging and analysis, and two habits paid off enormously.

1.  **An adversarial agent for everything I built.** It would hunt for flaws, cross-check against my past sprint learnings (all indexed as prior art), and even pull context from the Kaggle discussions. Its decisions were much better for it.
2.  **Curated discussion and deep data analysis.** I'd hand it a hand-picked set of great threads (ty [@lightmk](https://www.kaggle.com/lightmk) !!, for instance) and let it use the Kaggle APIs to mine my own games for weaknesses. That's literally how I found out I was embarrassingly bad at 4p, then drilled into which seat was losing and why.

### Local evals: the only thing that tells you the truth

The first real thing I built was a solid eval and observability setup, because it's the only honest signal of an agent's true strength. The public ladder is a noisy random walk where single games can move you anything from a few points to nearly a hundred. My visualizer took in any agent and did a few things.

*   It kept a local leaderboard, where every bot I made plus all the strong public bots played, like a mini Kaggle ladder. I kept the top 50 or so.
*   It ran round robin so I could read head to head matchups and kill the rock paper scissors trap. I'd only promote a bot once it comprehensively beat all the stronger priors, including its own exploiters.
*   It visualized the attention and decision-making of the policy.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F52977fae6916867d26d7624955baa205%2FScreenshot%202026-06-27%20at%2020.51.01.png?generation=1782595629685964&alt=media) _The local ladder. Every bot I made and every strong public bot lived here and played each other, so I always had a private leaderboard that wasn't at the mercy of the noisy public one._

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F2f427196cce47fc92d72e473461733c2%2FScreenshot%202026-06-27%20at%2021.31.38.png?generation=1782595653930220&alt=media) _The round robin view, which is how I killed the rock paper scissors problem. A bot only got promoted once it cleanly beat all the stronger priors, not just one of them._

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F87ca35ce9883722e961432c8bd7f1a55%2FScreenshot%202026-06-27%20at%2022.27.56.png?generation=1782595686110123&alt=media) _I could also peek inside the policy itself. Soft rings are where it's paying attention, the bold arrow is the launch it committed to, and the dotted ring is a planet it chose to hold._

That overlay caught my favourite bug of the whole comp. Before I had a proper HOLD action, my policy learned to "do nothing" by deliberately aiming at planets it couldn't reach, on the far side of the sun, so every launch just got dropped as infeasible. It had basically hacked the action space to express restraint. I'd never have spotted it without the overlay.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2F4b228c0bb4e12ba1b2b354ffbf0b6794%2FScreenshot%202026-06-27%20at%2022.28.26.png?generation=1782595715849619&alt=media) _Aiming in this game means leading a moving target to where it will actually be, around the sun. The buggy policy abused exactly that, firing across the sun so the shots were thrown out and it "held"._

### A few things I tried

**A first-class HOLD head (autoregressive).** Like a lot of people, I had the classic disease of too few HOLDs and loads of scattered little launches. Inspired again by AlphaStar, I gave the policy a proper autoregressive head. It emits launches one at a time, first the source, then the target, then the amount, and fires a STOP token when it's done. So "do nothing" and "do exactly these two things" become first-class ordered decisions instead of independent per-planet coin flips. It cost me a little training throughput, but the decisions got a lot cleaner, similar in spirit to Ender. I liked that his micro-step version could launch multiple times from the same planet. I didn't factor that into mine.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2825452%2Fec8dc33786135c34c49d19ef753b43be%2FScreenshot%202026-06-27%20at%2022.29.11.png?generation=1782595776593532&alt=media)

**Exploiters.** In the final stretch I played with exploiters (à la AlphaStar). The idea is to train a fresh model purely to beat my best one, and it did, within about a million samples. On its own it tends to overfit to that one opponent. What really helped was training the exploiter back into its league, which stabilized it and actually improved the main bot sometimes. I didn't take it far, but I can see its use. In 4p, plain self-play settles into a passive survivor equilibrium and just stops improving, and the exploiter was the only thing that broke it, worth about 15 to 17 points of first-place rate in my tests.

### What I learned, and what I'd do differently

Honestly? I learned a ton. About machine learning, about agentic coding, and somehow about life too. And I'm happy with all of it. I came in to mess around in my free time and came out with a much clearer sense of how I'd attack the next one of these. That alone made it worth it.

Am I thrilled I only landed around rank ~55? No, not really. But that happens sometimes, and you move on.

The harder question is what was actually lacking, and I'll be honest, I don't really know. Maybe I needed to engineer a lot harder, the way [@sinkingpoint](https://www.kaggle.com/sinkingpoint) squeezed a top-10 finish out of a tiny model and a $170 budget. Maybe it was just about scale, the way [@pressman1](https://www.kaggle.com/pressman1) or [@simjeg](https://www.kaggle.com/simjeg) went big. Maybe I simply kept bolting on machinery instead of letting one clean run cook for long enough. From the inside it's genuinely hard to tell which of those it was, or whether it was all of them a little.

So I'll throw it open to the room. Does anyone else finish a comp not really knowing what they did wrong, or how they'd actually improve? I'd love to hear it.

### Closing

I came in to write a little code for fun and ended up watching a brain teach itself. It's truly been a pleasure to be part of this, and learning from everyone. Can't wait for the next one - Will secure gold next time for sure!

If there's one line to leave you with, it's the one I started with. _Let the machines execute, and keep the thinking for yourself. At least while we still can…_

gg all 🚀