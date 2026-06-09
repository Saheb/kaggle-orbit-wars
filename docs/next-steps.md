# Next steps / idea backlog

Living doc — efforts and ideas to invest in, roughly prioritized. Started 2026-06-09.
Discipline: **one delta per cloud run.** Status tags: 🔵 in-flight · 🟢 ready-to-build · 🟡 idea · ✅ done · ⏸ parked.

---

## 0. In-flight

- 🔵 **Stage 2 VDN run** — does per-planet credit stop the per-slot collapse? Watch `Vμ` (stays positive?)
  + Ajay/1166 +1/2/3M (matches/beats joint-warm ~9%/~62%?). Early read: `Vμ` positive but very selective
  (`srcs_multi`→0.34) — watch for over-suppression into **passivity**. Decision gates the rest below.

---

## 1. Highest-value, cheap, independent of VDN result

- 🟢 **Dense→sparse shaping anneal** — anneal shaping coefs (First Strike / early-capture) to **0 over
  training**, not just within-episode. Principle: shaping *bootstraps* competence but *creates degenerate
  Nash* if kept; remove the liability once the pool can teach the same thing. Cheap (a coef schedule).
  Best fit to our recurring failure modes. **Top non-VDN pick.**
- 🟢 **rollout 32 + ppo_epochs 1** — a top-LB player's suggestion. Short rollout = local credit (opening/
  tempo emphasis); ppo_epochs 1 = more on-policy. Cheap (flags). Different axis from VDN (horizon vs
  structure) — complementary. Caveat: two deltas; separate if a clean read is needed.
- ✅ **Cross-checkpoint eval panel** (`gpu_run_artifacts/cross_eval/run_cross_eval.sh`) — cycling/forgetting
  detector. **Use it every ~1M during training.** Make it always-on alongside the held-out ladder.

## 2. Diversity / anti-cycling (the structural lever)

Cross-play study finding: our opponent set is a **transitive strength ladder, no non-transitivity** →
fixed heuristics have limited anti-cycling power. Two complementary fixes:

- 🟢 **Distill Ajay → fast neural clone (DAgger)** — adds the *selective-targeting style* we lack; orbit_lite
  can't be GPU-batched so distillation is the only way to pool it. Spec: `docs/ajay_distillation_spec.md`.
  One new tool to build: `dagger_collect.py`. Validate with `study_opponents.py`; success = adding it to the
  pool raises held-out real-Ajay WR.
- 🟡 **Neural exploiters / mini-league** — the principled anti-cycling + novelty engine: train fast neural
  agents whose job is to beat the current main agent, fold into the pool. All-neural → no slow-opponent
  speed problem *by construction*. Heaviest, but the real endgame for "keep finding novel strategies."
- 🟢 **Pool curation** — fold cross-run best selves (rev38/rev53b ready in `preseed_pool/`) as fast `self`
  members; subsample within-run history (not dense recent); keep self-pool **small (8–12 diverse)** so PFSP
  gets ≥30 games/opponent. Run as a *separate* delta (not bundled with VDN). `--preseed-pool ../preseed_pool`.
- 🟡 **rev31/rev32b as opponents** — most distinct archetypes (First Strike), but pairwise-12 → need
  `export_agent.py` to auto-detect feature dims. (Spawned task.) Adds real diversity to eval panel + pool.
- ✅ **Hellburner** added to cross-eval — note: it's *surpassed* (our strong agents beat it 100%), so it's a
  fixed **eval yardstick**, not a hard pool opponent.

## 3. Hyperparameter / dynamics probes

- 🟡 **gamma 0.999** (NOT 1.0) — longer horizon propagates the win signal earlier → less shaping dependence.
  Avoid γ=1.0: removes "win fast" pressure → risks the passivity/timeout Nash we keep fighting. Pairs
  naturally with #1 dense→sparse. Horizon changes shock the critic (Rev23) → expect a re-warm.
- 🟡 **SSDR × pool** — SSDR (asymmetric starts) gave only *transient* gains because symmetric self-play
  averages the asymmetry away. Fix: couple it with **fixed strong opponents** so the handicapped seat faces
  a real adversary, not a co-adapting mirror. SSDR-alone = transient; SSDR×pool = potentially sustained.
- ⏸ **Auxiliary losses** — low priority: BC aux *hurt* conversion (Rev34), ROI aux was *inert* (Rev48); they
  help representation/sample-efficiency, not Nash drift (our actual problem). **One exception:** a per-planet
  "will this launch capture?" predictor as a **VDN assist** if the bare per-planet value learns poorly.

## 4. VDN follow-ups (conditional on the in-flight result)

- If **VDN works**: run longer (5M+) to find the peak (rev53b's best Ajay 10.9% was at +3.6M, *past* +3M);
  full Ajay panel; **submit if it beats 10.9%**. Then commit VDN to main.
- If **VDN over-suppresses → passive**: tune (raise fire entropy? scale/clip the per-planet advantage?).
- If **VDN collapses too**: shelve the per-slot direction with a clean negative result; the win is elsewhere
  (diversity / shaping anneal).
- 🟢 **`export_agent.py` VDN tolerance** — needed only when exporting a VDN checkpoint to submit (mirror the
  eval.py `value_pp_*` fix). Do it then.

## 5. Measurement / instrumentation

- 🟢 **Always-on held-out ladder during training** — eval vs a *fixed, diverse* set every ~1M (rev53b did
  1166). Broaden beyond 1166 to span archetypes (a rusher, a turtle, a selective-tempo bot). Absolute-progress
  dashboard; self-play WR and `Vμ`/EV are blind to absolute regression (EV is on-distribution only).
- 🟡 **`V(s)` vs material-score over one game** — intuition-builder demo (where the critic foresees a bad
  commitment before material drops). Pending offer.

## 6. Parked

- ⏸ **FFA / 4p** — above par on wins (the only metric that counts), **no validated lever**, and **no faithful
  local eval** (can't reproduce the LB gang-up). Don't spend GPU. 2p-vs-strong conversion is the headroom.
  Memory: `project_ffa_not_the_gap`.

## 7. Ops / tech debt

- 🟢 **Commit VDN to main** once settled → GCP launcher + reproducibility "just work" (it rsyncs main, excludes
  `.claude/` worktree).
- 🟢 **Bake auto-destroy-on-training-completion into launch scripts** (not a separate poll; not an intermediate
  checkpoint → don't throw away checkpoints we paid for).
- 🟡 **Pool-opponent `state_dict` reload** fires ~123×/iter (revealed by the VDN print spam) — a pre-existing
  rollout-throughput inefficiency. Optimize only if SPS becomes the bottleneck.
- 🟡 **GCP for VDN-lineage runs** needs the launcher to rsync the *worktree* (or VDN committed to main) + the
  24 GB L4 may force `--num-minibatches 32`. Prefer H100/H200/A100 (≥40 GB) for clean mb 16.
