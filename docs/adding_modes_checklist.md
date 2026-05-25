# Adding a new action decode mode (or any action-space feature)

Use this when adding a new flag like `--action-decode`, `--ship-bin-mode`, a new
action head, or any change that touches how actions are sampled, executed, or
interpreted.

**Train/eval mismatch is the primary risk.** The training stack and the
kaggle-submission stack are parallel implementations of the same semantics.
A change that only lands in one side produces a policy trained on X but
evaluated as Y — silently, because nothing crashes.

---

## File map by phase

### Training stack
| File | Role |
|---|---|
| `orbit_wars_rl/train_torch.py` | Orchestration: rollout sampling, buffer storage, batch construction, PPO update loop |
| `orbit_wars_rl/torch_env.py` | `get_features()` — observation → feature tensors (training-side counterpart of `features.py`) |
| `orbit_wars_rl/torch_env.py` | `_apply_actions()` — action tensor → fleet launches (training-side counterpart of `actions_from_policy()`) |
| `orbit_wars_rl/ppo.py` | `compute_loss()` — log-prob computation, PPO ratio, value/entropy losses |
| `orbit_wars_rl/model.py` | `EntityTransformer.forward()` — shared with eval |
| `orbit_wars_rl/config.py` | `ModelConfig` — holds flags like `action_decode`, `ship_bin_mode`, `num_ship_bins` |

### Eval / inference stack
| File | Role |
|---|---|
| `orbit_wars_rl/eval.py` | `build_agent_fn()` — wraps model as a kaggle agent; `evaluate_panel()` |
| `orbit_wars_rl/action_mask.py` | `compute_action_masks()` — counterpart of `torch_env.get_features()` masks |
| `orbit_wars_rl/action_mask.py` | `actions_from_policy()` — counterpart of `torch_env._apply_actions()` |
| `orbit_wars_rl/features.py` | `extract_features()` — counterpart of `torch_env.get_features()` features |
| `orbit_wars_rl/model.py` | `EntityTransformer.forward()` — shared with training |
| `orbit_wars_rl/export_agent.py` | Submission packaging — picks up `ship_bin_mode`, `action_decode` from checkpoint |

### Checkpoint / persistence
| File | Role |
|---|---|
| `orbit_wars_rl/ppo.py` | `state_dict()` config blob — must record every flag needed to reproduce inference |

---

## Checklist

When adding a new decode mode or action-space feature, touch each row. Check
the box only when the change is actually present, not just "probably fine".

### 1. Config
- [ ] New flag added to `ModelConfig` in `config.py`
- [ ] CLI arg added to `train_torch.py` (`--action-decode`, etc.)
- [ ] `cfg.model.<flag> = args.<flag>` wired after arg parsing in `train_torch.py`
- [ ] Flag saved in `ppo.py` `state_dict()` config blob

### 2. Model
- [ ] `model.py` `forward()` signature updated if new inputs needed
- [ ] Shape compatibility guard added (e.g. padding `target_mask` to `max_planets`)

### 3. Training — feature extraction
- [ ] `torch_env.get_features()` computes and returns any new mask/feature
- [ ] Storage buffer allocated in `train_torch.py` for any new per-step tensor
- [ ] Buffer stored during rollout loop

### 4. Training — action sampling
- [ ] `sample_action_batched()` in `train_torch.py` samples new action component
- [ ] New action component stored in rollout buffer
- [ ] New `lp_*` (log prob) stored in rollout buffer
- [ ] Action tensor passed to `env.step()` has the right number of columns

### 5. Training — action execution  ← **where the `action_decode` bug lived**
- [ ] `torch_env._apply_actions()` reads the new action column (e.g. `target_idx`)
- [ ] Ships fly on the angle derived from the new head, not a stale head
- [ ] `VecTorchEnv.__init__` accepts and stores the new mode flag

### 6. Training — PPO loss
- [ ] `ppo.py compute_loss()` builds the right distribution for the new head
- [ ] New `new_log_prob_*` and `old_log_*` computed correctly
- [ ] Heads not executed in this mode have their log probs zeroed (e.g. angle in target mode)
- [ ] Batch dict includes new action and log-prob keys
- [ ] Minibatch loop passes through new scalar/string batch keys

### 7. Eval / inference  ← **where the `target_mask` bugs lived**
- [ ] `eval.py build_agent_fn()` reads the flag from the checkpoint config blob
- [ ] `eval.py` constructs any new mask (e.g. `target_mask`) before the model call
- [ ] `model.forward()` called with the new kwarg (smoke test: one forward pass)
- [ ] `actions_from_policy()` in `action_mask.py` handles the new decode path
- [ ] `features.py extract_features()` returns any new feature the eval path needs

### 8. Submission
- [ ] `export_agent.py` reads the flag from checkpoint and threads it through
- [ ] Submitted agent tested with a 1-step forward pass before any panel eval

### 9. Verification
- [ ] One-step forward-pass smoke test (catches shape errors before 256-game panel)
- [ ] 8-game sanity check: `moves produced > 0` (see eval_runbook.md §1)
- [ ] **Confirm train and eval use the same decode logic** — spot-check: log angle
  of a launched fleet in training vs. angle `actions_from_policy` would pick for
  the same observation and same `target_a`

---

## The canonical train/eval mismatch pairs

These are the files that must stay in sync. A change in one without the other
is the failure mode:

| Training side | Eval side | What must match |
|---|---|---|
| `torch_env.get_features()` | `features.extract_features()` | Feature channels, normalization, padding |
| `torch_env.get_features()` masks | `action_mask.compute_action_masks()` | Mask shape, semantics |
| `torch_env._apply_actions()` | `action_mask.actions_from_policy()` | How action components map to fleet angle/count |
| `train_torch.sample_action_batched()` | `eval.build_agent_fn()` | Which heads are sampled, how masks are applied |
| `ppo.py compute_loss()` | (N/A — training only) | Log-prob must match sampling |
| `ppo.py state_dict()` config blob | `eval.evaluate_checkpoint()` | Every flag needed to reproduce inference |
