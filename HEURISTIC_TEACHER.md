# Heuristic Teacher Notes

`main.py` is currently an adaptive wrapper around two strong local heuristic
references:

- `candidate_suneet_lb1200.py` is the default policy. It is strong into
  Zach-style slower openings and gives the PPO agent a high-quality expansion
  and midgame teacher.
- `candidate_zach_public.py` is used only when hostile fleets are observed
  before turn 30. This covers Marco-style early pressure, where Suneet waits too
  long and loses its home.

The switch is intentionally simple:

1. Reset per-player mode at step 0.
2. If any enemy fleet with at least 8 ships appears by turn 30, mark that player
   as facing an early rush.
3. Use Zach through turn 90 in that mode, then return to Suneet.

Local seed-0 slot-0 probes after wiring this into `main.py`:

- `main.py` vs `candidate_zach_public.py`: win, material `[5235, 0, 0, 0]`.
- `main.py` vs public Marco: win, material `[2950, 0, 0, 0]`.
- `main.py` vs `candidate_suneet_lb1200.py`: loss, material `[0, 0, 0, 3926]`.

For RL, this is a useful behavior-cloning starting point because it exposes both
delayed/value-heavy expansion and fast anti-rush expansion. The first target for
PPO improvement is the Suneet mirror: learn when to deviate from Suneet after
the opening instead of copying it symmetrically.
