"""Reverse-edge reinforce cooldown — THE canonical rule, shared by train (torch_env),
eval (action_mask), and export so all three masks are byte-identical.

Pathology (project_reinforce_pingpong): reinforce waste is a TEMPORAL A->B->A ping-pong —
strong agents almost never reciprocate (rank1 recip<=3st <0.01; we sit 0.06-0.10). This is a
veto with guardrails, NOT the out-massed wall fix: judge it first by recip falling / reinforce
hygiene, watch garr@loss / abandoned / too-late don't worsen (feedback_veto_mask_removes_not_teaches).

Rule (v1 — deliberately NO threat exceptions; exceptions risk preserving the bug):
  - Applies to OWN-target reinforce ONLY. Attacks / captures / recaptures are never blocked
    (they are not own-target, so the caller never even consults this for them).
  - Recording a reinforce A->B sets last[(A,B)] = t.
  - A candidate reinforce s->d is BLOCKED iff the REVERSE edge d->s fired within K steps:
        (t - last[(d, s)]) <= K
  - OWNERSHIP-CHANGE RESET (better than a static "skip if A is enemy" exception): when a planet
    p leaves the player's ownership, clear every edge touching p. So if A flips away and is later
    retaken, a stale A->B record can't block urgent reinforcement into the just-recaptured A.
  - EPISODE RESET: clear all state (same bug class as the decmass prev_decisive_suff re-arm —
    if state survives a boundary it mis-blocks the fresh game).

Keyed by planet IDENTITY (the env's planet id), not list position, so reordering can't corrupt it.
In torch_env, planet index == id within an episode, so the vectorized (N, players, P, P) mirror
of this rule is index-keyed; eval/export key by planets[:,0]. test_reinforce_cooldown locks the
edge/window/ownership/boundary semantics; the torch_env mirror is checked by its own boundary test.

NOTE: applied as a mask in BOTH train and eval, recip<=Kst is forced to ~0 by construction (the
reverse edge is literally illegal for K steps) — so a low recip post-lever is tautological, not a
learning signal. Read the guardrails (reinforce-rate, garr@loss, abandoned, too-late, WR); to test
INTERNALIZATION, probe native recip on the UNMASKED policy separately.
"""

NEVER = -(1 << 30)   # "edge never fired"; (t - NEVER) dwarfs any K so it never blocks.


def is_blocked(last: dict, t: int, src_id: int, dst_id: int, K: int) -> bool:
    """True if reinforce src->dst must be vetoed: the reverse edge dst->src fired within K steps."""
    return (t - last.get((dst_id, src_id), NEVER)) <= K


def record(last: dict, t: int, src_id: int, dst_id: int) -> None:
    """Register an executed own-target reinforce src->dst at step t."""
    last[(src_id, dst_id)] = t


def on_ownership_loss(last: dict, planet_id: int) -> None:
    """Clear every edge touching a planet we no longer own (recapture = clean slate)."""
    for k in [k for k in last if planet_id in k]:
        del last[k]


def reset(last: dict) -> None:
    """Episode boundary: clear all cooldown state."""
    last.clear()
