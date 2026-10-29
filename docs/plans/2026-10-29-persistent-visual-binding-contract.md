# Persistent public visual binding: proposed deployment contract

This changes representation support around one frozen learned relational policy. It adds no fit, reward shaping, semantic object role, external detection, or analytic motion model. The original observed-only endpoint and its source remain frozen. Native work requires the parent’s separate review.

## Public input and existing lifetime

Use `latest_encoded['entities']`, rather than the six-field compact scene. Each public entity already provides `id`, `normalized_position[2]`, `observed_velocity[2]`, `observed`, `velocity_known`, `observation_age_seconds`, `velocity_age_seconds`, and `association_strength`. IDs are visual association identities, not game identities. Read existing positions directly; do not advance them again or fill in missing entities.

`vision/state.py` gives a new public ID to a selected lineage, retires its old ID on lineage discontinuity, and clears retired histories. `vision/association.py` retains confirmed hypotheses through age 64 camera samples, while tentative hypotheses last at most 2. `vision/memory.py` publishes confirmed entities through age 64 and carries positions using its existing learned association correction. Runtime `VisualStateEncoder(64,64,.02)` converts ages to seconds; the existing maximum lifetime corresponds to 1.28 seconds. `state.compact()` removes expired/empty hypotheses. The controller relies on whether its exact IDs are currently published and adds no lifetime, confidence threshold, or new extrapolation. Existing association strength can decay while a row remains published; survival is not an accuracy guarantee.

## Deterministic binding and fallback

1. Validate unique public IDs, finite normalized coordinates and velocity, and Boolean observation flags. Invalid entity metadata clears binding and uses baseline.
2. On every scene with exactly the frozen policy’s `n` observed entities, apply its original horizontal content ordering `(x,y,vy,vx,observed,known)` and bind selected ranks `i,j` to their current public IDs. Rebind on this rule even if a genuine full-observation ordering change selects different IDs. The resulting score/action must exactly equal the original policy with the same previous actual command.
3. With fewer than `n` observations, retain the last selected ID pair only if both IDs remain published. Supply their current public positions and observed velocities to the unchanged relational formula. Unknown velocity remains zero. Do not require either entity to be currently observed. Do not sort remembered ghosts into replacement ranks.
4. If either selected ID disappears, clear binding and use baseline until another exact-`n` observed scene can establish a pair. More than `n` observed entities also clears binding and uses baseline. There is no guessed successor identity or proximity reassignment.
5. Update bindings on every public camera, including cameras where the opening gate forces baseline. Record the actual chosen baseline/fallback/servo action only after its real transition; hysteresis always uses that previous actual command. Reset all binding/reward/action state on a true episode reset.

The prospective controller API is `choose(public_entities, baseline_action) -> (action, decision)` and `observe(hits, natural_terminal=False)`. The public entity list is mandatory; six-field archives cannot reconstruct public IDs. Expose `past_hit`, `previous_action`, `pending_action`, `ordinal`, and `bound_ids` for deterministic audit. Frozen parameters include the original policy/scales, this binding-rule version, and the fixed opening-gate mode. The controller state is not learned model state.

Decisions record bound IDs, current observed count, each selected entity’s observed flag/age/strength, rebind/continue/fallback reason, score, actual previous command, and baseline action. No extra feature enters the learned score.

## Separate prospective deployment arms

The same binding mechanism can run with the current actual-first-own-hit gate or from the first supported observation. The former preserves baseline commands through the first actual hit; the latter uses feedback as soon as binding is available. Binding state updates identically in both arms. The two arms must be fixed before their comparisons; opening performance cannot be inferred from a gated winner. Any retained learning procedure should then be independently relearned and qualified on unused starts.

Implemented only in new `scripts/experimental/pong_relational_persistent_binding.py` with focused checks in `tests/test_relational_persistent_binding.py`. Together with the immutable original-core checks, 18 tests pass, including 240 random full-observation exact score/action comparisons, public-ID renaming invariance, partial-observation continuation without invented coordinates, retirement and excess-cardinality fallback, deterministic full-observation rebind, episode reset, both opening gates, actual-command hysteresis, and serialization/fingerprint parity. A production `VisualStateEncoder` output compatibility check uses no environment or image acquisition. No current native or training run is authorized by this document.

Supporting TRAIN-only continuity measurements and exact hashes: `/workspace/m1b-experiments/pong-relational-archive-review-20261029-r1/analysis.json`; interpretation: `docs/reconstruction/2026-10-29-relational-sensory-review.md`.
