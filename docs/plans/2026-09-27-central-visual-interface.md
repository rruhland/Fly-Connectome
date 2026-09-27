# M1A.5 central visual interface v1

Keep all entity and forecast records, including every mixture component. Add a
separate sparse transport code; do not claim that pooled features replace the
structured state or prove control utility before M1B. No neural rewrite, labels,
reward, controlled-object identity or game rules enter this adapter.

`VisualStateEncoder(height, width, sample_period_seconds).encode(state, frame)`
returns copied structured records, normalized coordinates/actual observed velocity,
explicit age/availability and sparse indices/values with a versioned fixed layout.
Use max(height,width) for both axes so normalization preserves aspect ratio.
Motion is measured from actual observations only, with elapsed sample time; carried
hypotheses do not become new motion measurements. IDs route histories but are not
feature values. `reset_state()` is required at a real scene reset.

Transport groups: retinotopic observed/inferred support, signed motion channels
and motion-known mask; future mixture-center mass at horizons 1..8 with explicit
off-field mass; directed nearest-neighbor relative geometry/motion; positive and
negative coarse current-image contrast; counts/cadence/missing-context metadata.
The code pools entities symmetrically and uses no slot-specific policy semantics.
Full Gaussian mixtures/intervals remain available; center-bin transport mass is
not calibrated occupancy probability. Neighbor sparsification reports pair counts.
No entities or mixture modes are dropped from structured output. Compute is a
measurement, not a hard cap on learning ideas.

Verify ordering/ID invariance, exact retained mixtures and invertible geometric
normalization, no learning from inferred motion, missing/stale image semantics,
reset behavior and large static visual context. Measure production+adapter latency
and active feature counts at two/eight entities with learning enabled. The adapter
does not supply actions or implement M1B learning.

## Transport v1 layout

The output includes named `(start, stop)` slices into a 4,389-dimensional vector.
Only nonzero indices/values are transmitted. Allocation/encoding itself is not
event-driven. All spatial coordinates use a centered square extent of max(H,W);
off-image checks use the actual rectangular image extent. Features are summed,
not normalized to probabilities across entities.

- `current`: 7 x 16 x 16: observed count, inferred count, positive/negative y
  velocity, positive/negative x velocity, known-velocity count.
- `future`: 8 x 16 x 16 mixture-center mass for remaining horizons 1..8 samples.
- `future_outside`: 8 off-image mixture-center masses. These are not Gaussian tail
  probabilities; full centers, weights and sigma remain in structured forecasts.
- `relations`: 5 x 9 x 9: directed pair count and signed relative velocity channels,
  pooling up to four nearest neighbors per in-image source. Relative geometry uses
  twice the image square extent. Unknown relative velocity contributes no motion.
- `context`: 2 x 8 x 8 positive/negative coarse mean-centered image contrast.
- `metadata`: entity count, observed count, elapsed seconds, current-frame available,
  any-frame-ever available, context age seconds (zero if unknown), off-image entity
  count, represented directed-pair count. Structured `context_known` distinguishes
  unknown context from fresh zero contrast.

Structured motion uses normalized image units per second and retains its measurement
age. Declaring cadence converts units; it does not recalibrate learned forecasts to
a different physical sensor cadence. Preserve the training cadence or re-evaluate
and learn at the new one. Structured mixtures are authoritative, and M1B still needs
the planned code-versus-full-distribution usefulness comparison.
