# Generic state and observed-dynamics checkpoint

## Scope and decision

Preserve the generic local patch associations, shared-observation identity rule,
and direct horizon-specific dynamics as promising experimental components.
**M1A.5 is not yet accepted.** Production `src/` is unchanged. The successful
sensor arm receives a current visible image every frame, in addition to events.
This is a concrete sensor/architecture revision requiring review before promotion,
not evidence that event-only vision has passed.

These models use engineered spatial grouping and tracklets with locally learned
sensory selectivity and observed displacement associations. They are not a claim
that the measured fly graph has learned these representations autonomously.
No Pong identities, walls, collision rules, hidden masks or backpropagation enter
the models. Synthetic trajectories are scoring targets only; credit comes from
actually matched visible observations.

## Representation evidence

One generic 51-component local patch/direction key replaces the hand-selected
above/below cue. Pre/post covariance learns feature selectivity and a bounded
256-example memory learns displacement corrections from observed reacquisition.

With current visible images, aligned associations score 16/16 contextual top-32
localizations in each of four physical image rotations. Shuffled and frozen
controls score 4/16. Constant and two-mover regressions score 16/16; all continued
cases regain support and none of the vanished cases retain support after reveal.
Fresh event noise does not damage this frame-supported result. Event-only noisy
scores remain inadequate (familiar noise 9,5,8,8/16; heavy noise 2,3,2,4/16).

A different relation encodes the outcome in a stationary motif's **appearance**,
not above/below position. The unchanged learner scores 14/16 top-32 in every
rotation, versus 4–5/16 shuffled and 4/16 frozen. This is bounded evidence for
learned visual relations, not broad natural-image understanding.

An independent 64x64 camera audit has 18 two-entity scenes: three axes, three
offsets and both polarities. The original state swaps some identities at crossing,
especially with missing camera samples. Suppressing unique association credit
when one observed region contains two predicted identities gives 36/36 retained
identities, zero final center error, and zero extra hypotheses under clean,
Gaussian image-noise and five-sample outage conditions. All 180 outage location
probes retain support. This rule makes shared evidence ambiguous rather than
forcing it onto one identity.

The full-frame variant also uses unchanged visible components and a broad first
motion estimate for newborn regions. It passes the same camera audit and tests
stopping/restarting and unavailable-sample handling. Its contextual regression
audit now independently confirms 16/16 spatial-context and 14/16 appearance-context
top-32 in every rotation. Spatial shuffled/frozen controls score 4/16 and 0/16;
appearance shuffled/frozen score 2–3/16 and 1/16. The long audit runtime reflects
the full training/evaluation batch; an isolated 64-frame scene took 0.34 seconds,
with two track slots rather than runaway stationary-region births.

## Dynamics evidence

Training uses 24 observed streams; 36 held-out streams change phase, shape,
direction and scale. Families are constant motion, three periodic displacement
sequences, and clockwise/counterclockwise arcs. Learning uses only actual matched
positions, never propagated tracker estimates. Missing histories receive the
same 64-pixel penalty in every arm. Simulator centers appear only in scoring.

| Forecaster | Mean eight-frame endpoint error |
|---|---:|
| Persistence | 19.01 |
| Constant velocity | 11.00 |
| Two-step replay | 8.53 |
| One-step local learner, rolled forward | 5.46 |
| Same learner with varied training | 7.95 |
| Same learner with causal online updates | 4.72 |
| Direct delayed-credit horizon learner | 3.10 |
| Direct learner, shuffled credit | 17.57 |

Varied training did not resolve the one-step learner's problem; close that branch.
Learning the cumulative displacement directly avoids repeatedly feeding forecast
error back into the input history. It reduces the one-step learner's eight-frame
constant-motion error from 8.77 to 4.22 and arc error remains about 2.59.
However, two-step replay is still better on constant motion (2.28), so the direct
learner has not established no-regression acceptance. Report this limitation,
not only the aggregate gain. Eight-frame online/frozen direct results are equal
because the short scored prefixes end before their first eight-frame training
target arrives; that is not evidence that online adaptation is ineffective.

The full-frame state supplies 1041/1044 forecast histories, versus 976/1044 for
the event-conditioned state. The actual-observation eligibility correction alone
did not change the saved baseline numbers. The coverage gain is attributable to
the changed observation association, not that correction.

A separate local linear-correlation learner was rejected after one configuration:
8.86-pixel eight-frame error, worse than two-step replay. A single shared temporal
linear relation does not express the mixture of motion families adequately. No
regularization or lag-count sweep was run.

## Verification and remaining work

A read-only code review found stale samples could reconfirm base/shared tracks
despite `observation_available=False`. A reproducing test failed before the fix;
proposal generation now respects availability. Previously saved camera scores
used zero events during outages and are not invalidated. Prefix-causality tests
also verify future observation changes cannot alter earlier online forecasts and
evaluation does not mutate trained base weights.

Outstanding acceptance: calibrated uncertain
outcomes, wider camera-domain transfer, bounded streaming memory/throughput,
measured T4/T5 ablation and an explicit integration interface. Strong synthetic
capacity alone does not close these gates. Preserve all unsuccessful branches and
do not silently relabel the current engineering scaffold as production connectome
learning.

Detailed reproducible outputs: `associative-patch-state`, `appearance-relation`,
`camera-state`, `shared-observation`, `frame-observation`, `frame-dynamics`,
`dynamics-*-results` and `direct-dynamics`, all dated 2026-09-26 in this directory.
