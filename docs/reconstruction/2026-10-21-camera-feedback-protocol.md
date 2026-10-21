# Frozen-policy later-camera feedback protocol

Prepared prospectively during the immutable replica, then registered after both
independent training endpoints passed integrity and physics/optimizer audits. Both
fixed endpoints are included. This diagnostic allocates no training or final
holdout observations. Stage is separate from starting the immutable clock.

## Question and fixed policies

Does continuing camera evidence improve these trained policies' return control,
compared with misleading stationary evidence or honest missing evidence after a
fixed opening? This is a causal sensitivity experiment on fixed policies, not a
new acquisition screen or a claim that an optimally trained opening-only policy
cannot control. Neither feedback necessity nor forecast benefit follows merely
from state-dependent scores or prior whole-input blanking.

Include the FIRST curriculum's final endpoint and the independent replica's final
endpoint. Never choose the better seed, phase-zero policy or a best checkpoint.
Include the replica irrespective of its primary-screen pass/fail if its complete
two-phase training and endpoint lineage pass integrity/physics/optimizer checks.
A DEV evaluation deadline by itself does not invalidate correctly completed
training. If its training is invalid/incomplete (<2 complete updates in either
phase, persistent task failure, corrupt lineage or an ineligible endpoint), do
not substitute another policy or seed: report that outcome and run the first
policy only under the same prospective rule, labeling the reduced policy set.
Do not execute this diagnostic on an unverified endpoint. Both endpoints passed
all checks before this allocation was finalized: the registered executable roster
is exactly TWO policies and six cohorts; no reduced-roster fallback is needed.

## Three camera conditions, identical action/control interface

All conditions use the same frozen vision/model, exact T/cap, opponent .65,
greedy hold1 policy, ten-second full rallies and per-camera durability. No policy
or vision updates, task cutoff, labels, object masks or teacher inputs.

1. Live: every genuine image and normally derived events.
2. Warm hold: consume the first TWENTY genuine sensor observations INCLUDING the
   initial observation; thereafter feed the twentieth image repeatedly through
   normal event derivation and vision/encoder calls.
3. Warm missing: same twenty genuine observations, then call
   `vision.step(zero_events, None, learn=False)` and `encoder.encode(state, None)`.
   Do not manufacture disappearance events or ingest hidden subsequent frames.

The twentieth genuine image follows nineteen transitions/45 physics ticks
(.375 s). Action index 19 still uses that common image; observation after that
transition is the first masked observation, and action index 20 is the first
that can differ. Earliest possible first paddle encounter is .79545 s, so the
onset precedes either paddle's first encounter without inspecting hidden state.
All camera ticks/physical time and perception calls remain; no speed advantage
is created by omitting encoding calls. Actual compute cost can still differ by
mode; report per-condition wall/latency/camera costs rather than claiming equal
inference time. Every rally starts a fresh camera phase.

Interpretation limits are integral: held images assert a stationary scene rather
than missing evidence. Missing retains coarse context with age, last observed
velocities and tracking carry, but cached forecasts expire within eight samples
and tracking uses finite age gates (different for tentative/confirmed histories);
this does not promise any particular branch survives to an exact sample. Missing
also sets raw frame-present coordinate
4384 to zero; existing T centers through that bias and was fit on present frames.
Thus the interventions change missingness/conditioning and can be OOD. Never
silently force the bias back to one, fabricate observation presence, refit T or
modify the policy to improve these conditions. A missing-arm loss alone does not
separate feedback dependence from missing-input generalization. Concordant
contrasts strengthen the limited claim about these policies' later sensory use.

## Fresh development cohort and finite cost

Development root **1438291051**, tag **29109**. Check against all current
and prior actually declared training/DEV/calibration IDs when staging, then pin
that exclusion snapshot and the exact recipe BEFORE expansion. Draw sequentially
from `default_rng(SeedSequence([1438291051,29109]))`, rejecting exclusions,
reserved final IDs and duplicates until 32 accepted. No random-action root is
needed: all policies are frozen deterministic. Reserved final roots 308713321,
1203078761,1983461927 remain unexpanded.

Use the SAME 32 independently seeded fresh-phase serves for every policy/mode:
six cohorts, 192 one-rally jobs, at most 96,000 transition cameras plus separately
counted initial cameras. If the prespecified validity rule permits only one
policy, three cohorts/96 jobs/48,000 transition cameras; no replacement. All first
incoming geometry must match across every included cohort. Do not reuse training
or previous DEV trajectories, and do not add simple controls: the learned-policy
identity is the comparator in this intervention, not a new competence claim.

Four existing single-threaded workers/imports-only GC freeze, no engine rewrite.
Maximum **12 rollout minutes / 15 overall minutes**, including a
180-second annotation/statistics reserve. End when complete; no extension for a
weak effect or partial. Based on prior full evaluations, 192 all-perception jobs
should cost roughly 6–10 minutes, but this is a prospective estimate rather than
a guarantee. Preserve/report partial work, all camera/physics/wall/latency/memory
costs, snapshots/pins and any storage contention. No new training or calibration.

## Statistics and claims

Primary contrasts are live minus warm-hold and live minus warm-missing on FIRST
incoming success. For each fixed policy and serve this is a paired binary
outcome; average the two policies' differences with equal policy weight.
Resample the 32 SERVE clusters, retaining all policy/mode cells, for 10,000 draws
with fixed bootstrap seed **194870293**. Do not treat 64 policy-serve cells as 64
independent serves. Report every policy's point effect and paired interval too.
The finite set of policies is not a population estimate across training seeds.

A joint 'benefit of later visual updates on these tested policies' finding
requires BOTH primary aggregate intervals' lower bounds >0, positive primary
point contrasts for each included policy, and no invalid/missing draws. This is
a claim about the average over this FIXED policy set; positive individual points
do not establish statistically significant dependence for each training seed. Otherwise
report inconclusive/mixed/negative sensitivity as observed; no added serves or
changed onset. This joint criterion is distinct from the 20-point acquisition
screen and does not promote a candidate to final evaluation.

Secondary descriptive outcomes for each policy and equal-policy aggregate:
all-incoming return fraction, repeated-return rallies, returns/physical second,
raw opportunities/returns, game losses/timeouts/scores and opponent reliability.
For bootstrapped all-return effects, compute each policy's ratio of summed
returns/opportunities on the resampled serves, then average policy differences;
do not pool policies with trajectory-dependent denominators. Later opportunities
are affected by the intervention. Report both interventions separately; no
strongest-control maximization or substitution. All intervals are paired,
secondary per-policy/all-return intervals are unadjusted and descriptive.

## Minimal implementation and recovery checks after source unlock

Reuse FreshPongEvaluator and existing single-episode rollout/completion/physics
annotation machinery. Add an optional owned/hash-pinned camera intervention only
for declared frozen evaluation; live/default state behavior stays unchanged.
Reset observation count/cache at every episode. Cache a CLONE of the twentieth
image; do not alias environment frames. Keep its observation counter/cache durable
and reject restored mode/config mutation. Mask before event derivation/vision.

Focused checks: exact default/live parity; actions, records and physics/vision/
encoder subgraph hashes identical through the intentional onset; correct twenty-
observation boundary; held events versus genuinely missing events/frame semantics;
episode reset; durable recovery immediately before and after onset; cache ownership
and frozen-parameter mutation rejection; actual mapped-weight/worker handoff and
seed/exclusion/camera-cap proof. Whole-session hashes legitimately differ because
intervention descriptors/caches are pinned: exclude only those declared fields
from CROSS-arm prefix comparisons, while each arm's own replay remains exact.

Pin source/archive, policy/sidecar/cap bytes, original training lineage and current
registration/protocol before launch. Source immutable through own zero-new-work
verification and independent raw-physics/geometry/bootstrap audit.

## Execution and provenance

Coordinator: `scripts/experimental/run_pong_camera_feedback.py`. Artifact directory:
`/workspace/m1b-experiments/pong-camera-feedback-20261021-r1`. Source bridge permits
only the optional evaluator change in `control/pong_evaluation.py`; all other core
source/runtime/model/vision hashes remain identical to both training lineages.
The worker change validates and carries the owned intervention descriptor. A
32-transition prefix of each actual endpoint on an already-recorded serve is
re-executed through archived 886d05d and current default code; records, physics,
vision, encoder, previous/latest image state and learned hashes must match exactly.
Diagnostic preflight executions/replays and initial images are counted separately
from the fresh 32-serve cohort. Core tests cover genuine observation20, masking
on observation21, both-side restart, episode reset, ownership and frozen-config
mutation rejection. Completion reuse requires report, semantic journal and mapped
artifact pins; unpinned complete reports require actual zero-new-work recovery.

The clock begins only when `prepare()` first creates `manifest.json`. Resume
retains the original timestamp, source/config/files, jobs and budgets. Incomplete
cohorts remain partial; primary and secondary statistics require complete matched
cells. Complete status also requires the 900-second overall bound including
annotation/scoring, even if all rollout jobs finish. No timing extension or
policy/checkpoint selection follows the result. Both trained policies, their
exact mapped weights/sidecars, original T/cap, training manifests, audit proofs,
source archives, seed exclusions, protocol, coordinator/helpers and runtime are
hash-pinned before launch. Source stays frozen through both post-run verifications.

Both original projection descriptors are retained separately because their
absolute `matrix_file` paths name their own training directories. Matrix bytes,
cap and fit are identical. The common `projection.json` is the first seed's
unaltered descriptor, used only by the existing completion-artifact pin helper;
policy inference uses the exact mapped weights and common original radial cap.
Raw statistical results remain visible if a deadline is exceeded, but the
within-allocation evidence flag requires complete jobs and the full overall bound
after bootstrap/scoring.
