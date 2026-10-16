# Prospective score-only smooth linear policy search (.25 opponent)

## Hypothesis and scope

Previous reward-only SARSA arms, including capped raw/RMS, lower alpha and hold4,
failed their fresh development screens. A quarantined oracle-fit witness shows
that the existing full linear decision class can control on12DEV serves; it does
not establish linear Bellman-Q representability. This new experiment removes the
critic/bootstrapping assumption and uses coherent whole-rally parameter exploration.
No oracle fit/labels/coefficients enter this experiment or select its basis,
conditioning or hyperparameters. No backprop, reward shaping or physics state
enters a policy. Frozen vision, raw structured transport and SARSA pipeline remain.
Claims are explicitly opponent speed.25; original.45 remains paused and unchanged.
The restricted search class may lack the full class's demonstrated capacity.

## Fixed score-free coordinates and perturbations

Use the lowest3×3 orthonormal2D DCT modes in EVERY spatial channel:7current,
8future,5relation and2context planes. Retain8future-outside and8metadata
coordinates. This gives214coordinates before score-free rank projection. Do not
choose modes from oracle coefficients, rewards or DEV. Fit only the already
observed1024generic pre-action inputs from score-free Pong calibration
(seed849375169/action686753009, .25 opponent). Reuse its pure sparse-input cache,
containing only indices/values/indptr; no new calibration observations.

Let B be that fixed projection and b the existing frame-present coordinate
(raw index4384, projected index209). For each other coordinate subtract its
calibration mean times b; retain b itself. The all-zero input remains exactly
zero; individual missing coordinates may become nonzero in a present frame.
Use SECOND MOMENTS of these centered coordinates, retaining bias energy1, rather
than np.cov that would remove constant b. Eigh support threshold is strictly
lambda>1e-8*largest eigenvalue, fixed without reward. Apply inverse square root
(lambda+.01)^(-1/2) on supported modes, zero on others. Ridge.01 bounds inverse
gain below10. Actual calibration rank183 gives366effective free action contrasts,
while214coordinates are stored. The projection restricts the policy class; it is
not an invertible representation of all4389rawfeatures. Raw transport is unchanged.

Map each three-head parameter matrix theta back to ordinary full weights W=theta*T
before rollout, where T includes centering/whitening/DCT. Tests must reproduce
scores/argmax and blank input. Apply the existing cap256 at inference; its positive
radial factor preserves every mapped policy's greedy argmax but does not repair
corrupt coordinate ratios. No adaptive transform during search.

Optimizer RNG seed1873926119; run training seed172900127. Initial theta is .10times
one Gaussian direction, with action-common head component removed and normalized
to unit calibration Q-RMS. Each generation samples8new Gaussian directions with
the same projection/normalization, then evaluates theta±.25*direction. Each pair
and all16candidates use EXACTLY the same4training serves. Rotate fresh serve IDs
by np.SeedSequence([training_seed,872631,generation]); four draws in[1,2147483647),
excluding all DEV/final/calibration IDs, with declared uniqueness. Every training
serve starts its own fresh camera phase. No selection by serve outcomes.

Fitness is mean terminal score across4rallies: win+1,loss−1,truncation0. Hits,
natural/own-hit labels and teacher quantities NEVER enter fitness or updates.
Paired common serves cancel action-independent natural-win feedback. Rank8pairs
by max(plus,minus) fitness, ties by lower direction index. Use top4; sigma_R is
population standard deviation of their8fitnesses. Update=.10/(4*sigma_R) times
sum((plus-minus)*direction). If sigma_R=0, update exactly zero. Remove any common
head direction and clip update calibration Q-RMS to .10. This is a normalized,
clipped ARS variant, not an exact claim of standard ARS. No best-policy retention.

## Finite allocation, checkpoint and evaluation

At most8FULL generations:512attempted training rallies, hard60,000training camera
transitions and24minutes training wall. Overall shared wall30minutes includes
fit/setup, training, initial-control and final evaluation, reserving6minutes for
evaluation. Three persistent spawned workers reuse imports. Existing per-camera
DurableRun commits/source/runtime/frozen-policy checks remain, with full chunk
snapshots and opt-in warm sessions. No unvalidated batched-perception rewrite.
Before dispatch reserve each concurrent worker's maximum additional camera grant;
refund unused grants. No overshoot. Fewer than200remaining cameras may remain
unused. Incomplete rollout/generation work and costs remain preserved but NEVER
contribute a partial update. Outer optimizer commits theta/RNG only after every
one of64rallies finishes; deterministic generation definitions and immutable
candidate arrays permit recovery without redrawing or applying an update twice.
Completed reports and all semantic SQLite tables are hash-pinned before fitness use.
Actual recovery continues complete journal verification. Failed tasks stop search.

Freeze the LAST completed generation when8generations or budget ends, regardless
of its quality; no best/early checkpoint selection. At least2fullgenerations are
required for a valid evaluable endpoint. Record all incomplete work. No deadline
extension: unfinished final evaluation stays partial. Source/config/cache/transform,
optimizer state/RNG, full candidate journals and all costs are retained.

Only then evaluate endpoint and initial restricted readout independently on the
same24DEV serves/seed904271833, hold1,2400tick/20second limit, frozen visual state,
score reward and reset/camera semantics matched to proven four simple controls.
Reuse original initial/down,up,random,no-op controls with source/physics proofs;
zero policy is exactly the reused initial/down control. No final seed use. Initial
restricted control is additional to these baselines and can independently raise
thresholds. Total gate=max(14,strongest matched totalwins+4); own-hit gate=
max(6,strongest matched own-hitwins+4), independent maxima. Truncations are nonwins.
Report all rally outcomes, returns/physicaltime, hits, timeouts, natural and own-hit
wins, initial/endpoint comparison, every training/checkpoint/cost count. Passing
this DEV screen only permits separately registered independent-seed replication;
no automatic replication or untouched final evaluation is allocated.
