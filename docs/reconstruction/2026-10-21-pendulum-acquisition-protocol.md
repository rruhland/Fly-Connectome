# Pendulum survival acquisition with persistent visual learning

This is one prospective, finite comparison under the user's continuous-iteration
authorization. It adds an ordinary camera task and reuses the existing visual
learner, fixed linear projection and true-online SARSA. There is no backpropagation,
teacher, physics label in policy inputs, privileged fitted initialization or final
holdout use. This is a second-task acquisition/compatibility experiment, not by
itself complete M1B or proof of policy transfer.

## Fixed conditions and objective

The two predetermined visual starts are seed1's original generic prior and its
retained snapshot after the registered 2,000-transition Pong exposure (2,014 learned
images). They come from the audited online-vision study, not its 4,000-transition
prequential endpoint or the better-performing second policy. Explicit reviewed
control-only source bridges preserve every learned tensor, calibration/RNG field,
content hash and inherited learning count. Original Pong W/T/cap stay unchanged.

Both arms start a fresh zero three-action B head, use the SAME score-free B transform
and environment/exploration seeds, and learn visually on every initial and transition
image. The common transform was fitted once on the existing 1,024-transition frozen
generic-prior random-action cache: 214 DCT coordinates, ridge .01, tolerance 1e-8,
supported rank163 (326 free action contrasts). It was not fitted to reward, geometry,
action labels or either treatment's learned B memory. Cache/fit bytes remain intact;
the subsequent learner-seed/summary-only source delta is recorded explicitly.

Pendulum physics: dt1/120 s, gravity8, actuator magnitude12, damping.1, failure
|theta|>=pi/3. Initial signed tilt has uniform magnitude .03–.12 rad and independent
angular velocity U[-.2,.2]. Three torques are negative/zero/positive. The 64×64 raster
shows a pivot, rod and bob; no internal state reaches the policy. Camera cadence is
50 Hz with exact two/three-tick integration and a fresh phase at each episode reset.
Reward is actual elapsed survival seconds, including a partial failure interval.
Failure is terminal; the 5 s limit is a continuing-task truncation with bootstrap
from the actual final image. Genuine failure and the time limit both reset the scene
but retain learned visual memory. An eager next-episode initial image/pending action
can be learned inside the ending transition and is retained/counts even at dose end.

One fixed recipe: alpha .001, lambda .995 per camera, epsilon .2 independent per
camera, gamma .95 per physical second, radial squared-norm cap256 AFTER projection.
This is a first screen, not matched effective step sizes or a convergence guarantee.
The fixed calibration median/p99 norm² are13.76/199.31; two of1,024 rows activate the
cap. Alpha*norm² is bounded by .256. Lambda alone has roughly4 s e-fold duration;
including physical discount gives roughly3.3 s. The separately accounted256-step
per-arm plastic fixture stayed finite and recovered exactly; diagnostic heads and
updated memories are discarded. Nonfinite updates are rejected before journal
commit; |Q|, |delta| or max|weight|>1000 stops that arm as a failed allocation.

## Seeds, budgets, persistence

New roots: environment1836497107, learner720583127, B development915760327,
control actions1762054381, tag29114, bootstrap1107836249. Environment and SARSA
private RNG streams are distinct and explicitly pinned, shared across the two arms.
Same roots cease to imply matched trajectories after actions diverge. The64 B
development seeds are expanded from SeedSequence([DEV,tag]), rejecting all actually
declared prior/calibration/fixture IDs, reserved final IDs and new roots. Random
control action seeds use SeedSequence([action_root,tag,serve_index]). Roots and
exclusion metadata are frozen before any draw; no seed replacement or final-root
expansion. The A cohort is explicitly REUSED old online-development64, not fresh B
or final evaluation.

Each arm gets exactly20,000 transition cameras, with immutable theta/mapped W/cap/
owned visual snapshots at5k,10k and20k. Every camera remains a durable journal commit;
the existing snapshot/replay bounds and full content/mutation checks remain intact.
Checkpoint exports occur before the next transition. Recovery at a boundary rebuilds
an unfinished export from the exact checkpoint; a missing past export fails closed.
Reports and completion pins bind actual row counts, semantic journal hashes and all
endpoint bytes. Completed reports lacking pins undergo zero-new-work actual recovery.
Failures persist across coordinator restarts; no failed live object is reused.

Two single-threaded training workers, fixed20 min SHARED training cap; then four
single-threaded frozen-evaluation workers. The evaluation clock starts at the durable
training frozen timestamp, allows20 min INCLUDING scoring, and total allocation is
40 min from its original immutable clock. No restart extends a phase or overall
deadline. Export/checkpoint overruns remain visible and can invalidate budget status.
No candidate clock starts during staging. If either training arm lacks its exact20k
endpoint, the full pair remains partial;5k/10k outputs remain labeled descriptive,
never replacement candidates. All partial rows, failures and constructor/recovery
compute are retained. Uncommitted crash tails are unmeasured, never asserted zero.

## Frozen outcomes

No visual or policy updates occur during any probe. All64 B starts are shared by
both endpoints and negative/no-op/positive/random1/random10 controls. Random controls
have explicitly matched initial-state seeds and action RNG recipe. Controls bypass
perception only through the already tested physics/action-equivalent path. Both own
zero-Q initial policies equal constant-negative by the fixed first-index tie rule;
the negative raw evidence is explicitly reused as their own-initial control.

The endpoint competence screen for EACH arm requires mean survival>=3 s, at least
50% reaching the5 s limit, mean lift>=1 s over the strongest own-initial/simple
control, and paired95% bootstrap lower bound>0 for that lift. The control maximum is
recomputed in every10,000-draw shared-start cluster bootstrap. Continuous effects,
failure/time-limit fractions and experienced-minus-fresh CIs are reported regardless
of the gate. One paired training seed supports a condition comparison, not training-
population inference. No best checkpoint, early stopping or reward tuning.

Execution priority is B endpoints plus five shared controls; both A after-B probes;
then descriptive5k/10k B probes on the first32 of the same64 B seeds. All prescribed
jobs must finish within phase/overall bounds for a complete allocation and eligible
pair screen. Fixed midpoint curves are scored AFTER training, never used to select
the endpoint; time-to-criterion is at most coarse/descriptive.

Both after-B visual snapshots are immediately tested on the old online64 Pong seeds
using the SAME original seed1 A W/T/cap, with no A retraining. Before-B evidence is
explicitly reused: old seed1_before for fresh and seed1_after for experienced.
Original summary, annotations, snapshot bytes, own verification and independent
audit are pinned. First-return and all-opportunity effects use paired64-serve
resampling; no per-episode IID or formal noninferiority claim. A CI crossing zero
does not prove retention. Report B-only and inherited prior experience separately;
new B heads imply no claim of policy-head transfer.

Logical committed transitions, eager/constructor initial images, visual-learning
images, physics ticks, recovery replay and summed task/shared wall are separate.
Independent audit physics reexecution adds zero candidate observations. Report bank
growth, process-lifetime peak RSS and per-segment committed-camera latency. The
preflight's fragment/quantization limitations remain: entity counts are not semantic
object recognition, early subpixel motion can be visually unchanged, and observed
coverage does not establish angular-velocity observability. Do not redesign the
vision or raster after seeing this allocation's reward outcomes.
