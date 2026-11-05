# Independent challenge: discovering a numerical control adjustment

2026-11-05. Research and acceptance review by the independent challenge worker.
This note owns no execution implementation and initially uses no native games.
Companion execution and parent orchestration decisions must identify their own
prospective specifications; the recommendations below do not retroactively
define acceptance for collected data.

## The useful next claim

The 2026-10-30 result established carried, online reward-driven selection among
16 anonymous feedback structures, including recovery after polarity reversal.
Every structure had the same five literal-zero numeric coefficients. The
successful endpoint therefore does not demonstrate discovery of a new numerical
response law. Another finite menu or a polarity-only estimator would leave that
gap substantially intact.

A useful next result is a carried learner that changes continuous numerical
control state from its own chronological public sensor/action history, changes
actual commands within games, and thereby recovers reliable returns after one
unannounced actuator change. An inherited relational pair/axis and visual encoder
are legitimate declared priors. Success would remain numerical sensorimotor
adaptation over frozen vision and a servo/model prior, rather than discovering
vision, the whole controller architecture, or full M1B.

The strongest attribution is task performance under an actually acquired model
whose numerical updates are frozen, compared with the continued learner on the
same exogenous starts. A model fit improving its own error without improving
returns does not satisfy this claim. Nor does continually changing coefficients
when the inherited zero-coefficient controller already succeeds.

## Intervention selection before learning

The parent has authorized a bounded diagnostic scout, with at most 12 games,
12,000 transition cameras and 240 seconds: four prospectively balanced new starts
matched under native actuation, an eight-camera FIFO command delay, and a +0.60
continuous drive drift. This scout chooses a challenging but solvable target; it
must not be mixed into subsequent acquisition/confirmation starts.

Delay is the preferred diagnostic. Source inspection finds native activation
time constant 0.08 seconds, viscous velocity time constant 1/4 second, maximum
speed 0.7 and paddle half-height 0.10. An additional approximately 0.16-second
FIFO can increase pursuit overshoot enough that velocity damping and compensation
for already queued commands matter. The exact delay is held by the environment;
the learner may use requested-action history and public elapsed time, but receives
neither its value nor the physically delivered command.

Drift alone is a weaker candidate. A bang-bang position servo often rejects a
constant bias automatically by changing its command duty cycle. A large drift
also reduces force in one direction, making some trajectories physically hard;
failure can then reflect inadequate actuation rather than a useful learning
problem. The +0.60 scout is diagnostic, not a commitment to that target.

Other distinct intervention choices are asymmetric motor gains, a slower
activation time constant, or changed damping. These can require new numerical
compensation, but each still needs an explicit fixed-controller performance loss.
Do not successively tune all these settings on fresh games. If the declared scout
finds no useful task gap, record that and make one reasoned mechanism/target
decision before another bounded development allocation.

## Candidate mechanisms and their failure modes

### Preferred: learn an action-to-displacement model

Fit a small forgetting-factor linear model from actual anonymous public-ID
displacement to requested command history, possibly including a few autoregressive
motion terms and an intercept. A finite impulse-response model avoids unstable
autoregressive rollouts; an ARX model is more compact but requires explicit
stability and support checks. Fit coefficients continuously, rather than select
one of a list of response templates. The controlled pair/axis can be inherited
and must be declared as such; this milestone need not also solve entity discovery.

The predictor should score a response before incorporating that response. Store
that pre-update prediction and residual. Keep samples only across the same
published identity and valid observed endpoints, with known public elapsed time.
Unknown visibility, identity switches, and scene resets are missing observations,
not zero displacements. A one-frame position difference has substantial raster
quantization; a longer displacement window or existing observed-velocity fit may
help, but adds a real measurement lag that must be counted.

The controller can use learned coefficients for a generic damped pair servo or
for a bounded, explicit three-command rollout of the learned public-coordinate
model. Such a rollout is a model/control prior, not access to native simulator
futures. Feed the known requested-command queue into it; replacing that history
with hidden delivered actions would leak the intervention. Choose a fixed horizon
and task-independent control cost before acquisition. A one-step inverse can be
ill-conditioned because the current command has little immediate effect through
activation and delay. Large inverse gain from a tiny estimated coefficient is an
obvious failure mode.

Closed-loop data cause a second problem: actions correlate with target motion,
and clipping correlates with extreme positions. Mere command/velocity covariance
can identify the ball or opponent spuriously, or estimate the wrong sign. Modest
independent command excitation, known from the learner's own RNG, can establish
response support. Its failures count in live performance. Repeated blind dither
can itself prevent 95% reliability. A full autoregressive model can also explain
motion from its own past without identifying the actuator at all; inspect the
action-associated coefficients, excitation support and actual coefficient/action
ablation, not just total prediction error.

### Simpler control alternative: learned response time and damping

Use a learned lagged impulse response to estimate a continuous response/stopping
time and sign; apply that estimate to the inherited relative-position score using
observed velocity. This is a smaller control surface than general predictive
control. It still assumes the response can be summarized by a scalar time and
needs a declared handling rule for noisy/negative impulse coefficients. A sign
alone is insufficient for the new claim; task benefit must disappear when the
learned numerical damping is frozen or removed. A fixed hand-written mapping
from estimated delay into damping is a remaining controller prior, to be reported.

### Materially different fallback: reward-modulated local perturbation

Perturb one or two continuous coefficients (own-velocity damping and possibly
target-velocity lead), carry an eligibility trace for the actual perturbations,
and update from actual own-hit reward or public negative score relative to a
declared reward baseline. This directly optimizes returns and does not need an
accurate dynamics model. Sparse returns and long, varying motor delay make credit
assignment noisy. Whole-game perturbation resolves ownership but gives slower
updates; overlapping within-game perturbations need explicit lag/trace support.
Keep the parameter surface tiny and bound the attempt prospectively. Prior failed
large reward actors do not justify another broad actor sweep.

These mechanisms are alternatives, not permission to run all of them. Once one
shows useful acquisition and causal adaptation, qualify that unchanged route.

## Bounded success test

Before native acquisition, pin the implementation, numerical settings, start
schedules, intervention and scoring windows. Begin with neutral numerical evidence;
do not initialize with a known response coefficient or choose an initialization
from its observed outcome. The learner's update signature should admit only
public entities, requested actions, elapsed time and explicitly allowed public
reward/status. Keep environment seeds, phase indices, intervention flags, physical
state and reporter opportunity labels outside that boundary.

Recommended evidence, scaled to the final authorized allocation:

- Two independent chronological learner streams, with all games and exploration
  counted. Each first establishes reliable control, then encounters the same
  predeclared hidden change without a reset of learned statistics. Fixed late
  windows require at least 95% first and later returns with enough later
  opportunities to make the claim substantive. A shortage is unmet acceptance,
  not an excuse to choose a different tail or extend after looking.
- Branch the actually acquired model at the A boundary, freeze all numerical
  updates, and run it on matched B starts. Continued learning must produce both a
  material task recovery (suggested at least 20 percentage points over frozen)
  and the required competent late window. This identifies adaptation; normal
  acquisition or parameter drift alone cannot.
- Preserve exact learned-state continuity across games and serialize/restore it
  against an independent continuously running shadow. Public visual tracking and
  pending motor/sample windows reset normally between games. Do not fit a
  displacement across a game boundary or carry a future response from one game
  into another.
- Show actual same-game pre-update coefficient changes and resulting commands
  before later successful returns. Report phase-anchored recovery latency in
  games, opportunities, transition cameras and physical seconds; count early
  misses and any exploratory commands. Do not equate post-change game count with
  within-game learning time.
- Include the inherited two relevant polarity servos as explanatory controls if
  useful. A full 16-arm post-hoc search is unnecessary once continuously fitted
  coefficients and an acquired-state freeze ablation establish the new claim.
  A reset-each-game learner is an optional retention-benefit control: if the
  estimator learns fast within each game it might perform well, which would not
  disprove carried state continuity or within-game adaptation.
- Report complete and capped games, all initial and transition cameras, wall and
  physical time, first/later counts, failures and own hits per physical second.
  Keep terminal miss/plane reporting outside learner feedback, preserve the
  final public observation, and censor artificial truncation rather than invent
  a reward failure.

Qualification should test one chosen intervention convincingly. A reverse-back
phase is useful for continued adaptation but is not required to claim success on
the chosen change if it consumes the budget needed for a causal frozen control.
No claim of faster recall follows from return-to-A without a cold learner latency
comparison. Contacts within a game and games in one carried stream are dependent;
report each independent stream, not a fictional sample of independent contacts.

## Outside primary implementations inspected

Network policy and cloud readiness were inspected before network requests. Only
permitted GitHub hosts were used. No outside code was installed or used to train.

1. [Padasip RLS source](https://github.com/matousc89/padasip/blob/4c13692b8f7c812b74f7f7b0d31b76a0c36df156/padasip/filters/rls.py)
   implements inverse-correlation forgetting and error-driven linear coefficient
   updates, including `R = (R - Rxx'R / (mu + x'Rx)) / mu`. It explicitly notes
   practical sensitivity and forgetting factors near one. The source motivates
   a small continuous online estimator, not a guarantee for quantized,
   closed-loop Pong. Its coefficient update is algebraic; no backpropagation is
   needed.
2. [Padasip NLMS source](https://github.com/matousc89/padasip/blob/4c13692b8f7c812b74f7f7b0d31b76a0c36df156/padasip/filters/nlms.py)
   uses `mu * x * error / (eps + x'x)`, an inexpensive local normalized update.
   It is a useful simpler estimator comparison, though correlated action-history
   taps can converge slowly. RLS and NLMS are different update rules for the same
   response-model route, not two materially different control-learning ideas.
3. [Brian2 Izhikevich 2007 example](https://github.com/brian-team/brian2/blob/master/examples/frompapers/Izhikevich_2007.py)
   implements separate decaying eligibility and dopamine variables with the
   synaptic update proportional to their product. It cites Izhikevich,
   *Cerebral Cortex* 17(10), 2443–2452. This is a primary runnable demonstration of
   a three-factor mechanism, not evidence that its default constants solve Pong.
4. [FilterPy fading-memory source](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/fading_memory.py)
   documents covariance inflation for forgetting and explicitly warns that users
   must supply meaningful states and models. Fading-memory prediction is another
   possible estimator implementation; an accurate motor model and successful
   task controller remain separate questions.

Repository sources inspected: `pong_carried_structure_learner.py`,
`run_pong_carried_learning.py`, `pong_relational_feedback.py`,
`pong_relational_persistent_binding.py`,
`run_pong_relational_categorical_replication.py`,
`src/fly_connectome/control/environment.py`,
`src/fly_connectome/control/pong_evaluation.py`, and
`src/fly_connectome/pong.py`. The model/servo proposals above are research
recommendations, not an assertion of implemented or observed success.

## Diagnostic outcome and measurement audit

Read the completed scout's saved `summary.json` at
`/workspace/m1b-experiments/pong-numeric-challenge-20261105-r1`:
native 22/22 returns (first 4/4, later 18/18); delay8 8/11 (first 4/4,
later 4/7); drift060 7/10 (first 2/4, later 5/6). All 12 games completed,
using 7,461 transition and 12 initial cameras in 19.94 seconds. This small
development sample supports choosing delay8, whose capacity is preserved and
whose failure affects sustained returns. It does not yet establish a competent
learnable solution or a precise general failure rate.

Further source inspection confirmed that `install_exact_transport_copy` preserves
the default `VisualStateEncoder` arithmetic exactly. It does **not** install the
experimental four-observation least-squares velocity estimator. Default velocity
uses the difference since the previous observed position, divided by the sample
gap times nominal 0.02 seconds. Actual physical camera intervals alternate two
and three 120-Hz ticks. Normalized coordinates are approximately [-0.5, 0.5],
with `(pixel - image_center) / 64`; a half-pixel centroid movement per nominal
frame gives a velocity increment 0.390625. An unobserved carried entity can keep
`velocity_known=True` and an old nonzero velocity. These facts favor observed
same-ID position windows and explicit age handling over treating the published
velocity as instantaneous motor state. No new sensor field or native state is
needed to address them.

## First implementation audit and independent tests

Reviewed `pong_visual_motor_fir.py` and `run_pong_visual_motor_learning.py`
before native learner execution. The implementation estimates 32 continuous
requested-command taps and an intercept from four-interval public displacement
targets. Interval features use actual public elapsed time. The candidate forecast
uses known requested history and a fixed nominal 16-interval horizon; it does not
query any native future or receive the delay. The current requested command enters
the current interval's first command tap. Summed interval features align with the
observed window endpoints. Rebinding the selected motor ID clears displacement
windows, preventing a position jump across identities from becoming a motor label.

The inherited pair member and axis remain fixed priors. The fitted tap vector is
a predictive approximation under correlated feedback and raster measurements,
not a uniquely recovered physical impulse response or true delay identification.
No published observed velocity is used by this learner. Forgetting advances on
admitted observed-endpoint update exposure; unsupported gaps and reset time do not
advance its clock. This is an explicit supported-exposure clock, rather than a
claim of forgetting over all physical elapsed time.

Added `tests/test_pong_visual_motor_fir.py`. Eight independent tests passed in
9.37 seconds using `/workspace/Fly-Connectome/.venv/bin/python -m pytest -q
tests/test_pong_visual_motor_fir.py`, with no native Pong environment or cameras:

- A known synthetic delayed response is recovered numerically from actual
  variable-duration command intervals, with residuals computed before fitting.
- A delayed forecast consumes the correct old requested commands and hypothetical
  candidate suffix; earlier commands outside its support have no effect.
- Missing observations do not become zero displacement targets, while valid
  observed endpoints may span missing middle frames. ID jumps are not fitted.
- Public-ID renaming, entity order and unused velocity changes preserve learning.
- JSON restore at a game boundary preserves the model, RNG, subsequent commands
  and updates exactly; a fresh scene is not fitted against the prior game.
- A coefficient-frozen model preserves its numerical arrays while actual command
  history, public position history, diagnostics and exploration keep advancing.
- Cold online learning substantially improves tracking on a quantized synthetic
  motor with different gain, time constants and delay from native Pong: late RMS
  position error is below 0.045 and below 40% of the inherited servo's error.
- Full-public offline replay reconstructs actions, decisions, numerical updates
  and final state exactly; a corrupted final-state hash is rejected. Replay is
  forbidden from constructing a native evaluator in this test.

Execution added a guard requiring at least 256 acquired updates before branching
the frozen A model, preventing its update-count-dependent probe probability from
remaining at 10% while the live learner moves to 1%. No causal model/controller
blocker was found. Before pinning, the reviewer requested a harness timeout guard:
queued inherited controls can return `not_started` without a `report`; summary
code must retain that record without dereferencing `report['hits']`, and must
report camera upper bounds for interrupted work. This is accounting repair, not
an algorithm change or another development trial.

That accounting repair landed before native launch. A ninth focused regression
test passed (1.35 seconds, other eight tests deselected), preserving both a queued
`not_started` record and a partially executed record, including one ambiguous last
transition in its camera upper bound. The prepared r1 specification and source
were preserved, and execution repinned r2 before launch. No model/controller
change was required by the independent audit.

## First native pilot: interpretation and next hypotheses

Saved r2 receipts showed A acquisition 48/48 returns over eight games. B delay8
finished with live 80/83, frozen acquired A 70/76, and inherited-only 29/44. Thus
the learned numerical controller substantially transfers to delay8, while the
continued B-update gain is only about 4.3 percentage points. This is useful
development evidence, not sufficient material adaptation/reliability evidence.
The live first-return count is 15/16, below a 95% first-return criterion for that
phase. No completion claim follows from these pilot counts.

Read actual public trajectories for two representative failures, without native
queries. Game 12's late miss followed more than 20 cameras with the selected
target unobserved. Valid carried public target coordinates remained available,
but the numeric controller's observed-target gate disabled it and the inherited
servo controlled this interval. This is a control-support issue rather than
evidence that the numerical estimator failed. Game 16's initial public scene was
empty; commands were `[neutral, neutral, 0, 0, ... 0]` through the first missed
plane at physics tick 106, approximately 0.883 seconds. Once observations became
available, every command requested maximum motion in the needed direction. Public
own/target coordinates near contact were approximately 0.289/0.430, a gap of
0.141. That case appears constrained by sensor startup plus delayed actuation;
changing its post-observation control choice cannot improve on an already maximal
correct-direction request. This supports preserving early reachability in the
next challenge rather than simply increasing FIFO delay.

The independent challenge worker proposed two next mechanisms, with no native
work launched by this worker:

1. Increase positive actuator response gain while retaining polarity. Acquired A
   predictions then underestimate displacement from requested commands. The
   unchanged estimator can learn a continuous response scale, rather than choose
   another sign. Velocity saturation can weaken the effective gain change, so
   actual frozen-versus-live task performance must discriminate it.
2. Add neutral-drive drift while preserving both extreme-direction capacities.
   A pure +0.4 drive bias weakens one direction and can recreate short-serve
   reachability failures. The capacity-preserving affine transformation
   `physical_drive = 1.4 * requested_drive + 0.4` instead maps requests
   `[-1, 0, +1]` into `[-1, +0.4, +1.8]`. One extreme force is unchanged and the
   other increases; neutral now moves. The existing FIR plus intercept can learn
   this continuous offset/response change locally. The acquired A model assumes
   zero neutral response, making an intercept adjustment a concrete hypothesis.

Parent orchestration adopted the affine refinement for a bounded subsequent
diagnostic alongside the gain change. These external transformations are held by
the environment; their numbers or labels must not enter the learner. They are
mechanistically chosen diagnostic cases, not successful adaptation claims. The
decision rule remains actual reliable play and material improvement over the
acquired coefficient-frozen controller. If prediction improves while returns
remain poor, inspect support and the control horizon/objective; do not sweep
estimator constants. If both live and frozen succeed, report transferable
robustness. Native prediction failure instead motivates examination of identity,
wall saturation and excitation. Preserve normal-condition retention when choosing
any continuation.

The completed r2 pilot subsequently confirmed normal-condition restoration at
48/48 over eight A-return games. All 64 planned games completed, using 49,049
transition and 64 initial cameras in 402.95 seconds. Saved source/assets remained
pinned and all numeric full-public replays passed. These checks support the
integrity of the observed acquisition and transfer result; they do not supply
the missing material adaptation gap.

One further distinguishing hypothesis, proposed before any associated execution,
is a three-action cycle that moves which requested command is neutral while
preserving all three native physical responses. The current linear requested-drive
FIR cannot represent an arbitrary nonmonotonic map over three commands; adding
requested-drive-squared taps, or one-hot requested-action taps, permits continuous
coefficients to discover their separate responses. No controller catalogue or
physical mapping label would be supplied. Unlike a polarity reversal, the existing
16 fixed sign servos cannot generally implement a request that was neutral becoming
the active response in one direction. This would be an explicit representation
extension and a sensorimotor-association test, rather than another nearby gain
setting. It is a proposal pending parent orchestration, not an executed result.

The gain/intercept diagnostic completed all 32 games: gain2 live 48/48 and frozen
46/46; affine live 42/43 and frozen 45/45. It used 31,258 transition and 32 initial
cameras in 112.49 seconds, with pinned assets and full-public replays passing.
These targets also show transfer/robustness of the acquired A controller, not
required new adaptation. The one affine live miss had the numeric controller
disabled on 18 of the final 21 cameras because its target was unobserved, with
zero target lead throughout; requests were already maximal in the needed
direction. Different earlier rally trajectories mean this isolated live loss
cannot establish estimator instability or a general harmful effect of learning.

## Categorical action-response extension preflight

Parent orchestration selected the three-action cycle as the sole next changed
test case. Execution implemented a fresh 96-coefficient model: three anonymous
requested-action channels times 32 temporal taps, with all-zero features before
any actual request. It learns A before the cycle, then carries that newly acquired
state through the intervention and restoration. Old FIR experiments and their
checkpoints remain preserved. No action permutation is stored or selected by the
learner; its fixed prior is a general separate response channel for each discrete
request. A global-intercept-only representation would conflate an actual request
that used to be neutral with nonexistent pre-game history, whereas this basis
represents their different startup responses.

The added excitation prior is a fixed four-frame held random request, with the
same initial 10% and continuing 1% probe-start probabilities. Holds cancel when
the public relation is unsupported. Probe state and RNG are serialized. There is
no intervention flag, phase-triggered exploration or error detector. The readiness
gate uses the full predicted response range over all three requests, so the newly
active formerly neutral request can establish control support.

Added three focused independent tests in
`tests/test_pong_categorical_motor_fir.py`. Arbitrary nonmonotonic three-request
response fitting, absent-history/actual-neutral distinction, and exact held-probe
JSON continuation with frozen numerical arrays passed. A synthetic carried cycle
check initially failed when using the earlier helper motor's **additional six-frame
command delay**: late RMS tracking error was 0.285, and only 165/3,800 post-change
choices passed the response-gain readiness gate. The inherited fallback repeatedly
chose the newly neutral request (3,576/3,800 requests), starving response support
and spending 1,242 frames at the synthetic wall. Channel response signs eventually
became correct, but the short-horizon gain remained too small. The initial A model
fit had pre-update/zero-model squared-error ratio 0.107. This is a documented
combined lag/support failure, not omitted evidence or a native result.

One follow-up used the **same** synthetic motor, numerical settings, A/B lengths
and acceptance thresholds with that extra delay removed, matching the chosen
native cycle's zero-added-latency scope. It passed: late RMS error below 0.06 and
below 40% of the actually acquired frozen model's error, in 4.02 seconds. No learner
source or parameter changed between these synthetic checks. These focused tests
clear causal plumbing and feasibility for the chosen zero-added-delay test; they
do not qualify adaptation under delayed cyclic controls or replace native evidence.

## Pausing point: completed categorical pilot

The fixed pilot completed all 48 games: initial A 41/42 returns (first 7/8,
later 34/34); changed cycle B 78/82 (first 15/16, later 63/66), versus acquired-A
frozen control 0/16; restored A 30/33 (first 5/8, later 25/25). Restoration began
with three first-return losses, followed by five games with six returns each.
This is substantive development evidence of new within-game response association
learning, carried adaptation and recovery, while independent fresh qualification
and its prospective reliability criteria remain pending.

Final read-only review checked all 48 receipts, 31 live and 15 frozen carried-state
boundaries, 34 pinned files/assets, state and archive hashes, tape/public lengths,
hit counts, matched B assignments and unchanged frozen coefficients/covariance.
No discrepancy was found. Every saved exact public replay passed; 3,817 live
pre-probe choices differ from a game-start frozen model on the same actual public
history. This is descriptive within-game action attribution, alongside the
separate matched native frozen-control performance evidence.

The learned empty-history 0.32-second displacement forecasts for requests 0/1/2
changed from approximately `[+0.066, -0.001, -0.071]` after A to
`[+0.014, -0.021, +0.094]` after B. The changed association appears in continuous
coefficients; these are local predictive approximations, not uniquely identified
true physical responses. Restored control also should not be described as exact
restoration of the old model. Cost was 27,328 transition and 48 initial cameras,
319.81 seconds, with no unknown unsaved native cost. User requested a pausing
point; no additional experiment, implementation or qualification was started by
this worker after that instruction.
