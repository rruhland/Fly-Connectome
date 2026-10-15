# Proposed fixed toy preregistration — not executed

This supersedes the cursor proposal in the storage checkpoint. Astra conditionally
approved bounded toy reconstruction, with no Pong competence experiment. The parent
requested this exact budget/runtime proposal before any toy run. No toy data have
been generated or used to choose the following settings.

## Shared interface and settings

Both tasks render a 64x64 binary grayscale image at exactly 0.02 seconds per action.
There is no Pong physics tick conversion. Events are successive image differences;
vision and VisualStateEncoder use the same 0.02-second cadence. The learner discount
is the actual outgoing `0.95 ** 0.02`, with its prior incoming discount in the trace.
No simulated wall time or nominal frame count substitutes for that duration.

The target is a 3x3 white square centered at (row 16, column 16 or 48) on a black
background. The selected consequence is a 3x7 white horizontal marker centered at
(row 44, the selected column). Coordinates are rendering details, not controller
inputs. The target stays visible throughout every episode; the selection marker
is absent before the first action and remains visible after selection. No cue is
hidden or removed, so these tasks do not test invisible-cue memory.

There are two legal actuator indices: 0 selects left; 1 selects right. Reward is
zero except terminal score +1 for a matching side or -1 for the other side. There
is no distance, hit, motion, alignment, or intermediate shaping reward. No hidden
target/selection labels enter the feature encoder or action learner. Generic
structured/sparse visual features remain unchanged, including forecast features;
no benefit is attributed to forecasts without a separate ablation.

Pinned visual prior: SHA256
`a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173`, loaded via the
upgraded consensus-v3 load_default configuration. Vision is frozen; only the
new two-action linear readout learns. Every run starts from zero readout weights.
No feature balancing, scaling adaptation, architecture change or backprop is added.

Fixed parameters: alpha=0.0001, gamma_per_second=0.95, epsilon=0.2 during learning.
The two arms differ only in lambda: 0.8 versus 0.0. These are the documented rebuilt
defaults, NOT historical alpha=.05/lambda=.9/gamma=.99/epsilon=.1. Lambda is per
decision. Neither nominal trace decay nor held-out results will be used to tune it.

## Tasks

Immediate visible choice: the initial action selects a side and immediately ends
the episode after one 0.02-second transition. The final image includes the target
and selection marker. The two lambda arms should be equivalent here because each
episode has one update and traces reset; this is a useful experimental invariant.

Delayed visible choice: the first action latches a side. Both actuator commands in
transitions 2 through 8 mean hold the already selected side; neither can replace
that choice. The target and latched selection remain visible during all eight
transitions, including the terminal image. Score arrives only on transition 8
(0.16 seconds after the initial observation). Delay is deterministic; no action
changes its length. Thus the consequence remains observable while reward is
delayed; this is not a partial-observation memory challenge. Each episode has
exactly eight transitions, with natural termination rather than time-limit truncation.
Truncation correctness remains covered separately by controlled lifecycle tests.

## Fair allocation, exact caps and seeds

Learning seeds are 2101, 2102, 2103 for immediate and 2201, 2202, 2203 for delayed.
Each seed is paired across lambda arms with identical environment side sequences,
independent policy RNG streams initialized identically, and identical initial
parameters. Environment RNG and learner RNG are separate generators.

Each task/seed/arm receives exactly 200 learning transitions: 200 immediate
episodes or 25 delayed episodes. Immediate training sides are 100 left/100 right,
shuffled once by the environment seed. Delayed training sides are 13 left/12 right
for seeds 2201/2203 and 12 left/13 right for seed 2202, then shuffled once. The same
sequence is reused across arms. Total learning cap: **2400 transitions**.

No early stopping on apparent success, extra episodes, arm-dependent allocation,
parameter selection, or budget extension after seeing results is allowed in this
registration. Failure is a result. A later change would need a separately declared
development comparison and new held-out seeds.

## Fresh frozen evaluation, not a training-trajectory clone

The existing LearningSession.evaluate method is an isolation diagnostic only. It
copies the training trajectory and MUST NOT supply competence evidence here.
Before these proposed runs, implement a separate fresh-evaluation entry point:

- Construct a new toy environment from held-out seeds 3101–3103 (immediate) and
  3201–3203 (delayed), with 20 balanced episodes per seed: 10 targets per side,
  freshly permuted. Match these side sequences across all evaluated policies.
- Load a fresh pinned visual prior and fresh camera/vision/encoder transient state.
  Carry only the frozen learned readout weights and pinned prior/configuration.
  Never carry training physics, history, RNG, cadence, pending actions or traces.
- Reset all scene histories at each episode. Use deterministic greedy action
  selection (first action on a tie), epsilon=0, with no learner update calls.
- Verify training weights, traces, all RNGs and the full training state remain
  unchanged. Verify repeated fresh evaluation with the same seed reproduces
  outputs independently of training's last scene.

Evaluate both learned arms, the initial zero-weight frozen policy, and a uniform
random policy. Baseline evaluations are shared across arms rather than duplicated:
random-policy action RNG seeds are 4101–4103 and 4201–4203, separate from environment
RNG. All four policies receive identical held-out episode counts and side schedules.
Trained policies: 1080 evaluation transitions total; shared baselines: 1080.
**Overall proposed cap: 4560 transitions**, comprising 2400 learning + 2160 evaluation.

## Reporting, screens and runtime

Report per seed/arm: episodes, correct scores, total reward, transitions, physical
seconds, full process wall seconds, committed/replayed transitions, and snapshot/
I/O costs. Report raw 20-episode counts separately for left and right. These repeat two
fixed templates, so they are not independent visual-generalization trials. Do not
report an IID binomial interval as evidence of visual generalization.

Proposed screens: immediate >=16/20 correct on every seed; delayed >=14/20 on every
seed. These are fixed-template acquisition screens, not proof of visual generalization, lifelong
learning, or Pong learning. The immediate paired arms must have identical complete training trajectories
and final weights, not merely equal scores, under this one-step task.
For delayed credit, report per-seed lambda=.8 minus lambda=0 score differences.
A proposed positive exploratory trace signal is mean paired improvement >=10
percentage points and improvement in at least two seeds; uncertainty and failures
must still be reported. No trace benefit is assumed in advance.

Planning estimate from the existing CPU visual/recovery diagnostics is roughly
0.10–0.25 seconds per transition including conservative overhead allowance. At the
4560-transition cap that is about **8–19 minutes**, plus process startup and artifact
finalization. This is an estimate, not a measured toy throughput result. The recent
34-transition crash/replay proof took 7.84 seconds end to end, including three
process startups. Toy rendering may differ in cost. Record actual costs honestly.

Each invocation/chunk has hard caps of <=200 transitions AND <=180 wall seconds.
If time expires early, checkpoint and report the incomplete allocation; do not
silently run over the cap. Resume chunks from durable state without counting
replay as new learning observations, but charge replay work to compute/time.
Stop/restart gaps stay separate from measured active execution. No background
scheduler, paid compute, public push or merge is proposed. Cloud filesystem
persistence across replacement tasks is not guaranteed by these local proofs.

Before any toy run: independent review of this exact registration and fresh-eval
implementation/lifecycle tests. No Pong competence experiment is included.


## Review amendment before execution

The parent authorized the exact 4560-transition pilot after the required gates
pass and this amendment is committed. This is acquisition of TWO FIXED VISUAL
TEMPLATES, not visual generalization. The target and selected consequence persist
through the delay, so lambda=0 can learn through bootstrapping; the lambda comparison
is exploratory. The countdown is hidden, so no fully Markov value representation
is claimed. Left/right correctness will be reported separately. No target-seed,
parameter, allocation, success-stopping, or evaluation-policy changes are allowed.

Per chunk, reserve 20 seconds for saving within the 180-second subprocess deadline;
use a shorter work limit near the aggregate deadline. The supervisor caps total
wall time at 60 minutes including startup, evaluation, replay and persistence, and
retains/report partial allocations if exhausted. It runs one writer per database,
sequentially. Reset/initial-camera observations and transition-camera observations
are counted separately from new learning/evaluation transitions. No Pong competence
experiment is authorized. The user-selected model and two-thread arrangement are
unchanged.
