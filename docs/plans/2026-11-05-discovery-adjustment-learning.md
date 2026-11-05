# Discovery and numerical adjustment learning

User authorized continued orchestration on 2026-11-05 until this next scaffold
is demonstrated on a chosen test case, or they stop work. This builds on the
completed carried structural selector; it does not replace its recorded results.

## Research target

Learn a new numerical controller adjustment from the ongoing stream of public
visual observations, requested actions and permitted rewards, without
backpropagation, privileged physics inputs, expert action labels or a catalogue
of precomputed answers. Retain learned state across games. Prefer adjustments
within a game over waiting for terminal losses. Inherited visual processing and
the acquired anonymous relational binding/feedback prior are explicit scaffolds.

The test must actually challenge the inherited controller. Success at an
unchanged task, accurate prediction alone, or selecting another fixed structure
does not establish this milestone. A suitable changed environment must remain
physically controllable; no universal robustness claim is intended.

## Orchestration

Parent chooses questions, interprets results, reviews implementation and decides
pivots. Two Sol 6.1 XHigh replacements were needed because previous workers were
not live: discovery_execution explores and implements; discovery_challenge
independently challenges mechanisms, extracts outside evidence and audits the
decisive result. Reuse these workers throughout this milestone.

Competing starting routes: visual action-response identification with online
least squares and numerical feedback adjustment; low-dimensional reward-driven
parameter perturbation. Alternatives are welcome when evidence warrants them.
Do not exhaust either route by sweeping closely related parameters.

## First allocation: challenge diagnostic, not learning evidence

Twelve development games: four fresh balanced starts, matched across original
execution, an eight-camera-command FIFO delay (~160 ms), and additive motor
drive bias +0.60. Frozen acquired structure04 and frozen current vision. Each
game has a twenty-second/1,000-transition maximum; total 12,000 transitions plus
12 initial cameras, 240-second global bound, at most four workers. Pin source,
starts and settings before launch. No intervention labels enter the controller.
Store compact receipts/tapes and count incomplete work. This is a discriminating
diagnostic, not a grid or qualification cohort.

## Evidence needed for a completion claim

After development, pin a fresh confirmation with two independent retained
learners, a chosen hidden numeric change, and an unchanged/frozen-learning
control on matched starts. Report all acquisition and recovery mistakes,
first/later returns, time and misses to recovery, numerical parameter evolution,
within-game action effects, ordinary-condition retention and wall/observation
cost. Require reliable final returns (target >=95% first and later, sufficient
later exposure) plus material gain over disabled adjustment. Freeze exact
windows, minimum counts and gains before confirmation; do not extend until pass.
Use a second numerical setting or restoration phase to distinguish a single
hand-set compensation from a general adjustment rule where practical.

Check that only permitted public inputs drive learning, parameters really
persist, checkpoint continuation is exact, and reported opportunity labels
never enter the learner. Primary success is actual play, not provenance volume.
No complete M1B, new vision learning, or arbitrary-game mastery claim follows
from this bounded scaffold.

## Development decision: delayed control

The twelve-game diagnostic completed in 19.94 seconds with 7,461 transition
and 12 initial cameras. Original control returned 22/22; eight-command delay
returned 8/11 (first 4/4, later 4/7); drift returned 7/10. Choose delay as the
first case because it preserves actuator capacity while exposing sustained
control failures. These are small development samples, not reliability claims.

Parent and Sol selected causal finite-impulse-response identification instead of
an autoregressive predictor: quantized measured velocities and recursive poles
could otherwise conceal bad action effects or make forecasts unstable. The
model fits actual observed position differences across four intervals using
elapsed-time-weighted past requested commands. A fixed three-candidate forecast
through the learned model supplies numerical control; it never calls physics.
The 32-lag representation, 16-step horizon, control cost, limited target lead,
confidence gate and command excitation are explicit engineering priors.

Parent independently read Padasip's primary RLS implementation and its
predict-before-update example; challenge worker additionally read NLMS and
eligibility-trace implementations. RLS is a numerical update without
backpropagation. No claim of learning vision, the task objective or the inherited
entity relation is made.

Authorized development pilot: A8 normal -> B16 delay8 -> A8 normal; sixteen
matched B games each for acquired-A frozen numerical coefficients and inherited
servo. At most 64,000 transitions plus 64 initial cameras, 600 seconds, four
cores. Entire public input tapes support offline exact learning replay and
same-history game-start-coefficient comparisons, without extra native games.
Freeze learned coefficients only in the control; ordinary command history and
continued low-rate excitation still advance. Qualification will be selected
only after this pilot; it is not a final success test.

### What would change our direction

If the model predicts observed displacement better but returns degrade, prioritize
control objective/horizon or measurement support rather than another estimator
sweep. If both live and acquired-frozen models handle delay equally well, that is
robustness, not demonstrated ongoing adaptation; choose a genuinely challenging
numeric change before qualification. If prediction itself fails, inspect public
ID changes, wall saturation and insufficient independent excitation. After one
mechanistically justified correction, a second clear failure should trigger a
different route (e.g. reward-driven low-dimensional damping) rather than RLS
parameter tuning. Normal-condition performance is a retention requirement, not
an optional extra to discard after delayed play improves.

## First learner pilot result and next allocation

Pilot r2 (`ce37f57612ea9a462501ede46a566fb35f9d1bf1197f1b2da301e2a26d8a6e75`)
completed all 64 games: normal 48/48, delayed live 80/83, normal-return 48/48;
matched acquired-coefficient-frozen delay 70/76; inherited servo 29/44. Total
49,049 transition and 64 initial cameras, 402.95 seconds. Every numerical game
passed complete public-input replay. This supports numerical acquisition and
transfer, with only a modest continued-update advantage under delay. It is not
final proof of reliably necessary adjustment after a change.

A single follow-up allocation tests gain2*u and affine1.4*u+.4, each with eight
fresh matched live/frozen games from the exact acquired A checkpoint. The
challenge worker proposed the affine variant to retain native extreme motor
force in both directions while changing neutral response; parent adopted it
before pinning. No coefficient/horizon/exploration settings change. At most
32,000 transitions +32 initial cameras, 600 seconds, four workers. Spec:
`cd77d360aaf639cc7a3c3056a20c8cbb92f6741ed0eb92e915cf62f932e045ca`.
If neither needs continued adjustment, do not search increasingly extreme
changes indefinitely: assess cold numerical acquisition under a chosen challenge
and label any transferred competence as robustness rather than change-specific
learning.

### Parent performance improvement (not installed into a running allocation)

`pong_visual_motor_fir_fast.py` batches candidate forecast features, preserving
exact chronological floating-point addition order via cumulative sums and the
same individual dot products. Five hundred random candidate comparisons were
bit-identical. Full public replay of r2 games 0,8,12,16,24,31 reproduced every
action, decision, update and final state exactly with the fast subclass. No new
native observations were used. A 1,000-call forecast microbenchmark measured
0.3590 s original vs 0.0297 s batched (~12x for this component, not total
training). Complete 1,000-step replay took 2.5–2.8 s with the fast implementation.
It may be used in a newly pinned run; active source pins remain unchanged.

## Follow-up result and substantive hypothesis change

Gain/intercept follow-up completed all 32 games: gain2 live48/48 vs frozen46/46;
affine1.4u+.4 live42/43 vs frozen45/45. Cost31,258 transition +32 initial cameras,
112.49 seconds. Neither demonstrates a needed continued adjustment. Preserve
these positive robustness results without calling them adaptation.

Independent Sol challenge proposed a different representational hypothesis:
the scalar-u response model presupposes an ordered up/neutral/down command map.
A three-action cycle that also moves the neutral command breaks that assumption,
while preserving every physical action and adding no reaction delay. Parent
selected categorical/quadratic command-response learning as the next route,
rather than trying increasingly extreme gain, bias or delay values. The learner
will estimate continuous response coefficients for each command, not select
from precomputed mappings. The intervention mapping remains execution-only.
Existing 16 polarity/axis structures cannot express the same full remapping.

First inspect feasibility and causal tests, then pin a short actual A-to-cycle
learning comparison against acquired coefficients frozen. Do not launch a large
confirmation before this route shows actual competence. Keep the control support
gate unchanged initially; do not bundle a visual-tracking repair with the new
representation. The batched forecast optimization is available with exact saved
public-stream equivalence already checked for the scalar model.
