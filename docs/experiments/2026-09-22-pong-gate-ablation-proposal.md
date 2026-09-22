# Proposed bounded always-open local forecast and credit test on Pong

## Decision requested

Approve one opt-in full-graph rule ablation: remove the controlled-task
timing window from **both** issued forecast expression and issued eligibility
for the 2,660 directly observed cells. The prediction remains their existing
physical signed recurrent current, clipped by the same threshold; the
context-selected magnitudes still control that current. At the next camera
frame (+8 neural ticks), the same locally observed ON/OFF/quiet event, prior
local event-count gain, and issue-time eligibility update only the issued
context component on the same 17,557 measured edges. Magnitudes remain in
[0,10]. All other edges are frozen. No neuron, connection, sign, delay,
kernel, reward-driven visual update, backpropagation, or Pong-state neural
input is added. Keep production defaults and checkpoints untouched.

This is a test of whether the previously helpful periodic-motion gate is
blocking learning under irregular Pong timing. On the frozen 500-frame Pong
stream, it opens at only 27/612 ON and 47/610 OFF event targets; even perfect
unit forecasts at every open event could average only .044/.077 anticipation.
Credit-only ungating while retaining that expression window cannot meet the
.1 event criterion. Removing both gate applications is therefore the smaller
decisive experiment. It uses no new temporal representation.

## Fixed experiment and checks

First verify zero-learning parity of spikes, physical currents, raw forecasts,
edge eligibility, event targets and sensory state against the approved
full-graph candidate. Only the intended forecast/credit gating differs.
Fixture tests should verify positive/negative local contexts, fixed edge
sign/order, eight-tick confirmation, closed-window surprises now receiving
local credit, quiet confirmations, component-only updates, bounds and exact
resume. The original gated run remains an immutable control.

Use the same bounded source checkpoint, one explicit initial clip of its
single over-bound edge, 500 training frames on development seed 1101 at the
previously selected eta 1.0, and 500 frozen initial/trained frames on seed
1102. Do not tune on seed 1102 or touch final seeds 1201-1204. Continue neural,
camera, eligibility and weights across Pong rallies, with the existing
scripted M1A paddle and eight neural ticks/frame. Save exact-resume state
every 100 frames; measure speed and update reconstruction.

Report next-frame ON and OFF anticipation and event MSE separately, whole
and quiet MSE versus zero and persistence, and two quiet-alarm rates at
|forecast|>=.1: all quiet cell-frames and **event-adjacent quiet** cell-frames
(same cell has an event one camera frame before or after the quiet target).
Also report changed neural spikes and stable firing. Whole-field improvement
alone is insufficient because nearly all targets are quiet. Require both
event signs to improve meaningfully, with the prespecified .1 anticipation
threshold and no more than 5% event-adjacent quiet alarms, before any longer
Pong run. If the ablation fails or becomes unstable, stop: the next decision
is a new local timing/credit mechanism, not another blind rate or duration
sweep. M1A and M1B remain blocked until their original acceptance checks pass.
