# M1B stage two: controlled learner correctness and continuation

New reconstruction on stage-one commit b7abbda7cbd05326f162b9f4e15685fedf31f0e9.
No historical M1B implementation or training state was imported. No toy competence
or Pong learning claim is made. The baseline checkout remains untouched.

## Algorithm and boundary

`TrueOnlineSarsa` uses action-block linear values, float64 weights and Dutch
traces. For transition t the TD bootstrap uses outgoing physical discount
`gamma_per_second ** elapsed_seconds`; the Dutch trace uses the saved incoming
discount from transition t-1, multiplied by per-decision lambda. The update includes
the true-online q-minus-q_old correction. Lambda is deliberately documented per
decision; nominal decay is not used to select it.

The next behavior action is sampled once against pre-update weights. The learner
retains that exact action and its observation for execution even if the update
changes the greedy choice. A terminal transition has no bootstrap or next-action
RNG draw. A truncation samples and bootstraps from its genuine final observation,
then clears episode traces/q_old/discount and discards the bootstrap action.
The subsequent reset observation starts a new episode explicitly.

The independent forward-view test recomputes interim lambda returns and replays
updates from initial parameters at each prefix. It does not use Dutch traces or
learner-computed deltas/targets. Cases cover overlapping nonbinary features,
nonzero initial weights, unequal durations, lambda 0/.3/.9/1, terminal and truncated
ends, and alternating action blocks. Additional tests force the post-update greedy
action to differ from the retained pre-update action.

`Pong.step` now accepts `auto_reset=False`; its default remains unchanged for
legacy callers. `PongCamera` uses that mode, stops repetition at the first score
or tick limit, and captures the actual image before an explicit reset. Reward is
score only. Actual advanced physics ticks set elapsed seconds. The global 2,2,3,2,3
cadence phase survives episode resets. The existing visual encoder still describes
nominal .02-second camera cadence; it is not a new variable-cadence visual model.
Only observed image/event information enters the encoder and learner.

## Frozen evaluation and persistence

`LearningSession` pins the bundled model SHA256 and the upgraded consensus-v3
load_default configuration. Source signatures cover control, Pong and visual
modules; a trusted local continuation is rejected if they differ. Both policy
learning and evaluation call vision with learn=False. Evaluation uses a deep copy
of physics/camera/vision/encoder state and a defensive frozen policy copy, and
consumes no training RNG. A test compares the whole original session plus global
Python, NumPy and Torch RNG states before/after nonzero-Q evaluation.

Snapshots retain the complete object graph rather than vision.save: learner
weights, traces, q_old, incoming discount, pending action/features, learner RNG,
physics/body alias, camera history, transient perception, encoder history,
latest encoded observation, global RNGs, source/configuration identity, cadence,
transition/episode/tick counters and physical elapsed seconds. SQLite commits
sample and active snapshot together under FULL durability. Stage-one tests cover
SQL rollback after sample insertion, stale writers, corrupt chain/state/artifacts,
and abrupt process exit in an uncommitted transaction.

## Exact scope of recovery evidence

Stage ONE demonstrated only frozen-policy continuation. It had no learner updates,
active eligibility traces, q_old, or retained learning action at checkpoints.
It cannot be cited as a learning-state recovery proof.

Stage TWO compares 32 uninterrupted with 16+16 fresh-process resumed transitions:
64 integration transitions, 12.37 seconds end-to-end. Exact audit samples and
canonical graph hashes match. Midpoint: 259 nonzero weights, 134 active trace
entries, q_old 0.00020774052793901609, pending action 0. End: 446 nonzero weights,
152 active trace entries. The controlled fixture forces an initial score and a
31-physics-tick episode cap, ensuring both terminal and truncated lifecycle paths.
This is synthetic setup for a recovery diagnostic, not a reward curriculum.

This demonstrates real updates and active learning-state continuation on this
Linux runtime; it does not demonstrate learning competence, Windows portability,
or power-failure behavior of the actual filesystem. Canonical graph comparison
covers object aliases including an explicit body alias; tensor cache version
counters use validity equivalence, as documented in stage one. It is not a general
tensor-storage-view alias proof. The learning process exits at a committed boundary;
abrupt mid-transaction rollback is tested separately at the journal layer.

Stage-two focused checks: 39 passed in 1.63 seconds. No broad or toy training ran.
The proof plus executed lifecycle/evaluation tests remain below 200 integration
transitions; the standalone proof enforces a 180-second subprocess deadline.

## Still pending before experiments

- Independent review of this reconstruction and its control/continuation boundary.
- Immutable visual-bank deduplication (active pickle still embeds banks), interval
  snapshots/delta replay, persisted stop/restart timing gaps and total process costs.
- Re-establish fresh-seed toy learning evidence before a registered Pong experiment.
- Resolve or explicitly scope the baseline native/reference exact-rounding failure;
  never quietly relax its assertion. Merely exposing forecasts proves no benefit.

Development experiment observations remain 0/0. Controlled diagnostic updates are
reported above and must not be mistaken for absence of all numerical learning.

## Full suite

**769 passed, 4 skipped, 1 failed in 46.90 seconds.** The sole failure remains
`test_native_neural_step_matches_reference[False]`, the baseline-only native
voltage exact-equality mismatch already reproduced before reconstruction.
No new full-suite failures occurred, and that assertion remains unchanged.
