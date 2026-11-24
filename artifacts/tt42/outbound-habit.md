# A short-flight preparation learned from camera-timed action

The outgoing ball can expose a short far bounce before the opponent moves
enough to reveal its intent or returns the ball. The cue is not a reliable
classifier of low returns on its own. The previous fixed warm pose reached
the three difficult practice balls in the private screen but did not land
any, and unconditional preloading shortened several physical rallies. This
study treats early preparation as a reversible motor bias and leaves the
incoming direct player unchanged.

## Preparation and evaluation boundary

Three native practice returns, on different releases and opposite faces,
had a real legal landing, a legal opponent answer, and no full-CAD
penetration after the cue-delayed intervention. Six fresh stereo-fit
exposures per release were replayed. Each supplies an outgoing ballistic
projection and the following 32 ticks of issued reference movement
relative to the actual starting encoder reference. The actual contact and
referee outcomes certify successful practice, but neither times a future
stroke nor enters a model feature.

The live routine requires a recent visible inward passage below y = −0.79 m,
then a fresh outward fit with the ball still on our side and a projected far
bounce between y = 0.05 and 0.65 m. An eight-variable local comparison uses
only the ball track and its projected bounce. At most one observation from
each practice point contributes. Weighted early motor trajectories provide
a ramped residual atop the normal direct reference. Unfamiliar situations
abstain. Pan and blade roll require a strongly supported exemplar. All
axes have bounded travel, rate and slew during the preparation. The first
fresh inbound fit or a fixed timeout returns control to the same direct
joint field. No private physical model, seed, release assignment, score
event, future opponent aim or contact time enters the runtime routine.

The assessment holds the original 24 measured replication starts and all
reply seeds fixed, with no filtering by opening score or preparation cue.
The exact same frozen camera-to-joint field supplies the stock comparison.
For each changed point the stock action tape is independently audited in
the same physics. For unaffected points the full actions and events match
the archive. Commands before each point's first intervention must match
stock exactly. The normal reactive opponent keeps drawing target placements
through the whole point. All integration steps are checked against full
articulated CAD without changing the dynamics.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt42_outbound_habit train
python -m scripts.experimental.tt42_outbound_habit freeze
python -m scripts.experimental.tt42_outbound_habit evaluate --resume
python -m scripts.experimental.tt42_outbound_report
pytest -q tests/test_tt42_outbound_habit.py
```
