# Learn on the supported +12 cm lateral clamp

The training teacher starts from the original six-joint mechanism mounted
12 cm to the side, as surveyed in TT63. It preserves the jaw, blade, table,
motor limits and continuously randomized reactive opponent. This is a
separate mechanical condition. Nothing changes in the original-body
benchmark. Its teacher first executes real scored openings, then visits
the next ordinary offered ball after several different physically earned
strokes. [Independent practice roster](protocol.json)

The teacher may use native state in isolated training. A proposal enters
the learning bank only when the real referee scores it, the unchanged
opponent makes a legal reply, the original actual incoming strike and
bounce are the same, and a full independent issued-tape CAD replay is
clean. TT63 contributes only its five distinct successful development
release tapes. No TT63 held result or TT68 point enters the new school.
[Actual practice commands and referee tapes](practice.json.gz)

The learner indexes issued direct joint-reference trajectories by observed
stereo flight phase, not referee or recorded game time. Measured joint
encoders and its own prior reference constrain the reachable command.
The optical receiver first assesses the observed paddle-side arc in its
independent calibrated dynamics model. The context-matched comparison
tests direct optical phasing alone and a small pose transport that uses
no live ball state or opponent target. A physically recorded actuation
path must be supportable by a current RGB trajectory and reached body.

The stricter, separately frozen qualification uses independent measured
starts from the TT39 assessment pool, excluding the original carry
release IDs, prior raw table training, TT63 practice and held, all TT68
roots and all newly taught releases. Fresh reply seeds follow the full
unchanged law without target shaping. The first completed receive on
each of twelve releases is compared against TT68's registered, carried
controller. An actual additional matched legal answer with no losses of
stock first hits is required before any full-game comparison. All points
count and the same original-condition benchmark stays untouched.
[Frozen RGB qualification](qualification.protocol.json)

When qualified, both arms of independent whole points use the live TT68
reward callback to update their own carried bank. The new lesson field
stays frozen. Whole-game gains and changes in score on identical delivered
balls are logged separately.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt69_clamp_school plan
python -m scripts.experimental.tt69_clamp_policy plan
python -m scripts.experimental.tt69_clamp_school run --n 22
python -m scripts.experimental.tt69_clamp_school run --n 22 --next
python -m scripts.experimental.tt69_clamp_policy fit
python -m scripts.experimental.tt69_clamp_policy first --n 12
python -m scripts.experimental.tt69_clamp_policy games --n 6
```
