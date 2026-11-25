# Can another fixture teach the stock body?

The six-degree pitched clamp supplied nine physically landed, answered and CAD-clean training strokes from three release IDs. An equally sized stock-clamp teacher used the same source IDs. All six source point tapes replayed the native event logs exactly. The learner's body was restored to the original flat end clamp for every comparison.

The teachers store achieved blade plane, offset from the ball and world-frame blade velocity at contact. On a different measured stock-ball flight, a privileged *training* coach reexpresses those contact relations for the receiving ball and solves for the original joint limits. It compares bounded short joint schedules with identical 144-path, three-round budgets. It does not transplant the original joint path or the shifted fixture. The four test releases are ordinary stock failures, independent of the teachers' three source releases.

Neither teacher produced a modeled legal path on any of the four stock-body practice balls. The stock teacher found 0, 2, 5 and 0 geometry seeds. The altered-body teacher found 1, 1, 3 and 1. A geometric seed without a legal candidate authorizes no actual retry or learning credit. These fixtures therefore did not transfer useful motor instruction through the tested contact representation. The stronger result from physically moving the whole clamp is separate.

[Frozen teacher roster](protocol.json) · [Self-contained source command tapes](lesson-tapes.json.gz) · [Replayed measured contact lessons](lessons.json) · [Restored-stock results](original-body.json.gz)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt63_fixture_teacher fit
python -m scripts.experimental.tt63_fixture_teacher run
```
