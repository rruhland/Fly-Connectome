# A reward-practiced elbow-forward workspace failed the low skill and transfer tests

A separate controller used camera-derived ball flight, current joint encoders and issued references. Geometric inverse/body models supplied feasible depth/face stroke families; actual physical legal landings and misses updated a Beta posterior over 18 depth/face/drive options. No backpropagation, court ID, requested opponent goal or true incoming ball state entered the player. This learns a choice among motor primitives; it is not a learned full joint policy from scratch.

Practice deliberately changed the initial arm posture and partner-request distribution. Two initial transition-hub pilots produced no scored stroke. A subsequent CAD-cleared local contact posture, current-encoder IK initialization, and removal of a virtual-win prerequisite produced actual reward. All prior failures remain archived. The actor then remained unchanged through lower and slower practice stages. Every fourth requested feed stayed broad ordinary. The physical partner, RGB renderer, actuator limits, ball dynamics and referee stayed unchanged.

Across the 24 contact-workspace practice points, six of thirty physical offers scored. All six had evaluator-projected height at least 1.00m. None of ten lower offers scored, although eight of those ten produced a real near-racket touch. The low-reward gap was not solved by reaching/touching the ball or slowing the partner requests. Twenty-six chosen-action outcomes—six positive and twenty negative—were retained; unsupported physical misses were counted separately. Evaluator heights never taught or gated the player.

## Fresh official comparison

The four release-disjoint cards and new opening/reply draws were fixed before outcomes. Every official game began at the normal physical arm state and used the unchanged broad TT37 draws. The original stock controller restored `season.CARRY` exactly. The two other arms used the same new controller, with empty versus practiced frozen reward memory. All twelve points naturally terminated before the 80-stroke cap. All frozen memory digests remained unchanged.

| Release | Stock | New skill cold | New skill practiced |
|---|---:|---:|---:|
|10762|6|0|0|
|15541|0|0|0|
|13344|1|0|0|
|5212|2|0|0|
|Total|9|0|0|

The practiced policy did not improve on the cold policy or preserve ordinary performance. No official online extension or promotion followed. This rejects this specific geometric-primitive/reward-posterior curriculum; it does not prove that all learned working-manifold controllers fail. The controller never acquired true low positive reward, so its practice scores are not evidence of low-ball learning.

All 50 physical points, including both rejected acquisition pilots and all comparison arms, cost 7,058 issued control ticks. Recorded processing clocks sum to 146.933 CPU seconds and 572.507 wall seconds; these are measured through each final point, before teardown. Only one native process was active for this experiment at a time. The initial low-contact proof has not been promoted as a full-CAD independently replayed skill. Detailed counts and per-stage costs are in [report.json](report.json); raw actions, actual referee events, requested and delivered goals are preserved in the compressed receipts.

## Reproduce

Use the checked-in plans; they pin source hashes and all card/seed assignments. For the archived first pilot, use `pilot-source.py` and `pilot-plan.json` rather than the later amended source.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt51_low_contact_practice practice --n 8 --cap 4 --out /tmp/tt51-contact.json.gz
python -m scripts.experimental.tt51_low_stage practice --memory /tmp/tt51-contact.json.gz --out /tmp/tt51-lower.json.gz
python -m scripts.experimental.tt51_low_slow_stage practice --memory /tmp/tt51-lower.json.gz --out /tmp/tt51-slow.json.gz
python -m scripts.experimental.tt51_low_curriculum_compare compare --memory /tmp/tt51-slow.json.gz --out /tmp/tt51-official.json.gz --n 4 --cap 80 --arms stock cold frozen
python -m pytest -q tests/test_tt51_low_contact_practice.py
```
