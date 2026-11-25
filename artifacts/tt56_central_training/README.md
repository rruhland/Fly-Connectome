# Training on actual central ball variation

The two previously coached central movements and the 40 natural optical examples are fixed sources. The new practice roster contains four other releases. The four held releases are different again and do not contribute a template to the new optical actor. The older exact forecourt case 4214 is not used.

Each case starts by replaying the same previously achieved outgoing stroke. Fresh independently seeded standard TT37 draws are admitted by a fixed, generous neighborhood of the requested goal to limit scout cost. They are not clipped or altered. Training eligibility and distance between variants use absolute RGB trajectory estimates on the later inbound flight. At most two optically distinct deliveries per release enter matched actual comparisons. The early shared arm body comes from the archived applied commands, while the begin point is triggered by publicly observed inbound motion. This is normal-attained-body practice, not whole-game qualification.

The ordinary optical and encoder controller runs first. On a missed offer, a bounded true-state teacher compares direct signed candidate paths, then submits at most two. Only an actually landed and physically answered return, independently replayed with strict CAD, can authorize a new observed movement. At least two release-disjoint acquisitions are required before opening the fixed held set. The held controller uses only the actual RGB track, own encoders, signed reference history and the resulting templates. It retargets the learned contact pose by small regularized position and face corrections and retains native model screening after normal control finds no robust option. No target, teacher clock or true plant state enters the deployed comparison. [Preassigned roles and draws](protocol.json)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt56_central_training train
python -m scripts.experimental.tt56_central_training held
python -m pytest -q tests/test_tt56_central_training.py
```
