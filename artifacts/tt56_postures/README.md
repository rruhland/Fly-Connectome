# Broad learned receiving postures

The candidate postures are learned from 336 physically scored movements,
with 111 left, 123 center, and 102 right optically observed arrivals.
Their population is not restricted to low balls. The fitted compromise
reduces a benchmark joint-travel proxy from .179 to .150 seconds. The
best pose is modest, primarily a less folded elbow and a changed wrist
pitch. All of the observed pre-contact blade normals faced the same
direction, so the library does not claim a demonstrated reverse-face mode.
[Observed encoder samples and selected postures](postures.json)

## Why this is a distinct comparison

TT31 compared geometric fixed starting poses and found a marked
left-vs-right tradeoff. TT37 built a lateral inverse-kinematic pose
surface. Later alternatives used a low-shot teacher or a preview of
the next direction. Here, the scored motion library supplies actual
achieved poses for a ball-agnostic return-to-ready action inside a point.
The initial stock home, camera rig, physical clamp, direct regulators,
grip, and normal incoming controller are unchanged.

The fit selects public encoder states from early and middle phases of
scoring approaches. Separate encoder states close to those earned
contacts grade the estimated travel time. Equal lane representation
penalizes a posture that shortens only one side or one common trajectory.
A static route screen excludes obvious collision paths. The empirical
balance score is an actuator-distance proxy, not a native legal-return
result. The figure can be optimistic as it does not model acceleration
or how the ordinary MPC chooses its stroke from a new posture.

During play, after the same incoming estimate used by stock, two fresh
consecutive outgoing readings can open a short bounded direct move.
An ordinary already leased stroke retains authority. The move stops
at the first supported incoming reading or its short time limit.
The fixed alternative targets the best hub. The portfolio alternative
uses only recent own encoder motion to choose the closest suitable
hub, with the same source balance score. Neither uses the requested
opponent target, a privileged ball state, or forecast direction or
height of the next ball. All late native stroke selection remains stock.

The comparison uses six preassigned distinct standard wide random
evaluation streams, none in the training bank. Each ordinary point
ends at the unchanged native loss or the diagnostic cap of twelve
alternating returns. A fresh stock replay checks compatibility with
the archived control reference. Later physically changed paths may
produce different actual opponent deliveries even with the same
requested random draws. The paired effect on an identical delivered
ball is therefore logged separately from branch-dependent total scores.
[Frozen comparison and unchanged controller inputs](protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt56_postures fit
python -m scripts.experimental.tt56_postures plan
python -m scripts.experimental.tt56_postures run
python -m scripts.experimental.tt56_postures_report
python -m pytest -q tests/test_tt56_postures.py
```
