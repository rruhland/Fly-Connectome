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

## Six matched short games

| Release | Stock | Fixed hub | Encoder-selected hub |
| --- | ---: | ---: | ---: |
| 10762 | 5 | 6 | 6 |
| 15541 | 6 | 4 | 6 |
| 13344 | 6 | 6 | 6 |
| 5212 | 6 | 6 | 6 |
| 7398 | 0 | 0 | 0 |
| 12246 | 4 | 6 | 6 |
| **Total** | **27** | **28** | **30** |

The two adapted arms preserved all seven ordinary-height hits on
exactly matched far impact and near bounce, and neither arm changed
the score on any of the eleven identical deliveries. Their higher
short-game totals followed earlier changes to the player's own shots
and different actual future deliveries. This is not a proof of a
previously lost easy ball being returned. The fixed hub also shortened
15541 from six to four. The body history selector used the more restrained
third hub on 28 of 30 active recoveries. It keeps the elbow about .05 rad
from stock, instead of the .29 rad change of the fixed hub, while retaining
most of the changed wrist posture. The result does not establish that
switching frequently between hubs is better. [Actual commands and referee records](held.json.gz)

A separate, very small terminal docket reserves the first two completed
older same-law ordinary misses with optical projected height above
1.025 m and one to three preceding learner returns. Their full ordinary
stock prefixes remain fixed. The treatment becomes admissible only after
the already completed preceding strike and still needs the same causal
outgoing cue. No known successful endpoint enters its decision.
[Sealed ordinary-miss docket](ordinary.protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt56_postures fit
python -m scripts.experimental.tt56_postures plan
python -m scripts.experimental.tt56_postures run
python -m scripts.experimental.tt56_postures_terminal plan
python -m scripts.experimental.tt56_postures_terminal run
python -m scripts.experimental.tt56_postures_report --verify
python -m pytest -q tests/test_tt56_postures.py
```
