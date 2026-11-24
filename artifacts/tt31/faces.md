# Switching blade faces during one live table point

The simulated table-edge SO-101 used both faces of the same held paddle on measured opening 7345. It legally landed the first ball using the positive face, received the powered carriage's physical reply, and legally landed that ball using the negative face. The near arm controlled the point from timestamped stereo RGB, local encoders and the paddle's own prior commands. The evaluator only measured the actual face after contact. [Action tape](faces-7345.json.gz) · [Exact replay and CAD audit](faces-7345-audit.json)

| Near-arm hit | Time since launch | Signed blade axis along the table | Contact alignment with blade axis | Contact distance from disk center | Outgoing speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 0.548 s | +0.951 | +1.000 | 35.0 mm | 4.44 m/s |
| Second | 1.956 s | −0.755 | −0.845 | 74.0 mm | 4.59 m/s |

We classified a contact by the projection of the native solver's outward contact normal on the blade's signed local normal, using +0.8 and −0.8 as face thresholds. The second hit was near the rim of the 76 mm disk. Both contacts occurred on the same lateral side of the arm, 0.234 and 0.310 m to the right of the shoulder. The replay reproduced every contact, separation and referee event. Across all 1,543 physics states, the smallest checked CAD clearance was 17.1 mm, and the peak wrist-roll speed was 2.74 rad/s. [Contact geometry and motor measurements](faces-7345-audit.json)

## Receiving stance and controller

The stock planner fits the paddle normal facing the incoming ball from the present motor targets. That fit treats each face as geometrically equivalent. A rated wrist target takes at least 755 ms to travel a full half-turn at 4.16 rad/s. The new stance starts the wrist with the blade near horizontal. After a predicted collision, the planner proposes a return toward that wrist angle and checks the complete action in its private dynamics model. At the next incoming flight it can generate plans from either signed inverse-kinematics solution. It tests recalled trajectories and feasible half-turn variants using the same physics screen. The same shot-scoring objective ranks both orientations. The controller receives no course labels or face assignments, and only a real legal landing adds a movement to memory. [Implementation](../../scripts/experimental/tt31_faces.py)

## Matched development points

The five measured launches and their partner seeds were identical across the three comparisons. The warm and original controllers started each point with the same 223 carried movements. Cold started each point with an empty archive and used the same wrist stance and search as warm. Scores count both players' alternating legal returns, excluding the measured launch. The cap was 14.

| Measured opening | Original | Both-face search, cold | Both-face search, warm |
| --- | ---: | ---: | ---: |
| 5736 | 2 | 2 | 2 |
| 7345 | 14 | 3 | 3 |
| 7271 | 14 | 4 | 4 |
| 11292 | 8 | 0 | 0 |
| 4187 | 6 | 2 | 3 |
| Total | 44 | 11 | 12 |

Original landed all 22 arm returns on the negative face. Warm used the positive face for six of its seven legal arm returns. Cold and warm produced identical actions and events on the first four points, including 7345. On 4187 the warm controller accepted one recalled motor schedule and landed twice on the positive face while cold landed once. The original lasted longer on four of the five development points. [Original](faces-development-original.json.gz) · [Cold](faces-development-cold.json.gz) · [Warm](faces-development-warm.json.gz) · [4187 exact replay and CAD audit](faces-4187-audit.json)

## Reproduce

Use the pinned DeepMind feeds and dependencies. Run from the repository root. [Endurance instructions](../tt29/endurance.md)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt31_faces run /tmp/tt25-rallies.json /tmp/faces.json.gz \
  --mode warm --checkpoint artifacts/tt29/envelope-carry.json.gz \
  --ids 5736 7345 7271 11292 4187 --cap 14
python -m scripts.experimental.tt31_faces replay artifacts/tt31/faces-7345.json.gz \
  /tmp/faces-7345-audit.json --id 7345
```

Use `--mode original` with the movement checkpoint or `--mode cold` without it. A standalone 7345 live run uses `--offset 91538` to account for its ordinal position in the five-point course.
