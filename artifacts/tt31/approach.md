# Retrying a missed stroke after moving toward the ball

We compared the original receiving controller with a fallback that moves toward a forecast intercept after the original planner fails. Both controllers used the stock lateral stance, the same frozen movement and placement memories, stereo RGB and the same physical opponent seeds. The test used three cards preselected for their delivery schedule. The scores include both players' alternating legal returns following the measured launch, up to four per point. [Practice-card manifest](course-plan.json.gz) · [Implementation](../../scripts/experimental/tt31_approach.py)

The original planner still had priority. After an unsuccessful search on an incoming RGB track, the fallback used the camera forecast, encoders and local control targets to move the blade toward the estimated interception point while preserving its current normal. Each jog lasted five control ticks, stayed within the rated target slew and had to pass a short nominal physics check. The original planner could take over at the next search. At most four jogs were tried per incoming leg, and a jog required at least 145 ms of forecast travel time to the interception point.

| Opponent card | Observed deliveries | Original returns | Returns with fallback |
| --- | --- | ---: | ---: |
| Fixed, seed 310810 | Right, right | 4 | 4 |
| Periodic, seed 311010 | Right, left | 3 | 3 |
| Periodic, seed 311011 | Left | 1 | 1 |

The fixed-point actions were identical. The right-first periodic point reproduced the same right-side stroke before taking four jogs toward the left reply. The nominal target-position error fell from 386 to 149 mm. The two left approaches still missed the paddle and produced zero legal arm returns. In the first-left point, the incoming ball struck the moving wrist camera foot at 1.465 s after the jogs ended. The original point ended with the incoming ball going out. The six exact replays checked 8,141 physics states and found zero penetrations among the arm, paddle and table pairs in the CAD screen. The ball-to-camera fixture contact was a separate referee event. [Paired action tapes, jog measurements and full CAD replays](approach-development.json.gz)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt31_approach /tmp/tt25-rallies.json \
  artifacts/tt31/course-plan.json.gz /tmp/approach-development.json.gz
```
