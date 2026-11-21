# Learning where to wait after a changed physical bounce

The unmarked stock clamp and its two fixed cameras stayed in place. A four-degree tilt about the table’s longitudinal axis lowered the tabletop toward positive x. The collision box, all five painted lines, and the net and its posts shared the same rigid rotation. The two cameras retained their calibration from a native level-table RGB frame. This kept the lateral estimates in the arm’s reference frame. [Fixture and audit](../../scripts/experimental/tt21_changed_bounce.py)

The early inbound RGB cue predicted lateral position at the ready depth of −0.88 m. A constant-velocity projection, the existing carried fit, and that fit with one adaptive bias used the same cue. The bias used only four earlier trajectories observed by the cameras. Their later stereo frames bracketed the ready plane, letting the learner interpolate a lateral training value. The estimate stayed fixed on ten fresh trajectories. Simulator coordinates evaluated the forecasts separately. [Optical and physical audit](changed-bounce-audit.json.gz)

| Table | Scorable held trajectories | Median absolute error, constant velocity | Carried fit | Optically adapted fit |
| --- | ---: | ---: | ---: | ---: |
| Level | 4/10 | 17.6 mm | 28.5 mm | 27.0 mm |
| Tilted | 10/10 | 259.4 mm | 250.8 mm | 67.2 mm |

On the tilted table, the bias grew from 106 mm after the first optical example to 184 mm after the fourth. Held stereo targets had 2.8 mm median lateral error against the evaluator. The real near-side bounce changed lateral speed by 0.292 m/s more than the level-table bounce, measured across the ten matched seeds. On the four pairs with a physical crossing on both tables, the tilted ball was a median 263 mm farther toward positive x at the ready depth. The stationary paddle intercepted the other six level trajectories. Both bounces contributed to the accumulated displacement. [Optical and physical audit](changed-bounce-audit.json.gz)

## Held strokes

Nine of ten tilted forecasts improved the implied shoulder target by at least 0.12 rad. This and the 192 mm median improvement cleared the predeclared stroke-test gate. On eight further ordinary serve starts, the two policies used the same frozen stroke planner, command-driven arm model, strong physical rival and strict referee. The experiment applied preparation to the initial incoming serve and measured the first return. [Physical stroke receipts](changed-bounce-strokes.json.gz)

| Ready-position predictor | Shoulder moved | Planned strokes | Physical contacts | Legal first returns | Point wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant velocity | 4/8 | 2/8 | 2/8 | 0/8 | 0/8 |
| Optically adapted | 8/8 | 7/8 | 6/8 | 0/8 | 0/8 |

The adaptive fit gained four matched contacts and lost zero. All six adapted contacts sent the ball sideways, with lateral speeds of 1.11–4.06 m/s at paddle separation and forward speed at most 0.45 m/s. Each point ended with the ball hitting the floor on the learner’s side. All 16 scored branches replayed exactly. [Physical stroke receipts](changed-bounce-strokes.json.gz)

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt21_changed_bounce
python -m scripts.experimental.tt21_changed_bounce --strokes
```
