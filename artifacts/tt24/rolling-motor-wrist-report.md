# Continuing the Pong motor learner during altered-arm wrist play

Refitting the saved rolling estimator during ordinary wrist play reduced performance on the eight unexposed drops. The adaptive map averaged **1.625 consecutive rebounds**, versus **2.875** for the frozen Pong map and **10.50** for the evaluator-supplied map. The two learned conditions reached no ten-rebound streaks. The supplied map reached ten on four drops.

| Fresh held wrist drops | Adapted during wrist | Frozen Pong map | Evaluator-supplied map |
|---|---:|---:|---:|
| Drive 0, mean consecutive on four drops | 2.00 | 2.50 | 10.25 |
| Drive 1, mean consecutive on four drops | 1.25 | 3.25 | 10.75 |
| Pooled drops with a second qualifying rebound | 2/8 | 4/8 | 6/8 |
| Pooled mean time after release, capped at 8 s | 0.91 s | 1.88 s | 4.15 s |

The committed estimator refit after all twelve adaptation releases. Its evaluator matrix error rose from 13.32% to 16.51% on drive 0 and from 5.29% to 15.44% on drive 1. The estimator received only the public requested rates before wiring and the next returned encoders. The physical wiring and rebound scores remained outside the learning path. During each release the previously fitted matrix remained active. The learned estimate changed only at the game boundary.

## Exposure and prior memory

| Adaptive practice, six releases per drive | Drive 0 | Drive 1 |
|---|---:|---:|
| Actual requested-rate and next-encoder samples | 848 | 534 |
| Ordinary held-ball visual preparation | 2.47 s | 2.42 s |
| Actual game time after release | 11.64 s | 6.43 s |
| Games shorter than one second after release | 4/6 | 5/6 |
| Pong blocks displaced in the rolling store | 7/12 | 6/12 |

All preparation, released-play, and terminal transitions entered the estimator. One fourteen-tick game supplied thirty physical command samples when the ordinary prep was included. No special calibration movements were added. The measured requested-displacement windows spanned all five directions when pooled within each drive. The weakest-to-strongest singular-value ratios were 0.387 and 0.334. The pooled span coexisted with many short individual games.

The fixture independently held and released the ball with the same altered arm, camera, gravity switches and visual preparation protocol for each paired condition. Each drive used six actual adaptation games followed by four unexposed starts. All 60 adaptive, frozen and supplied-map tapes replayed exactly. Both saved checkpoints retained the native source head and original map. Their respective held native Pong tapes reproduced two returns each, and the original stock context reproduced six archived physical wrist tapes.

Reproduce with `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_rolling_motor_wrist`. [Physical course and public motor samples](rolling-motor-wrist-course.json.gz)

A separate chronological comparison checked each proposed update against the following release before applying it. [Held performance and causal audit](rolling-motor-wrist-guard-report.md)
