# A tip-driven native Pong joystick identifies a dense actuator remap

On twelve fresh dense-remap wrist drops, an actuator estimate learned from ordinary native Pong raised mean consecutive rebounds from **1.17 to 9.42**. It won eleven paired comparisons and reached ten rebounds on seven drops. The estimate from the earlier single-direction joystick and the shuffled tip-driven pairs both stayed below three rebounds.

| Dense remap, twelve matched wrist drops | Mean consecutive | Ten or more | Longest |
|---|---:|---:|---:|
| Stale identity | 1.17 | 0/12 | 2 |
| Earlier one-direction estimate | 1.17 | 0/12 | 2 |
| Shuffled tip-driven source pairs | 0.75 | 0/12 | 2 |
| Matched local tip-driven pairs | 9.42 | 7/12 | 17 |

All 48 dense wrist traces reproduced their full referee events on exact physical action replay. The policy used the carried wrist head and optics and performed no wrist calibration or practice. The original body context stored its original map separately and reproduced all six previously recorded stock action tapes after the full dense fit. This study covers one physical installation and one fixed dense remap, with independent source and target starts.

## Five-direction ordinary play from one native control

A single rail projected the position of one point on the rigid paddle tool onto a scalar native joystick. The point was 60 mm off the tool axis, so wrist roll could move the joystick. The scalar came from measured arm encoders and public vendor kinematics. The native ball, opponent, visual raster, contacts and scoring stayed in the original Pong engine. The visual paddle controller sent a scalar drive, and the arm requested normalized motion along the local joystick gradient. All requests originated from the native visual game. Starting postures varied independently within a comfortable public kinematic range.

The arm made 35 legitimate returns on 13 of 18 source points, versus four returns on three points with a stationary arm. The active arm won five points and lost none. The stationary arm won five and lost twelve. Exactly replaying each native action tape reproduced its physical encoder trace and point result. The observed five-channel command singular values were 6.262, 3.508, 1.397, 0.311 and 0.221.

A first fit used filtered cumulative positions. Its source holdout error was 0.0369 rad against the 0.020-rad transfer threshold, so it never ran on wrist. Independent forensic replay found paddle-to-table contacts in five source points and zero actuator target limit frames. The shoulder row carried 191% relative error, while the other rows were within 1.6–6.4%.

## Fit across short command windows

The local fit used only 14 existing ordinary source points. It estimated each encoder row from eight-tick changes in joint angle and the corresponding completed command increments. A causal filter and robust residual weighting were selected from source command and encoder records. Windows with little measured encoder variation carried no weight for that row. The shuffled comparison rotated measured window changes within points. The fitter received neither table-contact indicators nor the true actuator matrix.

On six fresh natural source points, the arm returned the ball on every point, including one win. The matched fit's held per-joint 70th-percentile error was 0.24–1.11 mrad versus 6.32–36.67 mrad for the stale map. The weakest-to-strongest singular-value ratio on held usable windows was 1.40–1.41% across the five rows. The evaluator found 3.53% relative full-map error, with 7.23% on the shoulder row. The held encoder and matrix checks were complete before wrist testing. True geometry and the hidden remap remained evaluator measurements.

The same physical arm and command interface operated in both games. The transferred quantity was the response of the arm's actuator channels. It came from ordinary gameplay motions and served as a fixed motor decoder alongside the existing wrist bouncing policy.

Reproduce with `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_pong_arm_tip`, followed by `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_pong_arm_local`. The tip module's `--forensics` option audits earlier action tapes without invoking the fitter. [Earlier one-direction report](pong-arm-report.md)
