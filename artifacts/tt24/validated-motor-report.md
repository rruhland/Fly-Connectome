# Validate the motor map, then move only when calibration fails

An independent encoder movement detected most severe calibration errors. When the first map was unreliable, moving the same arm away from the poorly observed pose and fitting a fresh local map removed eight of ten severe errors on a new 56-case mechanical roster. Ordinary accurate maps stayed unchanged. The separate live game course produced fewer rebounds, and every inspection in that course was already accurate, so it did not exercise relocation.

## Physical mechanical assay

| Independent held-ball calibration | Original sampler | Validated sampler with relocation |
|---|---:|---:|
| Mean relative map error | 13.62% | 3.91% |
| Median relative map error | 0.297% | 0.252% |
| Mean decoded-command error | 21.61% | 5.47% |
| Maps above 20% error | 10/56 | 2/56 |
| Mean charged time | 2.03 s | 3.48 s |

The 56 matched cases covered 28 wide stock and 28 wide altered setups with independent unknown 5×5 command transformations. The table, clamp and held ball remained in the simulation. Relative map error measures the Frobenius distance from the hidden evaluator matrix, normalized by the true matrix norm. Decoded-command error measures the achieved command against identity after applying the same regularized inverse used in the game. The hidden matrix never entered calibration.

The check added a median 32 ticks, or 0.53 seconds, on the 47 inspections that did not relocate. All 45 initially accurate maps below 10% error were reproduced exactly. The largest difference from their original final joint angles was 0.000385 rad, about 0.022°. Relocation ran on nine flagged cases and added a median 332 ticks, or 5.53 seconds, relative to the original inspection. Eight relocated maps passed the final check. The maximum charged inspection, including one that relocated successfully, was 631 ticks, or 10.52 seconds.

A retrospective contact replay found paddle-to-table contact on 14 of 15 severe original inspections across the two rosters. The other case pressed the paddle against the clamp apron. The selected signs moved the missing encoder direction by no more than 0.00016 rad in that case, and the recovered map failed validation. The other remaining severe error had only 1.2% error on the single independent direction and escaped the initial gate.

## What the learner measured

The initial sampler stayed the same. The independent check sent a diagonal command and its reverse and compared both measured encoder displacements and return closure to the fitted map. It also tested the fitted map's gain range for poor conditioning. If the check failed, the arm physically tried both signs of the least-observed input direction and selected the pose with larger movement in the corresponding weak encoder direction. From that pose it fitted a row-wise robust affine model to fresh mirrored endpoints. A new diagonal move checked the recovered model before it could replace the original. Real commands returned the requested offset to zero. The learner used the requested commands and encoders throughout.

An earlier local-only version used redundant mirrored samples around the initial pose. On its separate 56-case cohort it repaired one of five severe maps and increased mean decoded-command error from 10.21% to 11.47%. Replaying relocation on those five development failures reduced map error from 35.3–77.9% to 0.11–0.19%. The measurements in the table above came from the different, independently seeded roster.

## One integrated wrist course

| Fresh 12-game course | Original | Validated |
|---|---:|---:|
| All verified rebounds | 226 | 219 |
| Rebounds on changed-interface games | 109 | 97 |
| Rebounds on stock games | 117 | 122 |
| Inspection time | 8.15 s | 12.10 s |
| Total device time with inspections charged | 96.27 s | 97.10 s |
| Time after release with cap | 82.42 s | 79.30 s |

Each alternative used its own motor memory under the same sequence of held physical releases, unknown map changes, camera pipeline and game policy. The ball stayed held through inspection, and the game began from the actual post-inspection joints. There were six inspections per branch. All six maps were already accurate at 0.17% mean error and the relocation fallback was never used. Both monitor branches missed the same unsignaled interface change before one game loss. This course tested the always-paid validation overhead under normal inspections. All 24 condition tapes replayed exactly, including calibration commands, terminal joint state, gravity changes and referee events.

Reproduce the mechanical runs with `python -m scripts.experimental.tt24_validated_motor` and with `--relocate`. The `--development` and `--contacts` options reproduce the development and forensic receipts. Reproduce the integrated course with `MUJOCO_GL=egl python -m scripts.experimental.tt24_validated_motor_course`.
