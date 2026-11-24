# Registering moved external cameras

The registration estimates each camera’s position and orientation from one ordinary RGB frame before play. It finds the regulation table’s four white boundary lines next to the blue playing surface and fits their projections using the known line dimensions. The focal length and nominal mount come from the original survey. The recovered calibration gets its new pose from image pixels. The known-geometry comparison uses the evaluator’s actual pose. [Registration code](../../scripts/experimental/tt29_registration.py)

The held test moved each eye independently across 12 stereo configurations. The test sampled lateral, depth and height shifts up to ±110, ±130 and ±75 mm, plus angular shifts on all three axes up to ±4.6°. The four table boundaries remained visible enough to register in all 24 images at 384 × 288. Reprojection error was measured over a grid through the ball’s playing volume above the table. [Held perturbations and per-eye errors](court-registration.json.gz)

| Calibration | Median position error | Median orientation error | Median reprojection error |
| --- | ---: | ---: | ---: |
| Original survey applied after the move | 96.94 mm | 3.543° | 13.080 px |
| Recovered from the court RGB | 5.32 mm | 0.098° | 0.249 px |
| Evaluator’s known geometry | 0 mm | 0° | 0 px |

The largest recovered errors were 11.27 mm in position, 0.241° in orientation and 0.551 px in reprojection. The estimator assumes unchanged lens intrinsics, a nominal mount close enough to find each boundary, and a visible table with white lines against a blue surface. The cameras were moved between points. [Tests](../../tests/test_tt29_registration.py)

## Native live-points comparison

The same moved stereo rig, robot, powered carriage and native referee were used in all three calibration branches per opening. The optical controller started with the same 40 frozen stroke examples. Neither measured opening appeared in that checkpoint. Each branch stopped at the referee’s decision, a 280-tick horizon, or four alternating legal returns after the measured opening. The registration reads only the pre-point court frame, while the player receives live stereo RGB and encoders. [Frozen stroke examples](camera-action-memory.json.gz) · [Action tapes and referee events](court-registration.json.gz)

| Measured opening | Original survey | Recovered from RGB | Known geometry |
| --- | ---: | ---: | ---: |
| 4527 | 0 | 4 | 4 |
| 8207 | 0 | 0 | 0 |

On opening 4527, the recovered and known calibrations both reached the four-return cap, including two learner returns. The original survey produced a paddle hit that landed on the learner’s half. On opening 8207, the recovered and known calibrations both ended at the same tick with a miss. [Native referee events](court-registration.json.gz)

Run the calibration assay and live comparison from the repository root.

```sh
MUJOCO_GL=egl PYTHONPATH=.:src:scripts/experimental LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m scripts.experimental.tt29_registration --feeds /path/to/rallies.json
```
