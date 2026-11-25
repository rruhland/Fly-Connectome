# TT88 · Learn the intact centered arm before asking it to return a ball

The learned inverse reached disjoint, physically attainable body targets substantially better than the same geometry fit followed by a fixed-lead joint schedule. The motor controller receives current joint angles and rates, last issued references, a desired blade patch position, signed normal, world velocity, and a horizon. It does not receive the original program that attained that target.

| Physical assay | Measure | Fixed-lead geometry | Learned drive |
| --- | --- | ---: | ---: |
| 48 goals from 12 untouched babbling episodes | Patch position error | 17.0 mm | 4.7 mm |
| | Face-angle error | 2.86° | 0.60° |
| | Patch velocity error | 0.287 m/s | 0.127 m/s |
| | Composite error | 1.355 | 0.571 |
| Six actual-stock 9254 pre-contact targets | Patch position error | 4.2 mm | 3.9 mm |
| | Patch velocity error | 0.229 m/s | 0.135 m/s |
| | Joint velocity RMS | 1.21 rad/s | 0.67 rad/s |

The learned drive improved 37 of 48 broad tasks. The other 11 include ties or regressions. Four of six 9254 tasks improved in composite error and the two six-tick tasks tied. On all six 9254 windows, the contact-free reduced mechanics reproduced the true stock arm's joint angles to numerical precision. Evaluation used a spot offset from the blade center, including the angular correction to world patch velocity. This is attainment, not a legal hit or game score.

## What is learned

The arm was physically babbled for 640 independent 240-tick episodes from ordinary neutral, for 153,600 issued controls and 1,228,800 native microsteps. Broad reaches, local sweeps, reversals, near-boundary commands and force-limited transient states generated matched encoder, reference and velocity observations. The scene removes irrelevant bodies and contact with the ball, while retaining the original full arm, gripper, wrist/camera mass, freely moving paddle with its real weld, all stock control rates and speed-dependent motor force bounds. A separate seed supplied all held babble episodes. No score, exact ball trajectory, lateral teacher tape or held outcome was used to fit the motor model.

A fixed nonlinear feature map and ridge fit identify coupled next-joint position and velocity. A multistart geometric fit gets plausible terminal joint modes for the requested patch. A small smoother chooses among rate-limited reference programs using the learned plant, without native motor integration at query time. The query took a median 0.13 seconds wall time on the shared host. Compare with TT39, which differentiates new native motor simulations at each query. The present study does not claim to beat exact native shooting. [TT39 inverse-body study](../tt39-terminal/README.md)

Regenerate the large raw tables from the pinned protocol using `python -m scripts.experimental.tt88_inverse_body babble`, then `fit` and `bench` with the standard one-thread EGL settings. The raw tables are intentionally not tracked, with seeds and byte checksums in the training receipt. The small fitted model and all disjoint target/command/error records are tracked.

[Native training receipt](training-receipt.json) · [Body-only benchmark](body-benchmark.json) · [Learned motor](motor.npz)
