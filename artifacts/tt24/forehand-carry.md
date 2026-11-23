# Table-tennis motor experience under remapped wrist actuation

A fixed wrist controller previously learned to bounce on three physically different hinged donor arms using rebound reward. This experiment asks whether prior table-tennis play helps that controller recover when the stock arm receives commands through an unseen actuator mixing matrix. The wrist controller and public RGB/encoder estimator are frozen throughout. Only the interface inverse can change. [Reward-trained wrist controller](tt24_donor.md) · [Public estimator](public-body.md)

In 12 paired MuJoCo drops, the table-tennis transition model plus one or two short probes recovered 6.25 and 7.67 consecutive rebounds on average. A model fitted from the same probes without table-tennis experience averaged 9.50 and 8.42. Ordered source data helped against the shuffled-history control at both budgets. The matched probe-only comparison gives no evidence of a table-tennis experience advantage for this actuator change.

## Source and held design

The source consists of 24 complete live table-tennis points with 26 legal learner returns and 1,801 command transitions at 30 Hz. The table-tennis pupil used two camera views and issued every learner command in these points. Its previous training included demonstrations. All 24 source command pairs reproduced the archived terminal tick and full referee events when reissued. The motor fit uses the actual issued learner commands, their integrated servo targets, and successive own joint angles and rates reconstructed by that replay. Its targets contain no teacher actions, observed ball information, scoring events or task rewards. The shared nominal channel order is pan, shoulder, elbow, wrist flex and wrist roll. Joint limits and normalized rate units are supplied. [Table-tennis source and condition metadata](rally-improve-combined-held.json.gz) · [Source transitions and coefficients](forehand-carry-source.npz)

A linear predictor estimates encoder displacement over the next 30 Hz interval. A six-point source audit compares that predictor against previous-displacement persistence and a fit with command-bearing observations jointly shuffled. Paddle position and the two horizontal normal components in the audit come from projecting the predicted and measured own joint motion through the same previously calibrated public model. These pose errors measure consistency with that model. The experiment uses no independent racket tracking labels from the table-tennis cameras. The fixed public body model used eight earlier stock stereo RGB/encoder calibration drops, which are separate from the target actuator probes. The held wrist controller uses own joint measurements and a color-defined ball feature from 192 × 144 wrist RGB at 60 Hz.

| Motion predictor on six withheld table-tennis points, 302 transitions | Encoder displacement RMS, mrad | Estimated paddle displacement RMS, mm | Estimated paddle normal-change RMS |
| --- | ---: | ---: | ---: |
| Previous displacement persistence | 38.12 | 9.50 | 0.295 |
| Shuffled table-tennis command history | 24.42 | 6.29 | 0.231 |
| Coherent table-tennis command history | 20.43 | 5.42 | 0.245 |

The held course has six unseen, dense 5 × 5 actuator mixes and two fresh paired drops per mix. Arm geometry, gravity and drive retain the stock bounce configuration. The table-tennis and bounce fixtures start from different wrist orientations. Each probe lasts 52 physical control frames, about 0.87 s. It requests five orthogonal directions using short balanced pulses in a contact-free arm reset. The second probe reverses the first from the same starting state. All fit methods see the same requested commands and encoder histories, with no RGB, ball, rebound or map labels. Zero probes leave the inverse at the nominal identity under every source condition because the newly drawn mixing has no observable signal yet.

The coherent and shuffled fits use their respective source transition model to estimate the new map. The probe-only control has no source transitions. It estimates a generic first-order servo lag and mixing matrix from those same probes. All three use the same damped inverse and bounded commands. The exact-map diagnostic is supplied the true inverse and still clips requested and physical actions. The nominal diagnostic keeps the ordinary actuator wiring. Every scored active branch issues new commands from live wrist RGB and encoders. Each resulting command stream is separately replayed for the referee audit. [Assessment code](../../scripts/experimental/tt24_forehand_carry.py) · [Full held commands, probe data and physical results](forehand-carry.json.gz)

## Held physical rebounds

| Controller | Short probes | Mean longest streak | Maximum | Drops with at least 10 |
| --- | ---: | ---: | ---: | ---: |
| Stationary on remapped arm | 0 | 2.50 | 3 | 0/12 |
| Frozen feedback with nominal inverse on remapped arm | 0 | 1.42 | 3 | 0/12 |
| Table-tennis transition model | 1 | 6.25 | 13 | 4/12 |
| Table-tennis transition model | 2 | 7.67 | 15 | 6/12 |
| Shuffled table-tennis model | 1 | 1.42 | 3 | 0/12 |
| Shuffled table-tennis model | 2 | 1.08 | 2 | 0/12 |
| Probe-only identification | 1 | 9.50 | 19 | 7/12 |
| Probe-only identification | 2 | 8.42 | 19 | 4/12 |
| Feedback with exact inverse on remapped arm | supplied | 12.92 | 18 | 10/12 |
| Feedback on nominal arm wiring | 0 | 12.08 | 18 | 9/12 |

The table-tennis model beat its shuffled control on ten of 12 paired releases at each budget. The probe-only fit beat the table-tennis model on eight of 12 releases with one probe and seven of 12 with two. Average relative Frobenius error across the six maps was 0.637 and 0.646 for the table-tennis model, 1.598 and 1.994 for the shuffled model, and 0.696 and 0.684 for probe-only identification. The probe-only fit scored more rebounds at each budget despite its higher mean map error. Releases are paired within six map draws, so map generalization is assessed on six distinct cases.

Both short probe clips share a physical reset pose and velocity for each map. All 120 assessed command tapes reproduce the terminal tick and complete rebound metrics, including the physical events. The wrist readout and public observer hashes match the frozen inputs. The stored commands also reconstruct the applied actuator inputs through the logged inverses and hidden maps. [Held data and verification](forehand-carry.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_forehand_carry \
  --output /tmp/forehand-carry.json.gz --source-output /tmp/forehand-carry-source.npz
```
