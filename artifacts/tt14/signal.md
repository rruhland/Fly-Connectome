# Can table-tennis motor experience help find a hidden source?

A new task asked each simulated arm to seek an invisible source using its tool position and a noisy scalar signal. In 16 trials where the source moved without warning, a learner that detected a change and relearned the reward surface acquired the new source 14 times. A learner that continued updating a single surface acquired it zero times. The detector made zero switches in 16 matched trials where the source stayed still. Whether table-tennis motor experience accelerated the search depended on the motor disturbance and the exploration protocol. [Gantry changes](change-gantry.json.gz) · [Five-axis changes](change-five.json.gz)

## What information was available

MuJoCo simulates the same gantry or five-axis body and actuator wiring used in the donor table-tennis game. The new task renders a single tool marker in two calibrated orthographic projections. The source has no visible marker, even at overlap. The learner sees the images and measures a single scalar field strength at the tool. The evaluator chooses reachable sources, randomly rotated anisotropic fields, offsets and either quadratic or logarithmically compressed profiles. Field strength contains noise and is rounded. No source coordinates, distances, principal axes, profile, gradients, change times or simulator state go to the policy. Delay and actuator noise change during each trajectory. The chosen family consists of smooth fields, each with a single maximum. [Task and learner](../../scripts/experimental/tt14_signal_search.py)

The player tries several reaches, fits a local quadratic reward surface from action-induced visual motion and scalar readings, and asks the existing visual motor servo to approach its estimated peak. During play the servo can adjust its motor model. Only its initial local motor geometry comes from table tennis. The RGB marker reader, exploratory movements, smooth reward prior and field-maximization objective are supplied. The test measures how useful learned motor geometry is to an unfamiliar task and whether a new goal can be learned online. Source seeking from scalar samples has also been studied on mobile robots. [Source seeking by extremum estimation](https://www.sciencedirect.com/science/article/pii/S1474667016356518)

## Finding a stationary source

Each trial ran for 144 physical control frames. A frame counts on target within 4.5 cm. Acquisition requires four consecutive such frames. Trials reset their search history and motor servo to the indicated initial state. The equal-probe controllers used identical commands for the first four frames per motor port. Each cell gives frames on target and sources acquired. The direct condition starts using the transferred map immediately. The flat-signal condition receives zero at every position. Every reported physical trajectory was replayed, including the sequence of unmasked sensor readings.

| Initial arm state and number of targets | Fresh motor fit | Original table-tennis map | Local table-tennis refit | Adapted table-tennis map | Adapted, direct start | Adapted, flat signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gantry, signed rewiring, 16 | 992 / 14 | 839 / 14 | — | 911 / 14 | 987 / 14 | 0 / 0 |
| Five-axis, signed rewiring, 16 | 972 / 14 | 964 / 16 | — | 817 / 12 | 1,112 / 14 | 17 / 1 |
| Gantry, coupled axes, 12 | 810 / 11 | 798 / 10 | 840 / 11 | 839 / 11 | 744 / 10 | 0 / 0 |
| Five-axis, coupled axes, 12 | 487 / 8 | 657 / 11 | 669 / 11 | 680 / 11 | 722 / 10 | 0 / 0 |

The coupled five-axis case showed the clearest transfer from table tennis to search: 11/12 sources acquired with each of the table priors, compared with 8/12 from scratch. The newer correction added 23 frames over the original inherited map, with five individually better paths, one tie and six worse. On gantry the new correction tied the local refit's acquisition count and came within one frame of its total. The two signed-rewiring trials used TT13's first changed-game maps. With equal probing, both adapted maps lost to the fresh servo in total time on target. [Signed gantry](signal-gantry.json.gz) · [Signed five-axis](signal-five.json.gz) · [Coupled gantry](signal-gantry-coupling.json.gz) · [Coupled five-axis](signal-five-coupling.json.gz)

## Changing the reward while the arm is moving

This independent suite used the two TT13 rewired bodies. In each 264-frame trial, the evaluator moved the source once at a hidden random frame between 102 and 134. The second source lay at least 14 cm from the first. Each changed trial had a matched stable twin with exactly the same actions before the switch. The arms stayed live throughout. On three successive unexpected discrepancies between observed and predicted strength, the adaptive learner retained the motor servo and its history and restarted only the reward-surface fit and exploratory reaches. The comparison kept fitting the single surface throughout. A fresh-motor control could also detect changes. [Detector and trial code](../../scripts/experimental/tt14_signal_change.py)

| After relocation | Keep one reward surface | Detect and relearn, table motor map | Detect and relearn, fresh motor fit |
| --- | ---: | ---: | ---: |
| Gantry, acquisitions out of 8 | 0 | 7 | 7 |
| Gantry, frames on target out of 1,123 | 0 | 443 | 461 |
| Five-axis, acquisitions out of 8 | 0 | 7 | 7 |
| Five-axis, frames on target out of 1,170 | 0 | 395 | 361 |

The table-initialized player recognized 14 of 16 relocations on the third post-change reading, then reached four-frame acquisition 55–114 frames after the change in those 14 trials. All 16 stable twins made exactly the same actions under detection and continuous fitting. The evidence supports selective relearning when the reward changes. The fresh-motor control's 14/16 acquisitions show that this capacity can also operate without prior table-tennis training. [Gantry](change-gantry.json.gz) · [Five-axis](change-five.json.gz)

## Did the coupled maps learn real motion?

For each installation, 24 unseen 56-frame pulse sequences were run on both the original and coupled physical harnesses. Motor disturbances were matched. The learner supplied a frozen prediction from its requested commands alone. This evaluator measured the discrepancy between the predicted and actual **change in tool displacement caused by coupling**, in three dimensions. Absolute millimeter distances are averaged across frames, then across pulse sequences. The final column applies the true mixer to the original calibrated model for a diagnostic reference.

| Mean change-prediction error | Fixed or signed-remap fit | Fresh table refit | Learned correction around the old map | Known mixer, evaluator only |
| --- | ---: | ---: | ---: | ---: |
| Gantry | 53.9 mm | 65.2 mm | 27.7 mm | 14.0 mm |
| Five-axis | 78.3 mm | 54.4 mm | 32.7 mm | 15.3 mm |

For absolute displacement under coupling, error fell from 77.5 to 39.0 mm on the gantry and 105.4 to 54.9 mm on five-axis. The end-of-game signed-remap controls had the original local geometry in both cases. The corrected models generalized to fresh motions on both bodies. Game scores and hidden-source acquisition show how much of that improvement the respective controllers used. Both physical harnesses and every pulse were replayed. [Gantry audit](motor-audit-gantry.json.gz) · [Five-axis audit](motor-audit-five.json.gz) · [Evaluator implementation](../../scripts/experimental/tt14_motor_audit.py)

## Reproduction

From the repository root use the [hall environment](../tt6/README.md) with the single-thread numeric libraries and experimental scripts on PYTHONPATH. Commands below reproduce the five-axis trials. The gantry seeds are 49362318, 56193471, 34726184 and 1246831 respectively. The coupling commands read the TT14 changed-game checkpoints from the supplied output directory.

```sh
python -m scripts.experimental.tt14_signal_search five /tmp/signal-five.json.gz --seed 6534179 --count 16
python -m scripts.experimental.tt14_signal_change five /tmp/change-five.json.gz --seed 84976112 --count 8
python -m scripts.experimental.tt14_signal_gain artifacts/tt14 five --seed 68437290 --count 12
python -m scripts.experimental.tt14_motor_audit artifacts/tt14 five --seed 587201 --count 24
```
