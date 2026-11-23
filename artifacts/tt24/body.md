# Learning a changed five-joint motor interface

The controller sees its issued five-dimensional motor command, successive measured joint angles and a goal in encoder coordinates. Each active probe uses a fresh random direction followed by its negative. Linear least squares estimates the effects on all measured joints. An inverse fit then steers the next movement. The controller compares predicted and measured displacement to detect changes, and stores previously identified effects for recall. Simulator wiring, body masses, delay and collision state stay in the evaluator. Home-like goal postures and the encoder channel identities are supplied in this test.

Each of 24 held streams changes the complete motor mixing matrix twice and then restores the original. The first change increases lower-arm and camera masses by 35% and 150%, respectively. The second sets the lower-arm mass to 80% of nominal and the camera mass to four times nominal. Both changed plants add one 30 Hz control tick of delay. Body indices and camera signals remain aligned. Four seconds of initial nominal exploration precede the first goals. The median new calibration costs two seconds of motor probes. These are simulated times.

| During the latter half of each stage | Adaptive median joint RMS error | Frozen median joint RMS error | Adaptive streams below 0.06 rad | Frozen streams below 0.06 rad |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.0066 rad | 0.0066 rad | 24/24 | 24/24 |
| First changed body | 0.0093 rad | 1.4472 rad | 21/24 | 0/24 |
| Second changed body | 0.0127 rad | 1.8223 rad | 21/24 | 0/24 |
| Original restored | 0.0075 rad | 0.1773 rad | 18/24 | 12/24 |

The median first of three consecutive goals with RMS joint error below 0.06 rad arrived 3.2 seconds after each new change and 1.6 seconds after restoration. Unrecovered streams are counted beyond the stage duration in those medians. On unsuccessful streams, the paddle could hit the table or clamp before the controller had identified the change. The remaining gaps motivated an independent recovery test. [Per-goal results](body-reach.json.gz) · [Controller and runner](../../scripts/experimental/tt24_body_trial.py)

## Independent recovery test

The revised controller takes smaller movements, briefly reverses an unexpected motor effect and checks previously learned maps during active calibration. On 32 new streams, 31/32 finished each of the two changed-body stages below 0.06 rad, versus 0/32 frozen. After the original body returned, 31/32 met that criterion versus 21/32 frozen. Median time to the first of three sustained accurate goals was 3.2 seconds after either new change and 1.6 seconds after restoration. All 32 streams recalled a previous map during the restored stage. The original nominal test gave 32/32 under both treatments. The fast recovery is established for measured-joint goals on this arm class. [Independent receipts](body-recovery.json.gz)

## Carrying motor learning from bouncing into table tennis

We learned a five-axis decoder from four altered-body paddle-bounce drops (136 ticks, 2.27 simulated seconds) after five nominal calibration drops (150 ticks, 2.50 seconds). The table-tennis pupil received the prelearned decoder immediately when the evaluator applied the same previously unknown dense motor mixing after the shared opening. In nine held points the arm earned a second legal return on **4/9 with the transferred decoder**, **1/9 with its frozen command mapping**, **4/9 on the original body** and **5/9 with the true inverse**. The post-opening return totals were 4, 1, 7 and 6 respectively. All 36 physical command tapes reproduced the scorer. The trained pupil continued to use fixed stereo, previous demonstrations and a supplied opening. This experiment establishes a transfer of body calibration across games on one changed interface. Longer rallies and unassisted skill transfer require separate evidence. [Physical actions and scores](rally-remap.json.gz) · [Experiment](../../scripts/experimental/tt24_rally_remap.py)

## Different physical arm, optics and contact dynamics

The alternate physical arm has a 15% longer upper arm, a tilted elbow axis, a moved wrist camera with 42 g additional mass and different drive strengths. The changed game also alters paddle contact damping and ball drag. A separate held test adds unknown continuous motor mixing. Twenty short, paired-direction motor probes learn effects on joint encoders and nine regions of the wrist image. The decoder tries to reproduce familiar sensory effects on the new arm. Game play uses a direct wrist policy previously trained with target practice. The contact scorer receives physical impacts, upward travel and apexes.

| Total verified rebounds on 16 releases | First changed interface | Second changed interface |
| --- | ---: | ---: |
| Prior policy on stock arm | 47 | 44 |
| Same policy on the altered arm with familiar wiring | 36 | 38 |
| Uncorrected new wiring | 23 | 17 |
| Encoder-learned decoder | 30 | 40 |
| Wrist and encoder decoder | 34 | 37 |
| Known wiring inverse comparator | 36 | 38 |
| Stationary paddle on the altered arm | 29 | 36 |
| Wrist and encoder decoder with live vision removed from the policy | 26 | 35 |

Changed-body calibration used ten drops / 290 frames and seven drops / 278 frames, respectively, equivalent to 4.8 and 4.6 seconds at 60 Hz. Earlier stock calibration used eight and nine additional drops. This recovers much of the prior policy's limited bounce performance after changing the five-axis interface. The score margin over a stationary altered arm is small, especially on the second cohort. [First run](retinal-mixed.json) · [Independent replication](retinal-replication.json) · [Retinal and encoder calibration](../../scripts/experimental/tt24_retinal_map.py)

## Unannounced changes during wrist-only paddle bouncing

Six independent 30-drop streams began with seven stock drops, moved to fifteen with the alternate arm, new contact dynamics and a different dense actuator mapping, then restored the stock setup for eight drops. All three conditions used the same practiced Pong-informed arm readout and exactly matched releases. The adaptive arm observed its wrist image, measured joints and its own issued commands. It detected each change from an error in predicting measured joint movement. It received neither the switch time nor the identity of the physical arm. The familiar-wiring comparison kept the same altered mechanics, optics and ball physics, with the motor command wiring retained.

| Actual rebounds pooled across six streams | Adaptive | Frozen | Familiar wiring |
| --- | ---: | ---: | ---: |
| Seven stock drops | 138 | 138 | 138 |
| First five altered drops, including practice | 40 | 30 | 76 |
| Next ten altered drops | 158 | 62 | 151 |
| All fifteen altered drops | 198 | 92 | 227 |
| Eight restored stock drops | 156 | 176 | 176 |

The first unexpected mapping triggered calibration at frames 12, 12, 12, 16, 16 and 12. The learner spent 850 total control frames probing across the six streams, a mean of 2.36 simulated seconds. It recognized the original mapping within 12–20 frames in every stream. One unnecessary recalibration cost five additional drops on restoration. Through the final seven restored drops, five of six adaptive streams matched their original-policy comparator's total exactly. Nominal setup used five calibration drops (145 frames) and five reference drops (625 frames). The underlying skill still averages only a few rebounds per drop. [All per-drop results](ongoing-body.json) · [Runner](../../scripts/experimental/tt24_ongoing_game.py)

In a separate cross-check, a wrist-and-encoder decoder acquired in five changed-body probe drops (267 frames) let the pre-practiced Pong-informed policy score 68 rebounds on 24 fresh drops, versus 34 under uncorrected mixing and 69 with familiar wiring. Cold-credit and shuffled-credit policies with the identical decoder scored 55 and 52, the frozen earlier arm policy 67, stationary 57 and vision-ablated 55. Earlier nominal motor calibration used eight drops (271 frames). [Per-policy receipts](carried-physics.json)
