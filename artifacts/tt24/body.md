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
