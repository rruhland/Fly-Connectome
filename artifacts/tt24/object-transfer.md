# Learning which mover matters from native Pong

The native Pong fit predicted point continuation on its own held games with 91.7% balanced accuracy. In the two-object wrist task, it produced 28 powered rebounds. Shuffled-native outcomes produced 43, cold tracking produced 42, and the correct-track diagnostic produced 80. The same frozen joint readout received every selected track. The source fit sees actual native Pong RGB under already learned control. The target task is wrist-camera bouncing in MuJoCo with a second plausible ball-shaped mover. The second object has the same physical radius and follows a replay of another stock ball's physical trajectory. It is render-only and receives no collisions or task reward. [Registered target design](object-transfer-plan.json) · [Yoked physical paths](object-transfer-yokes.npz)

## Native source

The source connected-component tracker discovered the moving controlled paddle from its visual response to issued commands. It recorded the compact mover and that paddle through 64 separately seeded native Pong episodes. Forty-eight episodes supplied 6,594 camera observations and 465 optical encounter windows, including 303 followed by at least 0.28 seconds of point continuation. Sixteen independent native episodes supplied 2,226 camera observations and 156 held windows. The fit used elapsed point continuation as supervision. It received no contact timing, ball state, true identities or scored spatial targets. Neither source fit received wrist observations. [Per-episode native receipt](object-transfer-source.json.gz) · [Frozen Pong and shuffled models](object-transfer-source.npz)

The fitted predictor classified native continuation with 91.7% held balanced accuracy versus 33.7% with the same examples and shuffled duration outcomes. The native Brier scores were 0.054 and 0.267, respectively. The model scores relative motion reversal, overlap with the controlled surface, and the recent motion of that surface. Spans and normal axes come from the source visual calibration and the target public geometric readout. In target bouncing, the two-dimensional tangent becomes unsigned displacement in the paddle plane.

## Held wrist task

Twelve fresh stock-arm drops use two supplied chromatic ellipse proposals. Each color is the live ball in six drops. Both spheres share one radius, and the yoke path and lateral sign vary independently of the palette. Each yoke is a previously recorded stock game. It is translated relative to the current blade home with a fixed 47 mm lateral shift and a 12 mm depth shift, whose signs vary. The released live ball and the rendered ghost need not be distinguishable from their early motion alone. Every condition receives the same 17-tick zero-command observation prefix and the same frozen public optical state estimator and reward-trained donor motor. Neither the estimator nor motor is refit. [Public calibration](public-body.md) · [Frozen motor provenance](tt24_donor.md)

The cold tracker uses visibility and trajectory continuity. The Pong and shuffled selectors add equally parameterized source-fit evidence at candidate encounters. A hand-weighted geometry control uses the same encounter measurements. Correct-track and one-ball conditions bound the effects of visibility and role assignment. Segmentation proposals and public paddle geometry are supplied to all active conditions. Post-hoc powered hits require a physically scored rebound and a measured contact-point upward speed above 0.055 m/s. Actual motor actions use only the selected wrist track and encoders. [Scorer and target driver](../../scripts/experimental/tt24_object_transfer_trial.py)

| Wrist condition | Scored rebounds | Powered rebounds | Powered in first active second | Longest streak | Early correct route | Five correct ticks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold continuity | 56 | 42 | 18 | 17 | 178/326 (54.6%) | 7/12 |
| Native Pong duration | 43 | 28 | 21 | 5 | 216/327 (66.1%) | 10/12 |
| Shuffled native outcome | 59 | 43 | 26 | 17 | 236/327 (72.2%) | 10/12 |
| Hand-weighted contact | 38 | 25 | 12 | 17 | 89/327 (27.2%) | 4/12 |
| Correct track, two balls | 97 | 80 | 28 | 17 | 328/328 (100.0%) | 11/12 |
| One visible live ball | 150 | 128 | 26 | 18 | 330/330 (100.0%) | 11/12 |

Early route accuracy covers ticks 17–46 when both proposals were visible. The single-ball comparison requires only the live proposal. Five correct ticks means a five-frame correct streak beginning at or after activation. Among successful assignments the median onset was zero ticks after activation for cold, Pong and shuffled, and one tick for the hand-weighted rule. One held drop ended before activation for all conditions.

The Pong selector assigned the role more often than cold early in the drop. Shuffling the duration labels produced more correct early routes and more powered rebounds than the native fit. Correct tracks enabled 80 powered rebounds under the same physical motor, compared with 28 from the Pong selector. Independent long yokes can have visually plausible turns close to the real paddle and can partly occlude the live ball. An encounter predictor fitted to single-mover point continuation therefore need not identify which trajectory responded to this arm. This study left the held models frozen.

All 72 motor tapes reproduced their exact physical referee records with wrist imagery removed on replay. The impact audit reads the simulator only after those commands were fixed. [Full paired game, proposal and powered impact receipt](object-transfer-held.json.gz)
