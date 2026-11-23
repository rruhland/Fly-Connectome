# What a reward-trained Pong player learns about a five-joint arm while it plays

A simulated SO-101 arm operated a virtual joystick in native Pong. A point on the physical arm model controlled the screen paddle. After twelve source points, the five joint responses it had learned from its own commands and encoders transferred immediately to wrist-camera bouncing. Across two new drive matrices, eight of sixteen stock drops reached ten consecutive rebounds. Leaving the old map in place gave zero. The same motor memory helped on a second, physically altered arm, where performance remained well below the supplied-map comparison. [Runnable source and target experiment](../../scripts/experimental/tt24_learned_tip_transfer.py) · [First portable learner](learned-tip-learner-0.npz) · [Second learner](learned-tip-learner-1.npz)

[Watch one of the native practice points followed by the first wrist release on the second drive](pong-to-wrist.mp4). The two target panels replay the same ball launch.

## What the learner knew

The screen player used a readout previously trained from actual Pong outcomes. It observed the native raster and issued a scalar drive. The supplied joystick adapter measured an off-center point on the physical arm and used its calibrated local geometry to turn the scalar request into simultaneous joint rates. The arm started each point from a different comfortable posture. The joint contributions varied naturally with posture and incoming play. There were no separate probe movements in Pong. Each new evaluator drive matrix combined a random five-dimensional rotation with unequal gains. The learning code received only the physical action requests and the resulting encoder measurements. [Trained screen parent](tt24_joint_pong.md) · [Joystick definition and original controlled comparisons](pong-arm-tip-report.md)

The learner fitted a local response from eight-tick windows, roughly 133 ms each. A robust fit reduced the influence of periods when a joint barely moved. The full sequences supplied 59.08 and 68.02 simulated seconds in the two training courses. Six further points per matrix checked the estimate before wrist evaluation. Their active native player made 10 and 7 returns, versus 3 and 0 with a zero readout. Evaluator-only matrix errors were 13.3% and 5.3%. On the earlier single-axis joystick, half a second was enough to transfer the one exercised direction. [Windowed fit and control study](pong-arm-tip-report.md) · [Single-direction transfer](pong-arm-report.md)

The stock wrist game received the learned matrix at the port for the same drive. It received zero target practice drops or probe commands. Its five-axis feedback readout had already learned bouncing from reward on donor arms. Its wrist observation pipeline and camera registration were already available. The Pong and wrist tasks used separate motor readouts, with learned actuator response shared between them. The loader selected the appropriate game head and the previously experienced control interface.

| Sixteen stock wrist drops, five seconds each | Sum of longest consecutive runs | Drops with ≥10 consecutive |
| --- | ---: | ---: |
| From ordinary reward-trained Pong play | 133 | 8/16 |
| Original drive map | 17 | 0/16 |
| Same source with encoder timing shuffled | 11 | 0/16 |
| True drive map supplied | 167 | 10/16 |

Each of the 64 target trajectories replayed exactly. Both portable files store the two motor readouts and the observed map alongside the original identity map. After wrist evaluation they reproduced their example wrist commands, plus every native Pong outcome and command on the twelve held points. These retention checks use explicit task and drive recall. [Drive A](learned-tip-transfer-0.json.gz) · [Drive B](learned-tip-transfer-1.json.gz) · [Checkpoint replay checks](learned-tip-reload.json.gz)

## Moving the learned drive to a different arm

A second test took the electronics measured in Pong on stock and retained them on an altered body. It had different arm dimensions, joint axis, strength and calibrated camera. Starts had wide random joint angles, an independent ball dispenser, randomized mass and drag, and two unannounced gravity changes during the 15-second horizon. All alternatives saw the same wrist image at the start and received equal preparation. The earlier alternate optical calibration, donor motor readout and fixed workspace adjustment were shared. No ball-bouncing practice or motor inspection occurred between the two environments.

| Twelve altered-arm starts | Sum of longest consecutive runs | Drops with ≥10 consecutive |
| --- | ---: | ---: |
| First four native points, 18.78 or 26.30 s | 28 | 1/12 |
| First twelve native points, 59.08 or 68.02 s | 55 | 2/12 |
| Original drive map | 16 | 0/12 |
| True drive map supplied | 153 | 6/12 |

Across the paired conditions all 48 altered-arm tapes reproduced the physical events. The longer source fit helped in both installations; the stock result substantially overstates reliability when the body, camera, release and physics change together. Even the true interface lost half the drops before ten. [Altered arm on drive A](learned-tip-body-0.json.gz) · [Drive B](learned-tip-body-1.json.gz) · [Runner](../../scripts/experimental/tt24_learned_tip_body.py)

On twelve further altered starts we inspected the two directions least covered in native play before release. At 1.60 seconds per drop the correction improved the measured map on 10/12, but pooled qualifying rebounds fell from 90 to 75. It is not the default. Separately, reusing the first single-axis Pong response and measuring its four-dimensional complement reduced receiving inspection time by 0.154 seconds on stock. The all-in path including the Pong prefix was 0.346 seconds slower than cold identification. [Two-direction follow-up](tip-two-probes.md) · [Four-direction completion and post-Pong switch](pong-dense-setup.md)

A strictly recursive implementation also ran inside the native control loop. It kept 7,608 bytes of numeric state and updated from each requested movement and subsequent encoder reading. It qualified on one of the two drives. On that drive eight new wrist games matched the batch fit at 6.375 mean consecutive rebounds, versus 1.00 stale. [Streaming learner and complete source gate](online-tip-report.md)

Keeping a rolling 0.5 MB of recent numeric commands and encoder readings reproduced both qualified estimates while updating at every eligible Pong point break. It reached ten on 6/16 more fresh stock drops, compared with 1/16 stale. [Download the continuing motor learner](rolling-motor.md)
