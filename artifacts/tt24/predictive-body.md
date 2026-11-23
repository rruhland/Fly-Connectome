# Predictive control from ball-free arm movement

A predictive controller fitted to 14.4 seconds of issued joint rates and encoder readings cut blade-position tracking error from **17.14 to 8.24 mm** on fresh stock-arm trajectories. Error fell from **20.73 to 10.20 mm** with weaker drive. With a frozen source policy and live wrist RGB, it increased scored rebounds from **9 to 23** across 16 new drops. The zero-command control scored **40** on the same drops. This source-policy checkpoint with better motor tracking still trails passive bouncing under the tested reset distribution. [Tracking receipt](predictive-body-reaching.json.gz) · [Wrist-bounce receipt](predictive-body-bounce.json.gz)

## Motor fit

The fit used the first six 144-frame clips in the public ball-free motion recording. It read only the 60 Hz encoders, already-issued normalized joint rates, timestep and train split. The other two clips were reserved for forecasts. There were zero target-game interactions, reward observations or stroke command labels. The separately acquired public body calibration supplied a blade position and upward-normal estimate. [Motion recording](world-arm-public-play.json) · [Public calibration](public-body.md) · [Fit receipt](predictive-body-fit.json.gz)

Separate ridge regressions forecast changes in the five measured joint angles at 1, 2, 3, 5 and 7 ticks. Inputs are current joint angles, inferred drive target, recent encoder differences, previous rate and candidate future rates. Regularization was selected by leaving one complete fit clip out in each fold. The resulting held forecast errors were 8, 18, 26, 35 and 37 mrad, respectively. Carrying the current encoder angles forward gave 35, 68, 98, 146 and 182 mrad. [Frozen motor model](predictive-body-fit.npz) · [Implementation](../../scripts/experimental/tt24_predictive_body.py)

At execution, the controller linearizes the public blade readout at the latest measured joint angles and solves for three bounded future rate commands. It assumes zero rates thereafter in the longer forecasts. Only the first command is issued, and the least-squares problem is solved again at the next encoder reading. The rate, integrated drive-target and home-offset limits are shared with the kinematic grounding. The predictive controller uses no prescribed stroke, incoming-ball planner, live task reward or hidden simulation state.

## Closed-loop blade reaching

The stock-arm and weaker-drive comparison each used 16 paired new starts. Eight starts requested continuous combinations of sinusoids and eight requested random ramps and holds. Each target was a relative blade position and absolute upward-normal x/y components, sampled independently of the ball-free fitting clips. Both controllers received the same target at each tick and used the same blade readout. The kinematic inverse starts from its last issued posture. The predictive controller closes the loop on the measured posture. Actual joint dynamics ran at every step. Encoder tapes replayed exactly in all 64 runs. Physical blade geometry was read only in the held scorer. [Driver and replay](../../scripts/experimental/tt24_predictive_body_trial.py)

| Held body and controller | Position coordinate RMS | Upward-normal component RMS | Normalized five-axis RMS |
| --- | ---: | ---: | ---: |
| Stock, kinematic | 17.14 mm | 0.0585 | 0.3214 |
| Stock, predictive | 8.24 mm | 0.0447 | 0.1721 |
| Weaker drive, kinematic | 20.73 mm | 0.0697 | 0.3879 |
| Weaker drive, predictive | 10.20 mm | 0.0479 | 0.2043 |

The predictive controller lowered normalized physical tracking error on 13 of 16 stock starts and 14 of 16 weaker-drive starts. The x-position error increased from 3.08 to 6.75 mm on stock and from 3.70 to 6.78 mm with weaker drive. Most position improvement came from the z axis. Scaling for the combined error used 45 mm for each position component and 0.29 for each normal component. Scoring omits the first eight command ticks in every run.

## Frozen source policy on wrist RGB

All 16 paired starts were new and used the stock held-drop regime. In both active conditions, the same body-aligned source policy received live wrist-only RGB through the public ball and blade estimator plus proprioception, and requested the blade target on every frame. Policy and motor fit were frozen. The 48 motor tapes replayed to identical physical rebound judgments, including contacts, rises and timing. [Body-aligned source policy](world-prior-body.md) · [Full judge and command tapes](predictive-body-bounce.json.gz)

| Controller | First rebound | Second consecutive | Third consecutive | Longest run | Total rebounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kinematic | 7/16 | 2/16 | 0/16 | 2 | 9 |
| Predictive | 16/16 | 6/16 | 1/16 | 3 | 23 |
| Zero command | 16/16 | 14/16 | 8/16 | 4 | 40 |

The predictive run with three physical rebounds started at seed `367013`. Better tracking helped the source policy make and repeat contact. The same start scored three rebounds with zero command. This comparison supports closing the motor loop with measured joint dynamics. The zero-command control yielded the strongest scores on these drops.
