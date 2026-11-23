# Estimating the tool angle from the wrist image

A correction measured from the visible, unmarked paddle improved short play on the altered arm. Across twelve fresh starts, ten reached ten consecutive bounces, versus eight using the earlier observer. Seven lasted the whole 15 seconds, versus four. On an independent two-minute course, the corrected arm completed a full release with 407 qualified rebounds through 20 unannounced gravity changes. It lost time on three other starts. A second course that also changed motor wiring favored the original observer. The calibrated variant is available separately. [Load the optional policy](wrist-rim-policy.npz) · [Continual variant with a learned interface bank](wrist-rim-continual.npz)

The estimator measures the visible boundary of the colored paddle in wrist RGB. It fits a circular tool face using previously calibrated camera intrinsics, tool center, radius and thickness. It requires reliable boundary support in all 64 image directions, a consistent repeat fit, and a solution inside the angle bounds. No target trajectories, game rewards, simulator labels or added markings enter the fit. It uses three frames from independent unscored starts, with three more to repeat the measurement. The tool and camera share the same mounting, so identical estimates across starts do not establish accuracy. The receiving observer already had a mounted camera and link calibration. Only the face-angle inputs to the frozen five-joint motor readout change. [Calibration and outputs](wrist-rim.json.gz) · [Image fit](../../scripts/experimental/tt24_wrist_rim.py)

The altered view supplied all 64 directions. The independent physical check placed the inferred camera-frame normal 2.90° from the true direction, improved from 5.50°. The cropped stock view supplied 33/64 directions and the fitted angle reached the search bound. It was rejected. Its raw fit reduced reprojection error but increased angle error from 1.89° to 15.10°. The held physical metrics were calculated after the image fits and never enter the controller.

| Altered arm on 12 paired drops | Qualified rebounds | Starts with ≥10 consecutive | Lasted ≥5 seconds | Lasted 15 seconds |
| --- | ---: | ---: | ---: | ---: |
| Original geometry | 267 | 8 | 6 | 4 |
| Wrist measurement | 432 | 10 | 10 | 7 |
| Same correction in the reverse direction | 378 | 9 | 9 | 6 |

Mass, drag, initial joints and launcher varied independently. The camera supplied every live visual input and gravity changed during play. Four control starts on stock reproduced identical motor actions in all conditions, with streaks of 52, 54, 39 and 44. All 48 action tapes reproduced their full physical outcomes. [Qualification](wrist-rim-course.json.gz)

## Two-minute confirmation and simultaneous motor adaptation

| Six new altered starts | Qualified rebounds, pooled | Longest uninterrupted run on each start | Seconds in play on each start |
| --- | ---: | --- | --- |
| Original | 719 | 1, 61, 11, 66, 60, 32 | 0.48, 75.88, 4.80, 48.18, 78.60, 11.03 |
| Wrist measurement | 547 | 1, 34, 9, 60, 15, 42 | 0.48, 19.80, 4.10, 120.00, 7.62, 16.05 |

The first full two-minute run on the altered arm reached 60 consecutive qualifying rebounds. Some subsequent contacts kept the ball in play with too little rise to extend the uninterrupted streak. The correction improved two release durations, tied one and shortened three. All 12 two-minute condition tapes reproduced every active gravity transition and full referee record. [Watch the two minutes](wrist-rim-407.mp4) · [Full 120-second record](wrist-rim-long-3.json.gz) · [Paired comparator with a 71-second survival loss](wrist-rim-long-4.json.gz) · [Long runner](../../scripts/experimental/tt24_wrist_rim_long.py)

The saved checkpoint reproduced both a stock and a changed-arm live game from the short evaluation, including every motor command. With the same-release encoder inspection enabled, it then played an existing 12-release course with dense control changes, carrying and reloading the checkpoint between games. It scored 113 total rebounds versus 160 for the earlier observer. Stock scores matched across all five releases. The original observer remains the default for continually changing hardware. [All 12 compound games](wrist-rim-continual.json.gz) · [Portable checkpoint build and audit](../../scripts/experimental/tt24_wrist_rim_bundle.py) · [Results from public impact motion](wrist-impulse.md) · [Stereo calibration comparison](acquisition-selfmodel.md)
