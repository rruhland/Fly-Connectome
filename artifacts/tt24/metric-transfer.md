# Metric stereo recalibration on the changed arm

The original stock action readout remained fixed. The target combined the alternate arm's longer link, tilted hinge, relocated wrist camera and payload with the changed-drive and ball regime. Four target drops collected 62, 80, 69, and 37 decision frames, a total of **248 frames or 4.13 seconds**. The frozen wrist pupil drove each drop with small zero-mean exploratory joint commands. The new calibration used its wrist RGB and encoder stream alongside two calibrated fixed RGB views. The source pupil and its original calibration came from earlier privileged training. [Implementation](../../scripts/experimental/tt24_metric_transfer.py)

Fixed-view triangulation retained 243 paired detections of the ball and had a 0.286 px reprojection RMS. The wrist recalibration had 231 accepted ball correspondences. The fitted weighted wrist-to-stereo position residual fell from 32 to 13 mm on those calibration frames. The calibration adjusts the source wrist transform and paddle-translation estimate with a prior around the stored source map. It also fits a smaller residual correction to the causal velocity readout from stereo-derived motion. Six separate development drops selected the full position-and-motion variant. At evaluation the pupil received wrist RGB and joint angles.

## Paired fresh results

The rows used the same 16 alternate-arm, changed-drive seeds, 256201–256216. Scores are consecutive referee-verified rebounds.

| Calibration | Target exposure | Mean | Median | Maximum | At least 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen stock pupil | 0 frames | 4.44 | 3 | 15 | 2/16 |
| Public metric stereo residual | 248 ordinary frames | 6.38 | 5 | 13 | 3/16 |
| Matched privileged position labels | Same 248 ordinary frames | 4.63 | 4 | 12 | 1/16 |
| Earlier privileged residual | 943 teacher-driven frames | 5.38 | 4 | 16 | 3/16 |

The public residual improved 11 paired drops relative to the frozen pupil, tied two, and lost three. The exposure-matched comparison replayed the ordinary decisions with identical encoders, then replaced the stereo positions with privileged ball and blade positions on the same supported frames. It kept the residual architecture and frozen head. The higher public mean describes this seed cohort. The earlier teacher-driven comparison reached a longer best run of 16. [Held metrics and exact replay commands](metric-transfer.json.gz)

The exported correction requires the original stock checkpoint checksum. Its live wrist-only replay reproduced all 300 decisions and 13 consecutive rebounds on held seed 256204. Run `python -m scripts.experimental.tt24_metric_transfer` to regenerate the comparison and checkpoint. [Public correction](metric-transfer-map.npz)

## Longer horizon check

With a ten-second cutoff, the same exported pupil reproduced the first 300 actions of seed 256204 exactly, then continued to 17 consecutive verified rebounds before the run ended at 6.83 seconds. This seed was selected because it reached the original five-second cutoff. Four new starts with the longer cutoff scored 2, 7, 7, and 1. Run the same module with `--extend-horizon` to reproduce these checks. [Longer horizon trials](metric-transfer-long.json.gz)
