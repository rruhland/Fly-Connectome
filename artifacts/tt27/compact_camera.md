# Camera registration during live compact-court points

A frozen TT24 camera-to-joint policy played complete points from fresh far serves on the TT25 solid compact court. The far arm replanned its replies around the ball actually returned. The test used the changed-rebound configuration, fresh serve placements and physical rotations of both cameras. The near policy received stereo RGB, measured joints and its own prior commands. All arms acted from the first control tick. [Compact court and opponent](compact.md) · [Experiment](../../scripts/experimental/tt27_compact_camera.py)

Registration started from a nominal image of the compact court. The carried branch acquired patch reliability from three earlier calibration-only stereo views of that scenery. At each new camera setting it matched eight patches per eye in one first stereo frame, then reused the fitted rotation throughout that block. Reset matched eight patches from the same frame using only nominal patch texture. The other branches consumed either the rotated views without correction or intact views. Every branch used the same frozen motor fit. [Registration method](world-camera.md) · [Calibration inputs and held angles](compact_camera-assessment.json.gz)

## Two fresh camera settings

| Full-point outcome across six releases | Carried, detect then project | Reset, detect then project | Uncorrected | Intact |
| --- | ---: | ---: | ---: | ---: |
| Legal near returns | 7 | 9 | 0 | 3 |
| Legal far replies | 7 | 9 | 0 | 3 |
| Starts reaching a second near return | 2 | 2 | 0 | 0 |
| Longest alternating exchange | 8 | 14 | 0 | 2 |
| Abrupt commanded reversals per second | 8.27 | 7.75 | 27.28 | 7.50 |

Both fits recovered multi-return exchanges from views that gave zero legal returns without correction. On one held launch the carried branch scored 3+3 and reset scored 2+2. On another, carried scored 4+4 and reset scored 7+7. Intact views scored 1+1 on each launch. Small changes to the visual input can substantially alter the subsequent physical exchange. The video shows 5.73 seconds of live play from the carried branch at 30 frames per second, followed by a short hold. All 24 physical points reproduced their motor-tape durations and full referee events. [Watch the 4+4 point](compact_camera-live.mp4) · [Video verification](compact_camera-live.json) · [Paired full-point tapes](compact_camera-assessment.json.gz)

## Detecting before or after projection

On 74 stereo frame pairs from the earlier 7+7 compact point, we rendered the same physical motion with nominal and rotated cameras. The replayed referee events matched the original point. The table compares ball centers against the intact-camera detector at 240×180 pixel resolution, pooling both eyes.

| Processing of rotated frames | Median error | 90th-percentile error | Missing detections |
| --- | ---: | ---: | ---: |
| Uncorrected | 22.56 px | 23.74 px | 0 |
| Carried registration, detect then project | 0.30 px | 0.55 px | 0 |
| Carried registration, resample images then detect | 0.47 px | 4.01 px | 2 |
| Exact rotation, detect then project | 0.23 px | 0.45 px | 0 |
| Exact rotation, resample images then detect | 0.37 px | 3.73 px | 1 |

Projecting detections preserves the detector’s original pixels. Resampling changes edge colors and occasionally disrupts detection even when the evaluator supplies the exact geometric transform. The true angles were used only to measure registration error and in this replay diagnostic. [Paired detector comparison](compact_camera-tracking.json.gz)

## Two further camera settings

| Full-point outcome on four additional releases | Carried, detect then project | Reset, detect then project | Carried, resample first | Reset, resample first | Uncorrected | Intact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legal near returns | 2 | 0 | 0 | 1 | 0 | 2 |
| Legal far replies | 2 | 0 | 0 | 1 | 0 | 2 |
| Longest alternating exchange | 4 | 0 | 0 | 2 | 0 | 4 |

The carried fit made two legal returns in one point when it projected detections directly. Warping its RGB before detection gave zero returns on that same point. Across the two courses, detecting before projection yielded nine near returns with carried patch reliability and nine with equal-budget reset. Uncorrected gave zero across ten releases. The carried fit found six to eight mutually consistent patches per eye across the four mount changes. Reset found four to six. These results establish that registration can restore sustained play with the frozen motor policy. The small courses give no consistent return-count advantage for retaining patch reliability over equally budgeted recalibration. All 48 live point tapes verified. The ten launches are disjoint across the two courses, earlier compact assessments and the motor training banks. [Follow-up tapes and held angular errors](compact_camera-tracking-held.json.gz)

## Reproduce

```sh
export PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt27_compact_camera
python -m scripts.experimental.tt27_compact_camera --inspect-tracking
python -m scripts.experimental.tt27_compact_camera --changes 2 --count 2 --turn-offset 2 \
  --seed 1196528008 --tracking paired --output artifacts/tt27/compact_camera-tracking-held.json.gz
python -m scripts.experimental.tt27_compact_camera --film
```
