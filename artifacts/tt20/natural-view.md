# Metric tracking from the unmarked court

The outside camera recovered a metric ball track on 154 of 172 moving frames, including 90 of 108 frames after relocation. The moving wrist used 60 of 172 frames with the optional continuity check. The check improved wrist calibration accuracy and preserved all 12 held receiving action tapes. [Native calibration audit](natural-view.json.gz) · [Matched receiving replay](natural-wrist-continuity-play.json.gz)

The observer uses normal painted sidelines, end lines, the center seam and the net tape. Colored calibration stands, blade rim dots and the gripper tip marker are physically absent. The fit uses declared lens intrinsics, the public 0.848 × 1.60 m tabletop and the station’s learner-side orientation. White line segments propose a pose, and projected court and net lines refine and verify it against the current image. A red or yellow foreground can occlude line samples. Camera poses from MuJoCo enter the evaluator after the tracking decision. [Fixture](fixture.md) · [Court registration](../../scripts/experimental/tt20_natural_geometry.py)

For one camera, the known 40 mm orange ball diameter and its sampled RGB silhouette give metric range along the current ray. A causal fit can improve depth and velocity from successive bearings. The blade center uses the apparent extent of the known circular paddle. One elliptical silhouette leaves two possible blade normals, so the single-view path withholds pitch. In multiple views, consistent ellipse orientations can also give a blade normal. These methods depend on the red rubber and orange ball remaining distinguishable from the hall. [Observer](../../scripts/experimental/tt20_natural.py) · [One-camera receiving](mono-flight.md)

The audit recorded the same native 480 × 360 control frames for independent wrist and outside estimators. The outside lens physically changes pose while its focal length stays fixed. The move occurs on frame 16, when the scheduled table verification runs. Two courses also hand the monocular observer from wrist to outside, and two hand it from outside to wrist. Both sensors use the same public table frame. A distinct estimator receives one camera per frame through the handoffs. [Paired native traces](natural-view.json.gz)

| Single view | Camera fits | Median / p95 eye error | Ball estimates | Median / p95 ball error |
| --- | ---: | ---: | ---: | ---: |
| Outside before move | 64/64 | 10.2 / 10.4 mm | 64/64 | 23.4 / 120.8 mm |
| Outside after move | 108/108 | 7.1 / 40.7 mm | 90/108 | 21.1 / 72.3 mm |
| Wrist | 116/172 | 4.4 / 433.6 mm | 57/172 | 19.1 / 110.3 mm |
| Wrist with continuity check | 117/172 | 4.5 / 74.0 mm | 60/172 | 17.9 / 109.6 mm |
| Outside to wrist | 21/32 | 4.1 / 402.5 mm | 14/32 | 6.9 / 13.8 mm |
| Wrist to outside | 32/32 | 25.6 / 40.7 mm | 29/32 | 25.4 / 120.6 mm |

Some wrist images exposed too few independent lines to distinguish the court pose after occlusion. The optional check compares a candidate with the last RGB fit within the same point and limits changes in translation and rotation over the elapsed control frames. Fits within both 50 mm of the true camera and 6 pixels on representative court projections rose from 101 to 107. The p95 court reprojection error fell from 348.6 to 19.4 pixels, and p95 blade center error fell from 364.9 to 73.8 mm. Some wrist fits still have substantial error. The handoff tests use the default observer. [Continuity and handoff evaluation](../../scripts/experimental/tt20_natural_trial.py)

The receiving check uses the same frozen controller and wrist-only size ranging on 12 held rival-service starts. Selecting the continuity option preserved all 12 action tapes and referee outcomes. Both conditions contacted the ball on all 12 starts, returned legally on eight, and won five. All 24 tapes replayed. The option remains explicit as `continuous_wrist=True`. Callers with variable capture intervals can pass the actual control tick to the tracker. [Receiving check](natural-wrist-continuity-play.json.gz) · [Reproduction](../../scripts/experimental/tt20_natural_play_trial.py)

```bash
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt20_natural_trial
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt20_natural_play_trial
```
