# TT20 paddle and gripper tag replay

A native RGB factorial independently toggles the twelve temporary court survey posts and the arm markings (four colored paddle-rim dots and one gripper-tip dot). All four scenes use the same warm-orange ball, red paddle rubber, natural court paint, clamped SO-101, camera lens settings, lighting and physics. The marker geometry is noncolliding, and the arm dots have zero mass. Each row sees exactly one 480 × 360 RGB stream, either the moving wrist camera or the fixed left camera.

Four completed strict TT20 point tapes come from the first two occurrences of each server role in the archived left-eye match. These are frozen command replays. They preserve the original official outcomes and all referee events. Each point’s complete position and velocity sequence hashes identically across the four appearances. The full assay comprises 16 point replays, with 183 frames per appearance and per camera. [Archived source motions](camera-carry-left-game-40582930-0.json.gz)

## Measurement

The natural RGB decoder fits each camera from table edges, painted lines and the net. It uses the apparent diameter of the orange ball and red paddle silhouette with their known respective radii of 20 mm and 76 mm for metric depth. It abstains on a unique blade-plane normal in one image because two planes can explain the same ellipse. The established four-rim-dot pose solver supplies a metric center and plane normal when the four colors are detected and pass reprojection and court bounds. We apply the same transparent, fixed RGB detector to natural paint, including the no-tag negative conditions. The gripper dot changes the appearance of the scene and is recorded as a candidate detection, but is unused by the paddle estimator. A separate survey-post fit quantifies any calibration opportunity from the posts. The evaluator checks metric center in millimetres and the unsigned plane-normal deviation in degrees. MuJoCo coordinates and normals enter only those scoring calculations.

## Results with the tags installed

Each cell gives coverage out of 183 frames, followed by median/p90 error. Blade and ball centers are in millimetres. Plane errors are in degrees. All rows in this table use the natural court registration, whether the posts are present or absent.

| Survey posts | Arm tags | Single camera | Blade silhouette center | Rim-dot blade center | Rim-dot plane normal | Ball center |
|---|---|---|---:|---:|---:|---:|
| Off | Off | Wrist | 119 · 10.8/15.3 | 0 · — | 0 · — | 58 · 15.8/62.1 |
| Off | Off | Outside | 182 · 16.2/78.9 | 0 · — | 0 · — | 152 · 36.7/157.6 |
| Off | On | Wrist | 116 · 10.6/14.5 | 116 · 8.9/11.6 | 116 · 1.7/2.0 | 0 · — |
| Off | On | Outside | 182 · 7.1/81.8 | 173 · 21.5/236.0 | 173 · 3.4/48.7 | 4 · 30.9/38.7 |
| On | Off | Wrist | 108 · 10.8/12.8 | 0 · — | 0 · — | 57 · 16.8/62.8 |
| On | Off | Outside | 0 · — | 0 · — | 0 · — | 151 · 35.5/157.8 |
| On | On | Wrist | 108 · 10.6/12.9 | 112 · 8.5/10.9 | 112 · 1.7/1.9 | 0 · — |
| On | On | Outside | 0 · — | 173 · 21.5/236.0 | 173 · 3.4/48.7 | 4 · 30.9/38.7 |

With no survey posts, the wrist obtained 116 dot-based blade normals at 1.7° median error and 2.0° at the 90th percentile. On those same 116 frames, the dots improved center error from 10.6 to 8.9 mm median. The outside camera obtained 173 normals at 3.4° median and 48.7° at the 90th percentile. On the same outside frames, the dot solver worsened center error from 6.8 to 21.5 mm median and from 72.6 to 236.0 mm at the 90th percentile. Its generic silhouette-angle guess had 5.4° median and 15.1° at the 90th percentile on the paired frames. The natural production tracker leaves that angle unset.

The existing tags disrupt ball ranging when the natural ball detector is held fixed. With the posts removed, enabling arm tags reduces wrist ball coverage from 58 to zero frames and outside ball coverage from 152 to four. The outside red silhouette is also lost entirely when the survey props are installed, despite all 183 natural camera fits succeeding. Opening-frame diagnostics identify role-color collisions. At the first own-serve opening, the outside orange detector is 0.2 px from the actual ball without tags. With tags, its candidate lies 0.7 px from the projected orange rim dot and 29.0 px from the ball. Adding survey posts moves the outside red detector from the paddle to within 0.1 px of survey marker 11. These observations are specific to the existing marker colors and unchanged role detectors.

The independent post-based camera fit registered 183/183 outside frames and 43/183 wrist frames with arm tags removed. With arm tags installed, it registered 183/183 outside and 3/183 wrist frames. For the posts-plus-tags outside branch, this alternate camera calibration gave a rim normal on 169 frames at 3.1° median and 49.3° at the 90th percentile.

## After removing the tags

The first own-serve tape (1066140738) and first rival-serve tape (1482841038) are the RGB teacher. With arm tags and no posts, compare the rim-derived and silhouette-derived distances from the optically fitted camera, clip training support to ratios between 0.85 and 1.15, and fit one median multiplicative scale. Also average the rim-derived world-facing plane normals and use that learned reference to choose between the two ellipse solutions. Both estimates use the same marked RGB image and the same optically registered camera. The fit requires at least five supporting frames spanning both source tapes. The test uses only stripped RGB from the second own- and rival-serve tapes (632487885 and 1556439130), with camera calibration initialized fresh per point. Compare center error against the unchanged silhouette and angular error against choosing the ellipse solution closest to the direction down the court. This generic normal-choice rule is an additional diagnostic, since the operational monocular natural tracker abstains on angle.

| Single camera | Tagged teacher frames | Held silhouette frames | Fresh → carried median center error (mm) | Generic → carried median angle error (°) |
|---|---:|---:|---:|---:|
| Wrist | 58 | 61 | 10.95 → 14.59 | 4.33 → 4.33 |
| Outside | 83 | 94 | 17.13 → 29.28 | 5.52 → 5.52 |

The carried calibration worsened held center estimates in both cameras. It selected the same ellipse branch as the generic rule on every held frame. These tapes give no evidence that this RGB-to-geometry calibration improves blade measurement after stripping the tags.

This is a measurement assay with four correlated motion sequences. The experiment tests optical coverage and accuracy on frozen actions. Gameplay improvement, robustness to differently placed dots and broader transfer across body layouts remain unmeasured.

[Per-frame receipt](arm-tag-factorial.json.gz) · [Reproduction script](../../scripts/experimental/tt20_arm_tags_trial.py)
