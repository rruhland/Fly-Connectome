# Continuous tracking from an anonymous opening

The tracker inferred the moving target's appearance from the first eight images in each fixed camera, then followed it without a supplied hue. It observed the target within 15 mm on 269 of 282 in-volume frames after the opening of a 14-contact archived rally. The supplied-color tracker reached 262 of 282.

## Sensor

The known table lines and lens intrinsics register both fixed cameras. Temporal foreground in the opening selects a common moving track across the two cameras. A stereo-first fallback links foreground across an early change of direction, so the usual clearing motion can continue during the opening. The initialization excludes the learner's near-arm workspace. Motion-foreground pixels along the selected trajectory teach each camera the target’s normalized RGB appearance and intensity range. Compact regions with that learned appearance are paired through stereo reprojection, then ranked with the recent measured path. A fresh point starts with an empty appearance model.

The ordinary TT22 player receives only the resulting position or a missing observation and its local clock. It keeps the existing calibrated body and physical shot planner. The evaluator reads simulation position separately for localization error. In live comparisons, the first eight learner and opponent controls and the RGB bytes must agree across the two branches. Neither observer can change an action during that common prefix.

## Archived playback

The 303-frame record from seed 593435140 replayed the original RGB digest, strict match events and duration. It included seven legal returns by each arm. The standard clearing controls were unchanged. Errors use evaluation-only positions inside the tracker’s registered playing volume.

| Sensor | All valid frames within 15 mm | After opening within 15 mm | Median measured error after opening |
|---|---:|---:|---:|
| Anonymous | 274/290 | 269/282 | 6.34 mm |
| Supplied-color | 270/290 | 262/282 | 6.41 mm |

The anonymous prefix's first accepted historical point was 41 mm from the evaluator’s center at frame two, before the end-of-prefix estimate settled to 9.7 mm at frame seven. The opening buffer is reconstructed once all eight frames are available and the first shared controls have already been issued. This playback checks perception on the original trajectory. Its legal returns belong to the archived command tape.

## Fresh points

The two sensor branches played four matched starts against the same cooperative anchor rival. They used the standard material in two games and a seeded random hue in two. The anonymous tracker learned its appearance anew from the common opening. Every pair shared both arm command streams and RGB frames for eight controls. The first learner command difference occurred at control 19 in each pair. An independent replay reproduced the RGB digest, point duration, strict referee events and final score for all eight branches.

| Seed | Ball material | Anonymous legal returns | Supplied-color legal returns | Longest anonymous exchange | Longest supplied-color exchange |
|---|---|---:|---:|---:|---:|
| 783616777 | Standard | 4 | 1 | 8 | 2 |
| 1208696531 | Random purple | 3 | 0 | 6 | 0 |
| 1480718242 | Standard | 0 | 0 | 0 | 0 |
| 354768949 | Random green | 1 | 0 | 2 | 0 |

The anonymous observer supported eight strict legal learner returns across the four points. The supplied-color observer supported one. Both lost every point. The changed-hue supplied-color branch generated zero in-volume stereo observations after the prefix. The anonymous branch localized within 15 mm on 131 of its 132 purple observations and 64 of its 65 green observations. These counts describe sensor accuracy on each branch’s own trajectory. The shot planner, training setup and rival remained the same within every comparison.

Reproduce with `python -m scripts.experimental.tt22_motion_object_trial archive` and `python -m scripts.experimental.tt22_motion_object_trial live --points 4` in the native MuJoCo EGL environment. The [archive receipt](motion-object-archive.json.gz) contains per-frame errors and the archived rival commands. The [fresh point receipt](motion-object-live.json.gz) contains both live arm tapes, optical inputs, planned strokes and strict match events.
