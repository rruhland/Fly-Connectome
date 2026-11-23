# Receiving after the cameras turn

One stereo frame recovered 58 legal returns across 128 fresh nearby feeds and four unseen rotations of the fixed cameras. Leaving camera coordinates at the factory setting landed 39. Keeping the last practice alignment landed 43. The intact camera rig landed 58. Each row below uses the same 128 launches, full-size solid SO-101 and physical referee. [Paired outcomes and replay tapes](world-camera.json.gz)

| Camera strategy | Legal returns | Paddle contacts | Different stroke choice from intact |
| --- | ---: | ---: | ---: |
| Factory alignment | 39 | 89 | 47 |
| Frozen last practice alignment | 43 | 93 | 44 |
| Recalibrate with reset patch reliability | 58 | 112 | 1 |
| Recalibrate with carried patch reliability | 58 | 112 | 1 |
| Intact camera rig | 58 | 112 | 0 |

The calibrated player gained 23 legal returns and lost four versus factory alignment. It gained 19 and lost four versus frozen alignment. Five and eight patch matches per camera gave the same legal counts. At the eight-patch budget, practice increased the number of agreeing matches across the eight camera views from 27 to 42 and lowered the median orientation error from 0.67° to 0.21°. The reset and carried players tied on realized returns in this cohort. [Rotation fits and paired changes](world-camera.json.gz)

Recalibrating from the first frame of every feed reproduced all 128 outcomes without a camera-change signal. [Per-feed calibration and replay tapes](world-camera-each-feed.json.gz)

Twelve larger randomized camera settings used a new feed seed and 96 nearby trials. At eight patches per camera, carried reliability gained eight legal returns and lost one versus reset reliability. The net gains came from two settings, with a loss in one and ties in nine. At five patches, carried and reset each landed 35 legal returns. [Larger rotations and paired outcomes](world-camera-stress.json.gz)

| Camera strategy on larger rotations | Legal returns | Paddle contacts | Different stroke choice from intact |
| --- | ---: | ---: | ---: |
| Factory alignment | 22 | 47 | 78 |
| Frozen last practice alignment | 22 | 47 | 80 |
| Eight-patch reset | 40 | 70 | 24 |
| Eight-patch carried | 47 | 77 | 9 |
| Intact camera rig | 50 | 81 | 0 |

On the larger rotations, the eight-patch carried player found 119 agreeing patches versus 87 with reset reliability. The median error fell from 0.35° to 0.25°, and fits failed on one of 24 camera views versus three with reset reliability. [Camera fit receipts](world-camera-stress.json.gz)

On 136 unmodified held recordings divided across the four moderate settings, eight-patch carried calibration landed five legal returns. Factory, reset and intact cameras each landed four, and frozen alignment landed two. The carried player gained one paired return and lost none versus factory or eight-patch reset. Its 11 stroke disagreements with the intact camera compare with 59 for factory alignment and 68 for frozen alignment. Legal returns remained rare in this broader cohort. [Recorded-feed outcomes](world-camera-recorded.json.gz) · [Held roster](../tt26/fresh-recordings.json.gz)

The renderer rotates the cameras about their optical centers while the court and arm remain physically fixed. The player stores a nominal stereo view of the room and learns a reliability count for each textured patch from three earlier camera changes. Agreement with the RGB-only rotation fit supplies the update. For a new orientation it selects a spatially distributed set of patches, matches them in the first stereo frame, and fits a rotation between matched rays. It then maps detected ball centers into the nominal image coordinates used by the existing 19-posture receiving policy. The reset control ranks patches by image texture and runs the same fit with the same budget. The camera angles are used only to render and score. [Implementation](../../scripts/experimental/tt27_world_camera.py) · [Receiving checkpoint](../tt26/match-broader.npz)

All policies get up to four frames before the nominal stroke decision. In the block trials the adapted policies use frame zero of the first feed for the new rotation fit. The receiver selects a posture before it sees any return reward. The arena physically plays each distinct selected command sequence and replays it against the referee. The nearby cohorts use new perturbations of previously stored measured feeds. The source camera practice uses stationary court images and supplies no receiving labels. [Launches and verification](world-camera.json.gz)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_world_camera
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_world_camera --each-feed --output artifacts/tt27/world-camera-each-feed.json.gz
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_world_camera --stress --per-change 8 --output artifacts/tt27/world-camera-stress.json.gz
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_world_camera --per-change 34 --feeds /path/to/rallies.json --output artifacts/tt27/world-camera-recorded.json.gz
```

## Delayed frame delivery

A separate 48-feed test held back the actual stereo frames by one or two control ticks. Every controller still chose its posture on tick three. The timestamp-aware players reconstructed unreceived positions from the most recent two delivered sightings. The unit model extrapolated at the measured velocity. The carried model scaled that velocity by 0.648, a gain learned from earlier recorded play. [Delayed trials and 63 physical replays](world-latency.json.gz) · [Earlier gain fit](../tt25/recorded-delay-visual-fit.json)

| Delay | Prompt RGB | Stale RGB | Timestamp-aware unit gain | Timestamp-aware carried gain |
| --- | ---: | ---: | ---: | ---: |
| One tick, 24 feeds | 10 | 9 | 10 | 10 |
| Two ticks, 24 feeds | 14 | 10 | 13 | 10 |

The unit model gained three legal returns and lost none versus stale RGB at two ticks. The carried gain gained two and lost two. This course supports extrapolating timestamped observations and gives no evidence for using the earlier time gain at this receiving decision. [Paired results](world-latency.json.gz) · [Implementation](../../scripts/experimental/tt27_world_latency.py)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_world_latency
```
