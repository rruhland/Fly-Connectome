# RGB registration across camera changes

The learner receives synchronized native MuJoCo RGB frames keyed by opaque stream identifiers and a surveyed map of 12 hue-coded markers. The markers stand at different heights around the table and are fixed to the hall floor. Marker positions and object colors are supplied. Each camera fit estimates rotation, position, and focal length with a centered pinhole and square pixels. The same frozen fit procedure registers a new view, checks visible marks in subsequent frames, and withholds a view when fewer than seven marks are visible or they fit poorly. The 3D estimator triangulates detected orange ball and magenta blade pixels and requires support from at least two views. Camera geometry, ball coordinates, and blade coordinates from MuJoCo are used by the evaluator. Three of the six confirmation courses used a 1.632 × 2.576 m table and three used the regulation 1.525 × 2.740 m table. They shared the hall survey.

## Fixed cameras and changes in coverage

Six held-out pairs of mount and motion seeds each produced 32 frames at 256 × 192. At the scheduled switches, two camera positions, orientations, and focal lengths changed, an oblique camera was added, two feeds were removed in succession, and a camera returned at a new pose. All visible streams registered on the first capture of their respective conditions. Each course used six camera solves and zero extra motor commands or calibration-only exposures. The frozen baseline kept the source RGB fits for the original stream identifiers. The true-optics comparison used the same object detections.

| Phase | Frames | Registered ball depth, median error | Frozen ball depth, median error | Registered blade depth, median error | Frozen blade depth, median error |
|---|---:|---:|---:|---:|---:|
| Source | 30 | 29, 28.80 mm | 29, 28.80 mm | 30, 6.88 mm | 30, 6.88 mm |
| Both cameras moved | 30 | 29, 5.39 mm | 0 | 30, 6.68 mm | 10, 608.30 mm |
| Third camera added | 30 | 30, 3.84 mm | 13, 250.78 mm | 30, 7.07 mm | 6, 560.40 mm |
| One of three removed | 30 | 30, 4.38 mm | 0 | 27, 8.00 mm | 0 |
| One view remains | 24 | 0 | 0 | 0 | 0 |
| Stereo restored | 48 | 47, 4.12 mm | 0 | 46, 6.12 mm | 0 |

After both cameras moved, the fitted cameras projected the true ball and blade centers within medians of 0.046 and 0.135 pixels of the renderer cameras. The frozen figures were 9.29 and 18.97 pixels. World errors use frames with accepted depth, and the counts show where a method abstained. For example, the true-camera comparison obtained 5.17 mm median ball error after relocation and 28.32 mm at source, where the narrower baseline and the far opening ball made depth more sensitive to the pixel measurement. In the monocular interval, two points 350 mm apart along one observed ball ray had the same reprojected center. The learner returns no free-space depth from one RGB observation. Size, motion and contact constraints could provide additional evidence in a different estimator. [Frame receipt](view-registration.json.gz)

## Camera attached to the altered arm

The compact and extended plants each ran three separate 28-frame rotary-arm trajectories with the payload camera and one fixed side camera. The compact wrist mount moved up to 129 mm and turned up to 12.44°. The extended proximal mount moved up to 65 mm and turned up to 17.36°. The RGB registrations followed those motions with median camera-center errors of 6.85 and 7.07 mm. The wrist view had no sighting of its own blade, so estimating blade depth from this pair yielded no observations. Six proximal-camera frames had insufficient or rejected mark evidence, and the estimator withheld that view. The arm commands produced the test motion. Registration requested zero commands.

| Plant | Mounted poses registered | Registered ball depth, median error | Frozen ball depth, median error | Registered blade depth, median error | Frozen blade depth, median error |
|---|---:|---:|---:|---:|---:|
| Compact with wrist camera | 84/84 | 74, 4.88 mm | 31, 122.98 mm | 0 | 0 |
| Extended with proximal camera | 78/84 | 71, 5.35 mm | 21, 49.58 mm | 78, 9.90 mm | 21, 18.00 mm |

On the wrist view, the median calibration-only ball reprojection error was 0.111 pixels with current fits and 13.55 pixels with frozen fits. On the proximal view, it was 0.072 and 15.26 pixels, respectively. The true-optics comparison returned 74 wrist-pair ball estimates at 3.82 mm median and 77 proximal-pair ball estimates at 4.59 mm. The measurement isolates view registration under physical arm motion. [Mounted camera receipt](view-mounted-registration.json.gz)

The receipts include image resolution, seeds, per-frame availability and errors, first-capture recovery, and host registration timing. Under concurrent simulation load, the median first-capture wall time was 378 ms for relocating both cameras, 77 ms for adding the third, and 159 ms for restoring stereo. That interval covers marker extraction from preprocessed image colors and pose checking or fitting. Rendering, color conversion, and object tracking fall outside it. First-capture recovery measures visual evidence required. Real-time control latency remains unverified.

```bash
python -m scripts.experimental.tt18_view_trial
python -m scripts.experimental.tt18_view_mounted_trial
python -m pytest -q tests/test_tt18_view.py
```
