# Ball bearing precision in the optical rally controller

The comparison keeps the SO-101 controller, fixed stereo viewpoints, native anchor opponent and strict referee. It changes the detected ball center after the stock RGB component association. The same ball observations qualify in all modes. [Bearing code](../../scripts/experimental/tt22_imaging.py)

The stock mode uses the chroma-weighted center. The same-resolution mode fits a circular silhouette to occupied and vacant pixel centers. The detail mode renders each fixed camera at 960 × 720 and searches only a small crop around the normalized stock bearing. It maps the refined center back into the original 480 × 360 pixel coordinates before combining rays with the same RGB table-line calibration. The remapping scales distances about pixel centers and preserves each camera's field of view. No simulator pose or true ball measurement enters either optical policy. [Trial](../../scripts/experimental/tt22_imaging_trial.py)

## Accuracy on archived incoming balls

The eight- and fourteen-return points replayed with the same scores and referee events. The evaluator captured each incoming segment with enough preceding frames for the causal velocity fit, then scored the controller's attack window. Stereo provided 76 estimates across 87 eligible frames. The evaluator also measured exact bearings with the RGB-fitted camera pose and stock bearings with the exact camera pose. Position error is measured against native ball location. Velocity error uses the controller's usual causal, gravity-corrected fit over recent observations. [Offline receipt](imaging-offline.json.gz)

| Bearing method | Median vertical pixel error at 480 × 360 | Median ball-height error | Median vertical-speed error | 90th-percentile vertical-speed error |
|---|---:|---:|---:|---:|
| Stock | 0.152 px | 2.92 mm | 0.0686 m/s | 2.280 m/s |
| Silhouette fit at existing resolution | 0.049 px | 1.51 mm | 0.0615 m/s | 2.278 m/s |
| Native 2× crop | 0.015 px | 1.47 mm | 0.0653 m/s | 2.408 m/s |
| Exact bearings with RGB-fitted pose | 0 | 1.38 mm | 0.0611 m/s | 2.409 m/s |
| Stock bearings with exact camera pose | 0.152 px | 1.49 mm | 0.0645 m/s | 2.277 m/s |

The evaluator labeled a bounce's first frame at the referee step and grouped it with the next frame. The large velocity errors were concentrated in those two frames.

| True flight phase | Velocity estimates | Vertical-speed error p90, stock | Silhouette fit | Exact bearings |
|---|---:|---:|---:|---:|
| Before bounce | 35/35 | 0.083 m/s | 0.082 m/s | 0.079 m/s |
| First two post-bounce frames | 20/24 | 4.423 m/s | 4.431 m/s | 4.430 m/s |
| Later post-bounce frames | 21/28 | 0.101 m/s | 0.098 m/s | 0.091 m/s |

The live comparison uses the existing-resolution silhouette fit. The extra native pixels changed height accuracy by another 0.04 mm compared with the fit and retained the large speed error after contact. This comparison covers fixed stereo.

## Native play

| Matched launch | Stock alternating returns | Silhouette-fit alternating returns | Point outcome, stock → fit |
|---|---:|---:|---|
| Archived 593435140 | 14 | 2 | Loss → loss |
| Fresh 1658245802 | 4 | 5 | Loss → win |

On the archived seed, the first requested commands differed at tick 19. The refined track led the same planner to try its first stroke at y = −0.815 m instead of −0.920 m. Both opening strokes landed legally. The second SO-101 contact with the refined track landed on its own side. On the fresh seed, the refined player's third legal return bounced twice and won the point. The silhouette fit made no ten-return rally in these two pairs. All four tapes replayed with exact referee events and point outcomes. [Paired native receipt](imaging-live.json.gz)

## Reproduction

Run from the repository root with the MuJoCo environment active.

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_imaging_trial --audit 2
python -m scripts.experimental.tt22_imaging_trial --live disk --count 1 --repeat
```

`--phase-audit` recomputes bounce labels and errors from the archived samples. `--verify-live` replays both control tapes against the referee without selecting new rival strokes.
