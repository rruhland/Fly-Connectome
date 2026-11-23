# Four-drop optical recalibration of an altered SO-101

The experiment carries the same demonstrated stock motor readout onto the alternate SO-101. The simulated change stretches the upper link by 15%, changes the elbow axis, moves and re-aims the wrist camera, and adds 42 g at its mount. The changed regime also reduces actuator gain and force and increases ball drag and contact damping. The fitter begins with the original vendor link frames. The alternate geometry, masses and moving poses do not enter the fit. [Implementation](../../scripts/experimental/tt24_public_morph.py)

Four altered-arm drops of ball-independent joint commands supplied 154 control frames, or 2.57 simulated seconds. The fit uses synchronized wrist RGB, two calibrated stationary RGB views and own joint angles. It updates the implied link shifts and axis orientations from disk silhouettes with regularization toward the stock links. It re-estimates the moving camera's optical registration and fits ball velocity from recent wrist estimates and stereo motion. The mounted circular disk's oriented face is estimated from silhouettes and four visually detected near-disk velocity reversals. The optimizer saw no reward, contact flags, moving simulator poses, or new demonstrated actions. The fixed cameras were used only in calibration. [Probe commands and fit metrics](public-morph-calibration.json.gz) · [Frozen observer](public-morph.npz)

The four drops yielded 115 matched wrist and fixed-camera ball measurements and 107 motion fit samples. Median RGB ball registration error was 4.85 mm in fitting, and paired silhouette support RMSE decreased from 2.36 to 2.10 px. One regularized link term reached its fit bound. The fitted parameter changes are an effective observer and do not establish recovery of the alternate arm's actual dimensions.

## Held physical simulation

All four treatments used the same 16 previously unseen releases and a 450-frame horizon. The three active treatments used the same frozen stock motor readout. At test the active observers used the onboard wrist camera and encoders. All recorded commands replayed to exactly the same physical judging metrics.

| Observation or control | Mean longest streak | Median | Maximum | Drops with ≥5 | All verified rebounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four-drop recalibrated RGB observer | 7.94 | 7 | 19 | 13/16 | 144 |
| Untouched stock public observer | 4.19 | 3 | 12 | 6/16 | 84 |
| Prior alternate metric RGB observer | 6.56 | 4.5 | 20 | 8/16 | 136 |
| Stationary arm | 2.31 | 2 | 4 | 0/16 | 37 |

The new observer improved the longest streak on 12 of 16 drops against the untouched observer, with two ties. The mean paired gain was 3.75 rebounds, with a 95% paired bootstrap interval of 1.63 to 6.31. Relative to the prior alternate metric observer, the gain was 1.38 with an interval of −1.25 to 3.56. That prior observer uses an RGB-calibrated residual on a stock observation map initially trained with privileged supervision. [Held release seeds and exact action tapes](public-morph.json.gz)

The stock observer and motor checkpoint hashes were preserved. Four independent stock returns before and after qualification yielded the same streaks of 9, 5, 14 and 4 and exactly the same actions. The new observer had 292 missing or rejected wrist readings over 3,702 held frames. Its poorest held run ended at one verified return. These results cover one changed simulated embodiment using the original five action channels. The motor readout still derives from the earlier stock demonstrations. [Source public observer](public-body.md)

## Reproduce

Run `OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -m scripts.experimental.tt24_public_morph`. Use `--fit-only` or `--evaluate-only` to run either stage independently. For deployment, load the original `WristPupil` and replace its `sensor` with a fresh `load_morph()` per drop.
