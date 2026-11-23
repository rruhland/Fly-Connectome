# Transferring the wrist pupil

The target used the alternate arm's 15% longer upper link, tilted elbow axis, 41 mm wrist-camera displacement, and 42 g camera payload. The changed-drive regime also weakened the servo gains and force limits, increased quadratic drag by 25%, and changed paddle contact damping. The stock and altered setups used the same joint-command interface and rebound referee. [Fixture](../../scripts/experimental/tt24_bounce.py)

The source pupil came from four stock-arm teacher demonstrations. The target used nested budgets of one demonstration with 300 commands and four demonstrations with 943 commands. The fourth target demonstration ended after 43 commands and stayed in the dataset. The residual method updates the source body, wrist, and motion calibration with **privileged training supervision** and keeps the source action readout fixed. Full target-only fitting learns a separate readout and a new map on exactly the same target trajectories. Runtime observations are wrist RGB and the arm's own joint angles. [Implementation](../../scripts/experimental/tt24_bounce_ident_transfer.py)

## Fresh alternate-arm and changed-drive results

All rows use the same 16 held seeds, 248201–248216. The score is the longest sequence of consecutive verified rebounds.

| Condition | Target demonstrations | Mean | Median | Maximum | At least 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stationary | 0 | 2.19 | 2 | 3 | 0/16 |
| Frozen source pupil | 0 | 4.38 | 3 | 14 | 2/16 |
| Fresh target map, fixed source readout | 1 | 2.00 | 2 | 5 | 0/16 |
| Residual target map, fixed source readout | 1 | 5.19 | 4 | 15 | 2/16 |
| Full target-only fit | 1 | 1.50 | 1 | 3 | 0/16 |
| Fresh target map, fixed source readout | 4 | 4.31 | 4 | 6 | 0/16 |
| Residual target map, fixed source readout | 4 | 6.19 | 5 | 16 | 3/16 |
| Full target-only fit | 4 | 1.44 | 1 | 4 | 0/16 |
| Target-only coefficients, supplied source feature pair | 4 | 5.19 | 4.5 | 16 | 3/16 |
| Clean target state, fixed source readout | 0 | 9.06 | 9.5 | 16 | 8/16 |

The four-example residual map improved ten paired runs relative to the frozen source pupil, tied four, and lost two. Full target-only selection chose feature pair `[1, 5]`. Supplying the learned stock pair `[2, 5]` and refitting all target coefficients recovered the same count of long runs as residual recalibration and averaged one fewer rebound. The returned source fixture matched all commands and outcomes exactly on four additional seeds, with consecutive counts of 2, 11, 7, and 12. The stock checkpoint's SHA-256 stayed unchanged. [Paired trials and commands](bounce-ident-transfer.json.gz)

The exported residual pupil retains the exact source action weights. Its live checkpoint replay reproduced 16 consecutive rebounds on target seed 248215. [Residual transfer checkpoint](bounce-ident-transfer-map.npz) · [Target-only checkpoint](bounce-ident-transfer-target.npz)

## Paired public images

The public target recalibration used the same teacher-driven trajectories. It fit correspondences from the actual wrist and two fixed training cameras with own encoders, then aligned the fixed views to coordinates inferred by the previously fitted stock pupil. Target calibration received public image and encoder pairs. Held operation used the resulting map with wrist RGB and own encoders. This variant averaged 0.25 consecutive rebounds from one target example and 1.25 from four, with maxima of one and three. [Public calibration data and exact command replays](bounce-ident-transfer-public.json.gz) · [Public calibration implementation](../../scripts/experimental/tt24_bounce_ident_public_transfer.py)

Run `python -m scripts.experimental.tt24_bounce_ident_transfer` to regenerate the supervised transfer and source-return test. Run `python -m scripts.experimental.tt24_bounce_ident_public_transfer` for the paired public calibration using the archived stock fixed-eye images.
