# Paddle-relative joint feedback

The direct pupil made **6 own legal returns and 13 alternating legal returns** on a broad random-opponent point it had never seen. The matched absolute-coordinate pupil made no own return on that release. Across all 24 TT37 held opening cards, the body-relative pupil made 23 legal returns, including 12 on later deliveries. The same policy fit and motor limits in absolute coordinates produced 16 legal returns, including 6 later. The six-return point does not establish a reliable six-return floor, and the target of 100 own returns remains unmet. [Paired physical results](position-opening-results.json) · [Body-relative tape](position-held_openings.json.gz) · [Absolute control](../tt37/long-rgb-held_openings.json.gz)

All 22 offered first balls matched exactly, and the two cards without a legal opening stayed identical. The pupil improved six points and regressed on two, including one four-return control rally it could not start. Action-only replay of the six-return point reproduced the entire referee stream and measured placements. The full CAD audit found zero penetrations in 5,242 solver samples and at least 16.0 mm clearance. [Independent verification](position-opening-results.json)

## What changed

The pupil still outputs one bounded joint setpoint per tick. It uses ordinary synchronized RGB ball observations and its joint encoders. The position variant weights both absolute ball position and the simultaneous ball-to-paddle-center vector when fitting the next command. Absolute coordinates remain informative because the net, table and gravity do not move with the arm. Paddle center comes from a separately calibrated function of encoder angles. No receiving plane, predicted crossing of the table, stroke selector, simulator copy, referee state, or opponent plan supplies the command. The position-only physical comparison uses the same 14,757 practice observations, learned joint labels, regression and motion limits as the absolute control. [Controller](../../scripts/experimental/tt38_body_relative.py) · [Frozen RGB training bank receipt](../tt37/long-rgb.json)

The body map uses a finite trigonometric basis fitted by least squares to 1,536 offline nominal rigid-body geometry labels with no ball. A separate diagnostic replay matched encoders and images on 120 training exposures. The median distance from the native paddle center was 1.13 mm for the calibrated encoder map and 5.29 mm for the stereo red-silhouette center. The native paddle readings appear only in that diagnostic. Given this discrepancy, directly replacing the map with the visual centroid would inject more location noise in this fixture. [Body calibration](body.json) · [Paddle sighting diagnostic](camera-audit.json)

## Holding out whole releases

The table shows how well each nonbackpropagated readout predicts the mentor's issued setpoint on the final 130 ms before a legal near contact. The blade error is the distance between the *command-equivalent* centers, not the actual paddle at impact. Each fold excludes every sample from one measured release, all continuation seeds included. [Complete cold-release metrics](leave-release-out.json)

| Readout | Joint target MAE (rad) | Nominal blade median / p90 error (mm) |
| --- | ---: | ---: |
| Absolute | 0.0894 | 41.5 / 97.0 |
| Paddle-relative position | 0.0867 | 41.3 / 93.5 |
| Also paddle-relative velocity | 0.0852 | 41.3 / 90.5 |
| Also transport predicted blade displacement through local Jacobian | 0.0815 | 34.5 / 83.9 |

An augmentation check generated 5,148 analytic teaching examples by translating the RGB ball position together with the nominal paddle centers implied by the observed joints and by the issued setpoint. The translations were within 37 mm laterally and 27 mm vertically. They did not improve actual held-label fidelity over the unaugmented position-and-motion field. Held last-130-ms blade p90 was 90.5 mm without augmentation, 91.2 mm with translation, and 93.8 mm with only the ball translated. These are synthetic shifts of position, not legal counterfactual strokes. They did not preserve the blade face or impact impulse. [Analytic labels and matched visual-only check](augmentation-leave-release-out.json)

## Reproduce

Use the pinned measured launch corpus and stock broad opponent, documented in [TT37 learning](../tt37/learning.md). No card from the reserved TT37 *school* confirmation roster was used.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt38_body_relative calibrate
python -m scripts.experimental.tt38_body_relative offline artifacts/tt38-body-relative/leave-release-out.json
python -m scripts.experimental.tt38_body_relative_camera artifacts/tt38-body-relative/camera-audit.npz
python -m scripts.experimental.tt38_body_relative_augment compare artifacts/tt38-body-relative/augmentation-leave-release-out.json
python -m scripts.experimental.tt38_body_relative physical artifacts/tt38-body-relative/position-held_openings.json.gz position held_openings
python -m scripts.experimental.tt38_body_relative_report artifacts/tt37/long-rgb-held_openings.json.gz artifacts/tt38-body-relative/position-opening-results.json artifacts/tt38-body-relative/position-held_openings.json.gz --replay
python -m pytest -q tests/test_tt38_body_relative.py
```
