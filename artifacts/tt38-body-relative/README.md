# Paddle-relative joint feedback

The direct pupil made **6 own legal returns and 13 alternating legal returns** on a broad random-opponent point it had never seen. The matched absolute-coordinate pupil made no own return on that release. Across all 24 TT37 held opening cards, the body-relative pupil made 23 legal returns, including 12 on later deliveries. The same policy fit and motor limits in absolute coordinates produced 16 legal returns, including 6 later. The six-return point does not establish a reliable six-return floor, and the target of 100 own returns remains unmet. [Paired physical results](position-opening-results.json) · [Body-relative tape](position-held_openings.json.gz) · [Absolute control](../tt37/long-rgb-held_openings.json.gz)

All 22 offered first balls matched exactly, and the two cards without a legal opening stayed identical. The pupil improved six points and regressed on two, including one four-return control rally it could not start. Action-only replay of the six-return point reproduced the entire referee stream and measured placements. The full CAD audit found zero penetrations in 5,242 solver samples and at least 16.0 mm clearance. [Independent verification](position-opening-results.json)

The gain did not transfer to the other TT37 roster. On 16 cards drawn by changing the continuation after four familiar physical opening balls, the position readout scored 7 legal returns against 15 in the absolute control. Both landed 3 later returns. The position readout converted only one of four distinct opening balls, compared to three in the control. Its best point was 3 own returns against 2 in the control. [Matched familiar-openings check](position-reply-results.json)

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

On the original 24 novel openings, both more elaborate readouts performed worse in play even though they copied the mentor more faithfully offline. [Motion ablation](motion-opening-results.json) · [Transported targets](transported-opening-results.json)

| Readout | Legal first | Legal later | Best own / alternating |
| --- | ---: | ---: | ---: |
| Absolute | 10 | 6 | 4 / 9 |
| Paddle-relative position | 11 | 12 | 6 / 13 |
| Also paddle-relative velocity | 11 | 1 | 2 / 5 |
| Also transport predicted blade displacement | 6 | 1 | 2 / 5 |

Action-only replays made the gap between command fidelity and return quality concrete. On release 6458, both pupils touched the first ball, but the paddle's upward speed at contact fell from 0.64 to 0.16 m/s and the ball's longitudinal velocity on the next control exchange changed from 3.14 m/s toward the opponent to 1.00 m/s away. On 3979, the position readout reduced first-contact distance from the paddle centroid from 69 to 24 mm and the resulting forward velocity improved from 1.80 to 3.10 m/s. These simultaneous differences in position, orientation and speed do not isolate a single cause. [Contact-only measurement](contact-quality-audit.json)

## Mistakes as online supervision

An online residual used unsmoothed ball positions triangulated from the pupil's own RGB and synchronized paddle positions from the body map. It only assigned a signed difference when two measured exposures bracketed the *moving* paddle center in depth, near enough to be relevant. No fixed receiving plane, simulated consequence or replayed alternative action supplied credit. At terminal failure it added a bounded lateral and vertical displacement estimate to the local pre-passage sensory contexts. A damped body Jacobian carried that estimate back to current joint commands when supported. A real legal score pulse blocked updates on an older passage. [Online implementation](../../scripts/experimental/tt38_body_relative_online.py)

Sixteen practice points yielded eight memory updates and 13 adjusted control ticks. On eight paired continuation draws the frozen and learned residual both scored 4 own legal returns, all on the same opening release. The learned branch changed 14 control ticks through the held assessment, but did not convert an additional point. The difference between a commanded center shift and the resulting paddle motion still matters. An instantaneous geometric Jacobian cannot tell how much the arm will move before contact, or how that movement will change the actual impact impulse. [Practice tape](passage-practice-learn.json.gz) · [Learned memory](passage-memory.json) · [Paired carry](passage-results.json)

## Body dynamics and drawing more practice

On the larger 449-return bank, paddle-relative position scored 13 legal returns and reached a best of three own returns. The same bank in absolute coordinates scored 17 and reached four. On a separate matched 330-return bank, a damped joint solve paired the body-relative field's desired center with the absolute field's desired face orientation. It tied the matched absolute control at 13 legal returns, including 5 later, redistributing successful points. A useful geometric representation on one bank does not stay useful automatically as the training data change. [Pooled position comparison](position-pooled-results.json) · [Center/face comparison](face-matched-rollin-results.json)

The pooled tapes make the servo delay measurable. The issued setpoint implied an 85.5 mm median move to its nominal blade center, but the paddle actually moved 14.7 mm per tick. A closed-form encoder autoregression estimated position and rate from the currently issued setpoint and the two most recent encoder samples. Holding out entire releases, the median next-tick center error fell from 4.55 mm with constant encoder velocity to 3.05 mm. Over three ticks with the archived issued commands supplied, it fell from 26.62 to 11.66 mm. Median error in final paddle velocity was 0.308 m/s. These checks predict real servo movement in archived tapes, not the effect of a new ball strike. [Encoder effect calibration](servo-effects.json)

A direct pupil supplements the original absolute joint target with a small rate-limited correction in well-supported near-ball states. The desired correction comes from measured three-tick paddle displacements on similar training observations. The learned encoder model predicts which small change to the current command will reduce that displacement error. It models the arm, not the ball or opponent. [Encoder-effect readout](../../scripts/experimental/tt38_body_relative_effects.py)

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
python -m scripts.experimental.tt38_body_relative_effects fit artifacts/tt38-body-relative/servo-effects.npz
python -m scripts.experimental.tt38_body_relative_effects physical artifacts/tt38-body-relative/servo-effects-held_openings.json.gz
python -m pytest -q tests/test_tt38_body_relative.py
```
