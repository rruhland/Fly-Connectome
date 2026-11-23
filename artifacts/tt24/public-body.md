# Public optical calibration of the stock wrist and paddle

The calibration used eight ordinary stock-arm drops. The learner issued bounded joint probes independent of ball motion. Two calibrated fixed cameras provided RGB observations alongside the wrist camera and five joint readings. The eight probe trajectories contained 337 control frames, or 5.62 simulated seconds. Their scored terminal records reproduce from the exact issued rates. The refits re-render those same eight recorded trajectories. [Implementation](../../scripts/experimental/tt24_public_body.py) · [Probe tapes and first fit](public-body.json.gz)

The blade fit uses the colored disk silhouettes. It projects a rim on both sides of a finite-thickness circle and compares the projected support along sixteen directions in two calibrated images. It learns the circle size and mounts the observed center and upward normal on the public vendor link frames. Shared-shape refits use the mechanical constraint that the handle lies in the paddle plane. One fits the directional supports and one fits opposing silhouette widths, which cancel position shifts. The fitted width model recovered a 73.6 mm face radius. None of the optical fits receives the simulator's moving pose, velocity, contact flags, or a teacher action. [Shared support fit](public-body-shared.json.gz) · [Width fit](public-body-width.json.gz)

The wrist registration uses 254 paired visible ball sightings. A rigid fit aligns the RGB angular-size and bearing observations with the fixed-view triangulation in encoder-derived grip coordinates. It estimates an effective ball radius of 19.7 mm and a weighted median registration error of 2.7 mm against those optical labels. The wrist velocity readout uses 230 public stereo motion samples and causal wrist history. The width checkpoint rejects gross mask outliers, requires short agreement to reacquire the ball after abrupt changes, and bounds extrapolation during gaps. [Optical and causal-motion fit](public-body-width.json.gz)

The same RGB tapes support a second normal estimate. Short stereo flight windows identify ten upward velocity reversals close to the optically fitted paddle. The method subtracts free-flight acceleration estimated from 248 stereo observations. Two reversals have estimated paddle speed below 0.14 m/s. The median grip-local impulse direction is 7.1° from the width-fitted plane normal. The fused fit uses the physical approximation that the normal impulse dominates this velocity change. It projects the robust direction onto the shaft-orthogonal plane and combines ten observed impulses with two silhouette pseudocounts. The fusion retains the width checkpoint's causal wrist observer. [Optical impulse evidence](public-body-impulse.json.gz) · [Fused fit](public-body-fused.json.gz)

## Held sensing diagnostics

A frozen checkpoint processes wrist RGB and joint readings on four independently seeded drops per row. Hidden simulator state scores the held diagnostics after fitting. Each range below is the range of per-drop RMSEs. Ball and velocity errors include all raw wrist sightings, even ones rejected by the causal filter. The variants use separate held cohorts.

| Checkpoint | Blade position | Upward normal XY | Ball position | Ball velocity |
| --- | ---: | ---: | ---: | ---: |
| Initial free per-frame face fit | 1.2–2.5 mm | 0.151–0.160 | 10.7–88.8 mm | 0.46–0.62 m/s |
| Shared support fit | 1.4–2.3 mm | 0.100–0.102 | 10.3–196.7 mm | 0.37–4.94 m/s |
| Shared width fit and consensus wrist observer | 1.1–1.4 mm | 0.134–0.135 | 9.9–13.8 mm | 0.29–0.45 m/s |
| Fused optical impulse and silhouette fit | 1.4–4.3 mm | 0.039–0.053 | 9.4–92.8 mm | 0.19–1.12 m/s |

The shared-support wrist tracker had one severe reacquisition failure. The fused tracker estimated ball position within 9–13 mm and velocity within 0.19–0.35 m/s on three of its four held drops. It rejected eleven masks on the remaining drop, which had 93 mm ball-position RMSE. The first checkpoint's fitted pose Jacobian was full rank at all four held homes. The smallest singular value after the policy bridge's normalization ranged from 0.410 to 0.498. [Fused held diagnostics](public-body-fused.json.gz) · [Initial diagnostics](public-body.json.gz)

`load_public(path)` returns an observer whose `.observe(obs)` produces ball position relative to initial blade, ball velocity, blade position relative to initial blade and upward normal XY. Its `.decoder.body(q)[12:17]` gives absolute blade XYZ and upward normal XY for the numeric inverse. The fused checkpoint's normalized Jacobians all have rank five, with a minimum singular value of 0.434 across the four held homes. The actual bridge returned five finite, bounded rates at each. [Deployable fused checkpoint](public-body-fused.npz) · [Bridge check](public-body-bridge.json)

## Reproduce

Use the existing [MuJoCo setup](../tt23/tt23_sequence.md#reproduce).

```sh
python -m scripts.experimental.tt24_public_body
python -m scripts.experimental.tt24_public_body --diagnose
python -m scripts.experimental.tt24_public_body --shared
python -m scripts.experimental.tt24_public_body --shared --diagnose
python -m scripts.experimental.tt24_public_body --width
python -m scripts.experimental.tt24_public_body --width --diagnose
python -m scripts.experimental.tt24_public_body --width --impulses
python -m scripts.experimental.tt24_public_body --fuse
python -m scripts.experimental.tt24_public_body --fuse --diagnose
```
