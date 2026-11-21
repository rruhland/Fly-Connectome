# Ball depth after losing a camera

A short continuation from stereo recovered ball depth on 19 of 24 held one-camera frames, with 8.6 mm median 3D error. An apparent-size estimate learned during scored play recovered 38 of 55 one-camera frames at the native game resolution, with 19.2 mm median error. Those 55 frames span two learner service tosses. In the live comparison, the extra ball tracks changed the player's commands and both tosses still ended without racket contact. [Held course receipt](mono-depth.json) · [Scored-game replay receipt](mono-game.json.gz) · [Live receipt](mono-live.json.gz)

## Held camera-removal course

Six new camera and trajectory seed pairs used the played landmark hall fixture at 256 × 192, with the table resized on three courses. Each course collected 20 synchronized source RGB frames before dropping to one view for four frames. Stereo contributed accepted depth on 19–20 source frames per course and gave ten apparent-size samples for the surviving camera on each course. The evaluator scored Euclidean 3D error against MuJoCo ball positions. The estimators received registered camera fits, current colored-ball blobs and earlier RGB triangulations. The fixture used zero calibration motor commands. [Trial and seed pairs](../../scripts/experimental/tt18_monocular_trial.py)

| Ball-depth method | Frames covered | Median 3D error | P95 3D error |
| --- | ---: | ---: | ---: |
| Stereo-only abstention | 0/24 | — | — |
| Matched source median depth | 24/24 | 170 mm | 238 mm |
| Raw learned appearance | 24/24 | 599 mm | 1,735 mm |
| Learned appearance within the range gate | 0/24 | — | — |
| Gravity continuation constrained to the surviving ray | 19/24 | 8.6 mm | 27.5 mm |
| Gravity continuation alone, on the same 19 frames | 19/19 | 11.6 mm | 34.9 mm |
| Linear continuation constrained to the ray, on the same 19 frames | 19/19 | 35.8 mm | 94.7 mm |

The continuation uses four consecutive stereo estimates and known gravity. It fits position and velocity, projects the prediction onto the new ray, and increases its range allowance with source scatter and elapsed time. It abstains when the latest stereo anchor is more than four frames old or the current ray disagrees too far with the predicted flight. All 12 frames in the first two positions after camera removal were admitted, compared with 7 of 12 in the later two positions. The estimated range interval contained the true along-ray range on all 19 admitted frames, with a median half-width of 53 mm. A contact can invalidate the extrapolation, particularly when it changes depth more than bearing. [Estimator](../../scripts/experimental/tt18_monocular.py)

The appearance method learns the typical product of optical depth and normalized apparent size for a particular registered view. It uses the weighted orange area, the camera focal length and the ray angle. Its gate requires at least eight stereo examples, a spread of observed source depths, support for the current apparent size and a range half-width up to 150 mm. At 256 × 192, the surviving view usually sees one or two orange pixels, which makes the apparent-size interval too wide for that gate. No physical ball size is supplied to either method.

## Replay of the scored camera-change game

The replay reproduced the original RGB digests and referee outcomes for the first eight points at 384 × 288. The first six points supplied 617 RGB frames, of which 603 produced accepted estimates for the size learner. The retained view used up to 256 estimates at its current mount. The next two points began with only that camera, so each rally's motion continuation had zero stereo anchors. The raw appearance method reported on 46 of 55 frames. Its gate admitted 38, with 19.2 mm median and 77.8 mm p95 3D error. The true along-ray range fell inside all 38 admitted intervals, which had a median half-width of 115 mm. The matched median-depth control, pooled over the earlier flight paths, produced 1,415 mm median and 1,491 mm p95 error on the same 38 frames. The measured appearance performance covers two close tosses from one retained camera. [Game replay script](../../scripts/experimental/tt18_monocular_game.py) · [Source game receipt](viewplay-1253086157-adaptive.json.gz)

A second replay comparison exposed only stream B at frames 20–23 of each of the six stereo points. The continuation had the required source observations and recovered 16 of 24 frames at 5.9 mm median and 16.0 mm p95 3D error. Gravity continuation alone on those 16 frames yielded 11.3 mm median and 45.7 mm p95. The ray-constrained linear control yielded 33.4 mm median and 73.5 mm p95. The underlying actions and outcomes came from the recorded adaptive game in both replay comparisons.

## Matched live scoring prefix

The live branch started from the same TT17 player and used the same first eight points. The first six points matched the reference RGB digests, requested actions and referee outcomes exactly. They supplied 600 accepted stereo size-training frames through the controller's existing ball sighting gate. The two single-camera points used the learned apparent-size estimates and passed the range allowance to the flight filter. The player received 39 ball tracks in 55 captures and had zero blade position or pitch sightings. It changed 49 of its 53 requested commands, missed both tosses and ended the prefix down 3–5, the same result as the branch that abstained. Both branches made zero legal serves on the two one-camera points. All eight live action tapes reproduced their applied commands, referee events and score. [Live trial and verifier](../../scripts/experimental/tt18_monocular_play.py) · [Live receipt](mono-live.json.gz)

```bash
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt18_monocular_trial
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt18_monocular_game artifacts/tt18/viewplay-1253086157-adaptive.json.gz
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt18_monocular_play artifacts/tt18/viewplay-1253086157-adaptive.json.gz
```
