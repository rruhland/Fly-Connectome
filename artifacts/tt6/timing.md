# Table tennis under hidden actuator delays

The camera player compares observed racket motion with the pose predicted from its own recent commands at delays from zero to five control frames. It updates a probability for each delay and advances the planned stroke by the estimated lag. A single camera observation can change the estimate by at most 0.28 frames. The player carries the resulting probabilities across point boundaries and updates its ordinary rally policy from public point rewards.

Each station uses the same undisturbed RGB motor probes and starts from the first-person mini-table prior. Two live practice points use a two-frame actuator delay and independent Gaussian rate noise with standard deviation 0.035. All conditions inherit the same rally state from practice. Stock uses the original servo. Frozen timing uses the practiced estimate throughout. Cold timing starts with the nominal delay prior and learns from the scored points. Continual timing retains the practiced estimate and keeps learning.

Scored points use the same serves and seeded command noise across conditions. The actuator changes from a two-frame delay to a four-frame delay after the retention phase. The within-point phase starts at two frames, increases to four at frame 10, and drops to one at frame 24. The final phase returns to the two-frame delay. Noise rises to 0.075 during the four-frame periods and is 0.050 during the one-frame period. At 30 Hz, a frame takes about 33 ms. A delay increase holds the actuator target and a decrease releases queued command increments at bounded rates. The disturbance schedule stays in the evaluator.

The main trial uses the cached native MuJoCo stereo renderer at both stations. The original-shading check uses a separate serve stream and the standard renderer. Each receipt includes the requested actions, the actuator-bound actions, the visual estimates and the physical scoring events. A 140-frame timeout leaves the point unscored. [Main trial](timing.json.gz) · [Original-shading check](timing-native.json.gz)

| Controller | Main points with a legal return, out of 24 | Main legal returns | Main wins–losses–timeouts | Original-shading wins–losses–timeouts |
| --- | ---: | ---: | ---: | ---: |
| Stock | 18 | 34 | 6–10–8 | 2–2–0 |
| Frozen practiced timing | 21 | 35 | 7–6–11 | 4–0–0 |
| Cold timing | 22 | 39 | 5–4–15 | 3–0–1 |
| Continual timing | 22 | 39 | 4–6–14 | 3–0–1 |

Across the six points with a sustained four-frame delay, stock landed zero legal returns, frozen practiced timing landed seven, and each online condition landed eleven. The continual controller's mean absolute error relative to the scheduled delay was 0.28 frames after the first twelve observations, compared with 2.24 frames for the frozen controller. The frozen controller won three points in that phase. The online controllers won zero.

The continual controller won the recorded original-shading point after both hidden switches. It contacted the ball on frame 29 and landed a legal return on frame 47. [First-person replay](timing-in-point.mp4)

## Gated timing under a wider change

The gated controller starts from its practiced estimate. It changes the stroke lead after four concentrated visual estimates agree within 0.65 control frames. The new lead moves gradually and carries across points. The gated trial starts with two practice points, then plays four matched points at each of three hidden settings. The delay is zero, five, then zero frames. The rate noise standard deviation is 0.025, 0.060, then 0.050. Five frames is about 167 ms.

Across two stations, the gated controller landed ten legal returns on six of eight five-frame points. It recorded two wins, three losses, and three timeouts in that phase. Both controls lost all eight points without a legal return. The two gated wins at the five-frame delay came from the first station. After the first twelve observations, the gated estimate averaged 0.26 frames of absolute error relative to the scheduled delay. The frozen estimate averaged 4.41 frames. [Gated trial](timing-gated.json.gz) · [Independent station and serve stream](timing-gated-confirm.json.gz)

| Controller | Points with a legal return, out of 24 | Legal returns | Wins–losses–timeouts |
| --- | ---: | ---: | ---: |
| Stock | 14 | 25 | 4–12–8 |
| Frozen practiced timing | 14 | 21 | 6–12–6 |
| Gated continual timing | 19 | 31 | 10–6–8 |

## Reproduce and replay

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt6_timing_trial /tmp/timing.json.gz --seed 80193572 --installations 2 3 --workers 1 --training 2 --phase-points 3 --cache /tmp/tt6-probes --cache-stereo
python -m scripts.experimental.tt6_timing_trial /tmp/timing-native.json.gz --seed 93158302 --installations 1 --workers 1 --training 2 --phase-points 1 --cache /tmp/tt6-probes
python -m scripts.experimental.tt6_timing_trial /tmp/timing-gated.json.gz --seed 204673576 --installations 0 --workers 1 --training 2 --phase-points 4 --conditions stock frozen continual --protocol separated --cache /tmp/tt6-probes --cache-stereo
python -m scripts.experimental.tt6_timing_trial /tmp/timing-gated-confirm.json.gz --seed 8053011 --installations 1 --workers 1 --training 2 --phase-points 4 --conditions stock frozen continual --protocol separated --cache /tmp/tt6-probes --cache-stereo
```

To replay a point with FFmpeg, select its controller and point index. The replay sends the recorded actuator-bound commands to MuJoCo and verifies the physical outcome.

```sh
python -m scripts.experimental.tt6_timing_replay artifacts/tt6/timing-native.json.gz /tmp/timing.mp4 --condition continual --course 0 --point 2
```

The tape audit regenerated the same actuator commands and physical outcomes for all 194 practice and scored points across the four receipts.
