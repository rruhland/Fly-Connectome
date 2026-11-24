# Measuring non-reversing paddle deflections from stereo

The older outgoing observer required the approaching ball to reverse its table-wise motion. That rule misses touches that send the ball sideways, downward, or still toward the arm. The new observer uses only synchronized camera measurements of the orange ball and red blade. It fits short tangents on either side of a candidate change and subtracts the acceleration expected under uninterrupted flight. A candidate must come from near the observed blade with two tracks that meet closely in time and space. Its evidence is a change in the velocity vector, regardless of sign. It returns the signed velocities and the time at which the post-event frames made the estimate available. [Estimator and calibration](../../scripts/experimental/tt36_impact.py)

The free-flight model and the jump threshold were fitted from camera tracks on 21 practice episodes on three releases. The acceleration regression used five-frame segments visibly away from the blade and above the table. The threshold was the 97.5th percentile of velocity changes on away-from-blade windows, with a 0.40 m/s minimum. The calibration code did not load physical contact labels or recorded ball velocities. Those were held in a separate audit file. [Training images reconstructed as RGB tracks](impact-practice-rgb.json.gz) · [Calibration state](impact-model.json)

The held set contained four inverse-attempt tapes each on 6253, 6861 and 12373, plus eight nominal raw-pupil points on other releases. All held release IDs were disjoint from the camera calibration releases. Each tape was reproduced from its saved commands and opponent seed. We recorded a 60 Hz view for the diagnostic, using alternate exposures for the original 30 Hz cadence. The comparison includes the old observer with its original ball tracker and with the same 30 Hz stereo points supplied to the new detector. Native labels entered only after both observers had emitted their results. [Held RGB tracks](impact-held-rgb.json.gz) · [Physical labels used for evaluation](impact-held-oracle.json)

| Split and observer | Detected touches | Detected nonforward touches | Unmatched detections | Median signed 3D velocity error |
| --- | ---: | ---: | ---: | ---: |
| Practice, prior reversal at 30 Hz | 3/14 | 0/8 | 0 | 0.297 m/s |
| Practice, same-input reversal at 30 Hz | 2/14 | 0/8 | 0 | 0.337 m/s |
| Practice, motion change at 30 Hz | 6/14 | 1/8 | 0 | 0.194 m/s |
| Practice, motion change at 60 Hz | 14/14 | 8/8 | 0 | 0.141 m/s |
| Held, prior reversal at 30 Hz | 7/14 | 0/6 | 0 | 0.293 m/s |
| Held, same-input reversal at 30 Hz | 6/14 | 0/6 | 0 | 0.342 m/s |
| Held, motion change at 30 Hz | 7/14 | 2/6 | 0 | 0.169 m/s |
| Held, motion change at 60 Hz | 11/14 | 4/6 | 0 | 0.145 m/s |

The contact matching window was ±67 ms. There were no alarms on the eight wholly no-touch episodes in either split. For the seven same held contacts that both the prior observer and the new 60 Hz observer detected, median velocity error was 0.293 vs 0.145 m/s. Across the 11 held 60 Hz estimates, mean absolute errors in x, y and z were 0.032, 0.112 and 0.059 m/s. All 11 reported the correct y sign. The estimate became available a median of 60 ms after contact. The three held misses at 60 Hz each had at most one triangulated ball position in the 100 ms after impact, including zero for 6861 attempt 9. This observer does not fill in a velocity when image evidence is missing. [Per-contact results and frame support](impact-results.json.gz)

On 6253, both attempts 8 and 10 stayed scoreless despite touching the paddle. The old observer saw neither touch. The new 60 Hz estimates, available 42 ms after contact, were `(−1.143, −2.899, −1.666)` and `(−1.180, −2.728, −1.868)` m/s in x, y and z. The physical separation vectors were `(−1.165, −2.882, −1.759)` and `(−1.197, −2.732, −1.951)` m/s. In both cases the ball continued toward the arm and down. The new observer provides feedback on these failed strokes, but the observation experiment made no changes to the command or score.

Use `rgb_point(frame, rig)` on each synchronized exposure and pass observed `(time_s, ball_xyz, blade_xyz_or_None)` triples to `estimate_impacts(track, models['60'])`. For existing captures, `python -m scripts.experimental.tt36_impact fit` then `evaluate` reproduce the comparison. To recreate the captures, run the same module with `practice` and `held` first. The engine events matched the source tape on all 41 replays, which took 124 CPU seconds total with EGL. [Observer code](../../scripts/experimental/tt36_impact.py) · [Calibration](impact-model.json) · [Practice evaluator labels](impact-practice-oracle.json)
