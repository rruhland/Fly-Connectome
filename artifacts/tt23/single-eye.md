# A wrist-camera rally against the learned five-axis coach

The near arm returned the ball legally at least twice on all six matched starts with the command-calibrated wrist camera. Its longest point lasted 19 alternating legal returns, with nine near and ten far. The near arm’s sole live view was its native wrist RGB. [Verified wrist points](single-eye-wrist_twin.json.gz)

The near arm used a fixed prerecorded service program and the learned far coach played live. Each point continued to its strict referee result. Six paired starts came from schedule `1743013585`. All conditions used the same physical near service without toss adjustment. The earlier fixed-stereo coach archive used optical toss adjustment, so its lengths describe a separate set of starts. [Earlier stereo coach comparison](coach.md)

The first wrist condition used the existing unmarked table-line camera registration, ball-size ranging and causal temporal fit. The ball was visible in the wrist image on the first start at frames 35, 45, 51, 54 and 56, but there was no camera registration at those frames. The tracker reported insufficient court lines after the near arm moved out of the serving pose. Across the six prospective wrist starts the far side made one legal return per point and the near side made none. [Wrist with table-line registration](single-eye-wrist.json.gz)

The follow-up uses the existing command-driven factory twin to predict the pose of its wrist camera. Its initial pose, mount and lens intrinsics come from the public arm model. During play the twin receives the near arm’s own issued commands. The ball estimate comes from the single native wrist RGB stream, using apparent ball size and recent bearings. The near controller uses the existing practiced return scorer and the frozen correction from earlier stereo practice. The far coach retains its evaluator state inputs. The follow-up reuses the six openings, including the first point used to diagnose the registration failure. [Wrist with command-based camera pose](single-eye-wrist_twin.json.gz)

The external-camera control uses the same monocular ranging with the left camera’s declared fixed pose. It processes only the left RGB stream and receives the same fixed near service. All cameras run at 480 × 360 and 30 Hz. [Single external eye](single-eye-left.json.gz)

| Near view | Near legal returns | Far legal returns | Starts with two or more near returns | Rallies ≥10 | Rallies ≥20 | Longest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wrist, court-line registration | 0 | 6 | 0/6 | 0/6 | 0/6 | 1 |
| Wrist, command-predicted camera pose | 27 | 33 | 6/6 | 2/6 | 0/6 | 19 |
| One left external camera, fixed calibrated pose | 26 | 32 | 5/6 | 1/6 | 1/6 | 25 |
| Fixed stereo pair | 39 | 45 | 6/6 | 4/6 | 2/6 | 25 |

The three calibrated conditions reached their longest point on seed `1781894630`. The wrist point lasted 13.47 simulated seconds, the single external-eye point 17.00 seconds, and the stereo point 16.33 seconds. All 24 runs made the near serve legally and ended in a far-arm point win. Both arms’ tapes replayed all 24 strict referee sequences and the recorded native RGB for each sensor condition. The measured task is continuation after prescribed near service, using a known fixture and a cooperative physical far arm. [Matched summaries and serve audit](single-eye-comparison.json)

On the first matched start, seed `214242714`, the calibrated wrist and fixed-stereo branches each made four legal near returns in independently played nine-return points. The longest wrist point, seed `1781894630`, ran for 404 frames or 13.47 simulated seconds. It ended when the near arm’s next ball landed on its own half. Replaying both command tapes reproduced the referee events, terminal result and the sole native RGB digest. The clip shows both physical arms from a spectator view next to the actual wrist frames. [Watch the wrist rally](single-eye-wrist-rally.mp4) · [Video verification](single-eye-wrist-rally.json) · [Fixed-stereo matched control](single-eye-fixed.json.gz)

Run `python -m scripts.experimental.tt23_single_eye --view wrist` for the first wrist test, `--view wrist_twin` for command-based camera pose, `--view fixed` for the matched stereo control, and `--view left` for a single external camera. Each run verifies both command tapes and the native RGB. `--compare` checks that all four sets shared identical service commands and referee events through the legal serve. Render the wrist video with `python -m scripts.experimental.tt23_single_eye_movie`.
