# Anonymous visual roles on native SO-101

The frozen Pong binder did not recover the physical paddle in either fixed camera. Its moving-target selections offered no consistent improvement over image speed or reward-time shuffling. We stopped at the perception assay, with the spatial player's role inputs left untouched.

## Assay

Four held unmarked TT22 anchor points provided 390 native 480×360 control frames per eye and six strict legal near-arm returns. Each player’s physical joint command tape reproduced the original RGB digest, referee events, point outcome and duration. The two eyes assigned roles independently. RGB and the learner's preceding joint command were the only observations given to either binder.

The generic front end grouped every region by the same coarse RGB quantization, then tracked regions by continuity, recent motion, appearance and scale. No palette value or object was specified as the paddle or ball. Increasing the track list beyond the front end’s shortlist preserved all retained observed hypotheses before role ranking. The ranker used the frozen action-response and reward-time weights from 60 rendered Pong starts. The speed control used causal velocity and excluded the same learned actor choice. The shuffled-command control drew commands from the preceding 25 frames. The source reward-time control had separate mover weights and shared actor weights. A random control sampled eligible candidate identities.

The evaluator read simulator segmentation only to determine that a role had at least three visible pixels and used the simulator projection only to locate the reference center. A match required a track center within 8 pixels of the paddle center or 4 pixels of the ball center. These observations were unavailable to the binders. A receiving window covered the ten native commands before each of nine real near-arm contacts, whether or not the contact became a legal return.

## Results

Each numerator below counts correct selected identities. The denominator counts visible frames within the receiving windows. The candidate ceiling counts visible frames with at least one eligible hypothesis within tolerance, even if the ranker missed it.

| Role and eye | Candidate ceiling | Pong | Reward-time shuffle | Command shuffle | Speed | Random |
|---|---:|---:|---:|---:|---:|---:|
| Paddle, left | 15/90 | 0/90 | 0/90 | 2/90 | | 1/90 |
| Paddle, right | 0/90 | 0/90 | 0/90 | 0/90 | | 0/90 |
| Ball, left | 65/86 | 52/86 | 50/86 | 53/86 | 50/86 | 10/86 |
| Ball, right | 74/85 | 36/85 | 34/85 | 70/85 | 44/85 | 13/85 |

The Pong binder proposed a paddle on 55 of 90 left-eye and 40 of 90 right-eye receiving frames. Its median projected error was 25.1 pixels on the left and 23.7 on the right, and raising the rank margin yielded zero correct paddle selections. Only one of 390 full-point frames had a correct eligible paddle candidate in both views. The mover ranked correctly in both eyes on 23 of 84 visible receiving frames, against a joint candidate ceiling of 61 of 84. The modest left-eye mover gain over frame speed came with a larger loss on the right, and shuffled preceding commands improved mover selection in both eyes.

The paddle was visible in every receiving frame. The proposal frontend seldom supplied a durable hypothesis near the center, and Pong's action-response ranking missed the supplied left-eye hypotheses as well. This assay found no working color-free replacement for both inputs to the spatial player. Frame observations are correlated within four points. No role-free gameplay or learning-efficiency result follows from the replay.

Reproduce with `python -m scripts.experimental.tt22_anonymous_trial` and the repository's native MuJoCo EGL environment. The [raw receipt](anonymous-roles.json.gz) includes per-frame candidate and selected center errors, the two model weights and replay hashes.
