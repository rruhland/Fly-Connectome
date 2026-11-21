# Bounded predictive control on the rotary receiver

The original first rotary visit hit at least one cumulative command budget on 37% and 53% of frames 17–35 of its two mixed serves. Both mixed serves in a separate earlier held installation hit that budget on 53% of the same window. This is the controller's public command budget. The physical actuators have their own limits and receive the evaluator's delayed, noisy requests. [Original course](combined-rotary.json.gz) · [Earlier held course](rotary-adaptation-confirm.json.gz)

The bounded predictive servo solves the same local ridge objective across three planned moves, with signed rates bounded by 1 and cumulative requests bounded by 3.85 after each move. A feasible active-set solver reallocates motion across coupled channels when the unconstrained plan exceeds a limit. Both players used the same RGB motor fit, mini prior, timing observer, impact adaptation, physical changes, and native reflected stereo. The solver consumes the fitted camera-space map, recent public requests, and visual timing evidence.

On the original installation, the instrumented clip controller reproduced all seven saved action hashes and score records. The unconstrained plans violated a rate or cumulative budget on 51 of 80 frames near predicted interception. On the same states, the active-set solve changed the first command by more than .01 on 50 of 80 frames and reduced the sum of the plan objectives compared with sequential clipping. Four serves contacted the paddle under each controller. [Paired receipt](bounded-mpc.json.gz)

| Course | Clip, serves returned | Bounded, serves returned | Clip / bounded, mixed serves returned |
| --- | ---: | ---: | ---: |
| Original diagnostic `48923017` | 0/7 | 0/7 | 0/2 · 0/2 |
| Fresh `99603119` | 1/8 | 0/8 | 1/4 · 0/4 |
| Fresh `73104537` | 6/8 | 5/8 | 3/4 · 3/4 |

The fresh courses produced eight legal returns with clipping and five with bounded planning. The bounded solver reported zero warnings in the final run. All recorded public rates and cumulative requests stayed within their limits. An exploratory solve using SLSQP gave the same serve counts and had frequent iteration-limit warnings. [Active-set receipt](bounded-mpc.json.gz) · [SLSQP receipt](bounded-mpc-slsqp.json.gz)

Keep clip-after-solve as the shared default. The experiment confirms saturation and an opportunity for command reallocation, but the improvement in the fitted-pose objective did not improve legal returns on these starts.

```sh
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.:src:scripts/experimental
python -m scripts.experimental.tt6_bounded_trial /tmp/bounded-mpc.json.gz
```
