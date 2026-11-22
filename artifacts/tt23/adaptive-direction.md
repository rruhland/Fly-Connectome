# A source-trust learner replayed on two release orders

The duration-only learner recovered some legal openings on independent flights, then gave up most of the frozen native policy's advantage on ordinary launches. After 48 selected practice points, both initializations performed almost identically. The warm policy averaged 14.31 legal openings versus 13.73 with flat initialization on the ordinary held bank, and 22.14 versus 22.48 on the independent-flight bank. [Frozen protocol](adaptive-direction-plan.json.gz) · [Complete paired trajectories](adaptive-direction.json.gz)

| Held bank, 64 starts each | Warm adaptive | Flat adaptive | Frozen native | Center |
|---|---:|---:|---:|---:|
| Ordinary | 14.31 | 13.73 | 24 | 13 |
| Independent flight | 22.14 | 22.48 | 16 | 25 |

The paired held-launch difference between warm and flat was +0.58 openings on ordinary releases, with a 95% interval of +0.38 to +0.80. On independent flights it was −0.34, with interval −1.93 to +1.15. These intervals average the two orders and 256 shared exploration draws before resampling held launches. The gain over frozen native on independent flight came alongside the loss on ordinary releases, so the held full-rally gate stayed closed.

## Reordering the physical practice

| Order | Ordinary after 0 / 12 / 36 / 48 points | Independent flight after 0 / 12 / 36 / 48 points |
|---|---|---|
| Ordinary → independent → ordinary, warm | 24.0 / 16.5 / 13.5 / 14.3 | 16.0 / 12.6 / 21.8 / 22.2 |
| Independent → ordinary → independent, warm | 24.0 / 18.2 / 15.7 / 14.3 | 16.0 / 24.6 / 23.4 / 22.1 |

Each schedule visited the same 48 distinct physical starts, with the same 12-point blocks in different orders. The warm and flat learners used one actual outcome from each visit. Their mean simulated exposure over 48 visits was 75.7 and 75.3 seconds respectively. Center used 81.6 seconds on the same visits. The warm learner averaged 13.9 legal practice points, flat averaged 13.4, frozen native had 15 and center had 19. The legal labels served only the evaluator. [Physical bank hashes, draws and exposures](adaptive-direction.json.gz)

After training, the warm policy held on about 18.5 of 28 visible mixed approaches, followed the frozen direction on 5.2 and reversed it on 4.4. Flat held about 18.4, followed on 2.8 and reversed on 6.8. The adaptive policy also held on roughly 57 of 64 ordinary held starts. On the 48 held native Pong points, the warm policy fell from 15 first returns before target exposure to an average of 8.83 after it. Mean native point duration fell from 3.348 to 2.773 seconds. The final flat values were 8.64 first returns and 2.762 seconds. In this two-context rule, adaptation toward holding carried across domains together with a loss of the useful ordinary direction. [Held camera conditions and native scoring](adaptive-direction.json.gz)

## Causal update

The agent starts with the same three-action menu on both schedules, with reflection around the observed mover's lateral side. It learns separate action estimates for visible position and velocity pointing together or apart. The warm prior puts a small initial bias on the native choice. The flat prior gives all actions equal weight. Each rank is computed from the just-completed point's duration and that agent's earlier selected durations within its observed camera group. Native release families and physical opening flags are used for evaluation. The training and held launch IDs are all disjoint. The analysis replayed existing physical tapes. Replacing every unchosen duration left the four audited learning histories unchanged. [Design](adaptive-direction-design.md) · [Learner](../../scripts/experimental/tt23_adaptive_direction.py)

Reproduce with `python -m scripts.experimental.tt23_adaptive_direction` in the project environment.
