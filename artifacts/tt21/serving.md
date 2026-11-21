# Can the end-clamped arm improve its serve during ordinary play?

The SO-101 received one unmarked outside RGB stream at 480 × 360. Both the exploratory and frozen players began with the same four CAD-qualified serve motions and the same receiving policy. The frozen reference used the serve with the highest point yield in the earlier TT20 outside-camera evaluation. All TT21 test schedules used fresh toss seeds. A win on serve includes points won after the ensuing rally.

## Learning through complete games

A contextual bandit estimated each serve's value from the final public point score. It compared optically similar tosses, tried rarely used strokes, and grew the bank by making small changes to the most promising program. In the continuous run, the learner started with zero target observations. It kept its value memory, all new motor programs, and exploration RNG state across points and games. Both players faced the stronger physical rival with the same strict scoring and service changes.

| Game in order | Carried online learner | Fixed TT20 serve | Previous → new serve observations |
|---|---:|---:|---:|
| 1 | 2–11 | 3–11 | 0 → 6 |
| 2 | 2–11 | 3–11 | 6 → 12 |
| 3 | 1–11 | 3–11 | 12 → 18 |
| 4 | 3–11 | 2–11 | 18 → 24 |

The learner won 8 points across the four games, compared with 11 for the fixed serve. It produced 17/24 legal serves versus 20/24. Each saved game links to the previous checkpoint and contains the final value observations, available motor programs, all executed commands and replayed strict outcomes. [Continuous checkpoint after game 4](online-strokes-left-strong-cold-107499310.json.gz) · [Matched fixed game 4](online-strokes-left-strong-fixed-107499310.json.gz)

A separate learner practiced on 48 actual strong-rival service points, totaling 2,730 target control frames. It produced 31 legal serves and won 15 points during practice. We froze its learned choice and compared it with a branch given the same strokes, tosses and practice rewards shuffled across observations. In two subsequent complete games the learned policy served legally 9/12 times and won 3 points. Shuffled credit yielded 1/12 legal serves and 2 points won. The carried TT20 serve yielded 11/12 legal serves and 6 points won. The reward associations conveyed information about serving, but the inherited TT20 program remains the stronger deployment choice. [Practice tape](online-strokes-left-strong-89173722-practice.json.gz) · [Learned game](online-strokes-left-strong-learned-89305127.json.gz) · [Shuffled game](online-strokes-left-strong-shuffled-89305127.json.gz) · [Learner](../../scripts/experimental/tt21_online_strokes.py)

## Isolated reward-only practice

A second routine used the occasional visible toss location and motor commands for the opening only. It scored ten programs on two independent real tosses each. The three leading programs, including the incumbent, each played five more unique tosses before promotion. Over eight generations it observed 280 actual service points, and received only ordinary point scores as feedback. The held evaluation froze the last promoted program and replayed every scored point. On 48 fresh regular-rival tosses it won 14 versus 19 for the initial incumbent, with 37 versus 42 legal serves. The isolated drill kept the arm neutral after serving, so these counts measure its opening. [Full training stream](service-drill-standard-practice.json.gz) · [Paired regular-rival points](service-drill-standard-held.json.gz) · [Drill implementation](../../scripts/experimental/tt21_service_drill.py)

On the same 48 toss seeds against the stronger rival, the trained serve won 11 versus 15 for the incumbent and produced 37 versus 42 legal serves. The opening decision used the same initial RGB samples in both fixtures. We retained the incumbent for competition. [Paired strong-rival points](service-drill-strong-held.json.gz)
