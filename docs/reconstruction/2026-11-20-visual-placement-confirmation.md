# Camera-gated placement on independent Breakout starts

The carried controller kept its full Pong checkpoint. It used offensive placement when the camera showed a moving paddle on the opposite rail. In the sealed Breakout confirmation it returned all 153 opportunities. The unrestricted carried controller returned 120 of 125. Both returned all 40 first-five opportunities and attained a three-return rally in every course. The gated controller attained that rally 5.91 simulated seconds sooner on average, with zero misses before attainment in both carried arms.

The [registered protocol](../plans/2026-11-20-visual-placement-confirmation.md) used eight matched physical courses with six lives each, a 10 s per-life cap, both serve sides and new held-out layouts. The 48 native layouts were distinct from the original qualification. The [schedule digest](../../artifacts/m2/2026-11-20-visual-placement-confirmation-seal.json) was committed before play. The original unrestricted-transfer qualification failed its registered rapid-transfer criterion.

## Returns and time to sustain play

Three consecutive returns in one life qualified. Time includes all earlier play in the course. Censored courses receive the full 60 s cap for paired time. Misses include the scheduled life losses observed before the third consecutive return or before censoring.

| Controller | Cold first return | First five | All returns | Sustained / censored | Capped mean seconds | Misses before sustained or censor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gated carry | 8/8 | 40/40 | 153/153 | 8 / 0 | 8.75 | 0 |
| Gated reset, initial −1 | 3/8 | 21/40 | 83/105 | 5 / 3 | 27.78 | 22 |
| Gated reset, initial +1 | 0/8 | 32/40 | 129/137 | 8 / 0 | 8.72 | 8 |
| Equal-weight gated reset | 1.5/8 | 26.5/40 | 106/121 | 6.5 / 1.5 | 18.25 | 15 |
| Unrestricted carry | 8/8 | 40/40 | 120/125 | 8 / 0 | 14.66 | 0 |
| Passive | 0/8 | 0/40 | 0/48 | 0 / 8 | 60.00 | 48 |

The reset estimate averages both initialized polarities within each physical course, with both runs retained in the records. Gated carry gained 33.75 percentage points on the first five versus this estimate and was ahead in all eight courses. The paired 95% bootstrap interval was 18.75 to 48.75 percentage points and the exact two-sided sign-flip p-value was 0.0078. It also avoided an average of 1.88 misses before attainment or censoring per physical course, with fewer in all eight courses. The paired interval was 1.00 to 2.75 fewer misses.

Gated carry saved 9.50 capped simulated seconds versus the reset average and was faster in seven of eight courses. The paired bootstrap interval was 0.13 to 19.13 seconds and the sign-flip p-value was 0.133. Three censored negative-polarity runs contributed to this result. The positive reset copy attained all eight and averaged 8.72 s. Among its five attaining courses, the negative reset copy averaged 8.46 s. Gated carry spent 5.91 fewer seconds than the unrestricted carry on average, with a paired interval of 2.04 to 9.73 seconds and sign-flip p-value of 0.0234. Shot trajectories changed when the third opportunity arrived, including how often it fell in a later life.

Each course cell below gives the first-five return count, capped seconds, misses before sustained or censor, and all returns per opportunity. A dagger marks censored courses. The passive controller had `0 / 60.00† / 6 / 0/6` in every course.

| Course | Gated carry | Gated reset −1 | Gated reset +1 | Unrestricted carry |
| --- | --- | --- | --- | --- |
| 1 | 5 / 8.00 / 0 / 18/18 | 5 / 8.53 / 0 / 18/18 | 4 / 8.71 / 1 / 15/16 | 5 / 19.73 / 0 / 15/15 |
| 2 | 5 / 7.42 / 0 / 18/18 | 0 / 60.00† / 6 / 0/6 | 4 / 7.52 / 1 / 15/16 | 5 / 6.92 / 0 / 14/16 |
| 3 | 5 / 8.57 / 0 / 17/17 | 3 / 10.01 / 2 / 11/13 | 4 / 9.82 / 1 / 14/15 | 5 / 17.38 / 0 / 16/16 |
| 4 | 5 / 7.89 / 0 / 21/21 | 5 / 7.98 / 0 / 21/21 | 4 / 10.62 / 1 / 18/19 | 5 / 9.92 / 0 / 15/15 |
| 5 | 5 / 7.89 / 0 / 20/20 | 0 / 60.00† / 6 / 0/6 | 4 / 8.43 / 1 / 17/18 | 5 / 8.43 / 0 / 17/17 |
| 6 | 5 / 7.01 / 0 / 20/20 | 0 / 60.00† / 6 / 0/6 | 4 / 8.04 / 1 / 17/18 | 5 / 19.15 / 0 / 17/17 |
| 7 | 5 / 5.80 / 0 / 19/19 | 3 / 9.58 / 2 / 12/14 | 4 / 8.11 / 1 / 15/16 | 5 / 17.82 / 0 / 14/15 |
| 8 | 5 / 17.43 / 0 / 20/20 | 5 / 6.17 / 0 / 21/21 | 4 / 8.53 / 1 / 18/19 | 5 / 17.92 / 0 / 12/14 |

## Offensive placement remains active in Pong

The gate remained closed throughout Breakout, with score tables and shot memory exactly equal at the start and finish of every gated course. In a three-start ordinary Pong check the gated and unrestricted carry each returned 3/3 available opportunities and had final scores of 0, 0 and +1. The gated arm saw the opposite paddle on 603 frames and executed 88 nonzero offset commands. This check exercised offensive placement under native Pong camera input.

The [evaluation archive](../../artifacts/m2/2026-11-20-visual-placement-confirmation.zip) contains the revealed schedule, source digest receipt, 40 course records with every opportunity, paired analyses and the Pong check.
