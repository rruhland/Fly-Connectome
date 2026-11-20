# What Pong experience carries into Breakout

The starting learner had played 96 Pong practice points and a 64-point course with changes to the controls and opponent. We loaded that exact saved state before showing it a Breakout frame. Both games use a 64×64 binary camera, three movement requests and the same paddle actuator. Breakout moves the paddle to the bottom of the screen, removes the opponent and adds destructible bricks. A life starts on a new board with the ball coming toward the player. All controllers use the same fixed visual tracker. All reported learning runs without backpropagation. [Source checkpoint](../../artifacts/m1b/2026-11-19-pong-after-mixed-course.json) · [Transfer plan](../plans/2026-11-20-pong-breakout-progression.md)

## Finding the paddle

With the camera turned 90°, 180° or 270°, the learner tried its buttons and looked for a rail that moved in response. It identified the correct rail on 72/72 unseen starts, in a median 0.333 seconds. Its first decision identified both the paddle and ball within two pixels on 65/72 starts. It returned 47/72 openings compared with 24/72 passive, then 44/47 later balls compared with 6/24 passive. Probing has a cost on the original orientation. The aligned prior returned 24/24 upright openings, while the learner that had to find the rail first returned 17/24. The adapter assumes a long thin paddle, a small moving ball and a quarter-turn geometry. [Role study and visibility audit](2026-11-20-rotated-pong-role-transfer.md)

## Entering an unseen game

Each of eight independent courses had 16 chronological lives on unfamiliar boards. Each opening required the paddle to move. The first-five column counts the first five actual opportunities of **each course**. A sustained rally requires three consecutive returns in one life. Time starts at course entry and includes earlier lost lives. Courses with no rally are charged the entire 160 seconds available. The table uses all 128 lives per arm for the last column. [Sealed protocol](../plans/2026-11-20-arcade-transfer-qualification.md)

| Starting state in ordinary Breakout | First ball of course | First five | Courses sustaining a rally | Mean game seconds to rally, capped | Returns across all lives |
|---|---:|---:|---:|---:|---:|
| Full Pong state, learning active | 8/8 | 40/40 | 8/8 | 9.65 | 324/341 |
| Pong motor and interception, score table cleared | 8/8 | 40/40 | 8/8 | 7.82 | 415/415 |
| Full Pong parameters frozen, role can align | 8/8 | 39/40 | 8/8 | 9.66 | 322/339 |
| Full Pong parameters and original orientation fixed | 0/8 | 0/40 | 0/8 | 160.00 | 0/128 |
| Matched fresh controller | 4/8 | 36/40 | 8/8 | 8.66 | 404/408 |
| Fresh controller that must estimate its own motion constants | 5/8 | 28/40 | 7/8 | 31.59 | 308/351 |
| Stationary paddle | 0/8 | 0/40 | 0/8 | 160.00 | 0/128 |

Pong experience prevented opening mistakes on four courses. The matched fresh controller started with the right direction on the other four and tied on all five opportunities. Full transfer therefore missed two original criteria. It needed to improve a majority of courses and sustain a rally sooner. Its acquired shot strategy also allowed more misses through ordinary play than the version with cleared shot scores. One measured trade-off was 446 bricks after an own return for full transfer versus 432 for motor and interception alone. Both versions saved their first ball on every course and the aligned frozen policy also did so. Early competence is evidence that an existing skill can be reused immediately once the objects are aligned. The fresh control retained the same family of numerical motion templates and could learn the missing motor and relation choices quickly.

A separate 48-life stress cohort changed button mapping, actuator timing, or both. On the first five opportunities per course, full transfer returned 4/20, motor and interception returned 6/20, and matched fresh returned 10/20. Their return counts over all 48 lives were 38/77, 47/83 and 46/82, respectively. The full learner sustained a rally in two of four courses and fresh did so in three. Simultaneously changing game and controls is a harder transfer problem.

The replay uses the very first registered ordinary life and displays exactly the binary camera each learner received. Both regenerated request sequences matched their recorded hashes. The carried learner intercepted three balls, broke two bricks and played to the ten-second limit. Its paired fresh learner missed at 1.08 seconds.

![The first scheduled Breakout life with the carried and fresh learners](assets/2026-11-20-first-breakout-life.gif)

The seven-arm comparison completed 1,232 lives and 357,093 camera decisions, representing 1.98 hours of game time in about 7.3 minutes on four CPU workers. The runner preserved failed lives and audited physics-equivalent openings across every paired arm. [Blind results and source hashes](2026-11-20-breakout-blind-transfer.md)

### Why the changed controls fooled the learner

We replayed one difficult stress course where the learner started missing immediately. The unmodified run reproduced all 12 action sequences and state hashes. After one short, scoreless life, it reversed its belief about which direction to chase the incoming ball and kept that reversal for the remaining eleven lives. Holding only that direction fixed between lives raised the total from 2 to 13 returns and rescued the first ball in five lives, while losing two openings the original had saved. In the original, both visible roles were localized to within two pixels on all 108 sampled pre-contact decisions. The active button order was wrong on six of the ten missed openings. This selected course gives a concrete hypothesis for future work: use observed movement to diagnose the buttons and timing before changing a previously successful pursuit relation. [Exact replay, reachability and intervention](2026-11-20-breakout-stress-diagnostic.md)
