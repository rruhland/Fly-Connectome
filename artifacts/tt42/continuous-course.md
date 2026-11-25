# What the arm learned during a continuously random game

A single physical rally reached **38 legal arm returns and 39 opponent returns** in 61.96 seconds. Both paddles struck the same ball through the point. On every reception the opponent drew a continuous lateral, depth, height and pace request and moved its paddle in response to the incoming shot. Replay of all 3,718 joint commands reproduced each score and all 39 target draws. The median outgoing ball speed was 5.01 m/s across both paddles. All 29,739 audited CAD samples had positive clearance, with a minimum of 15.1 mm. The arm used stereo RGB, joint encoders, a camera-initialized dynamics model, earlier rewarded motion examples and new score-gated motor updates. [Watch the complete rally](watch-38.mp4) · [Action tape](random-38-tape.json.gz) · [Replay and target verification](random-38.json)

## What changing the weights of past experience actually did

The paired control restores the exact motor memory available before each game, freezes updates during that game, then reconstructs the original live learner's memory before the next pair. All 12 consecutive games between practice checkpoints 12 and 24 were included. On the record point, the first 258 commands were identical, and learning raised the score from 6 to 38. Across the 12 pairs the total rose from 37 to 69. Two games improved, two declined and eight tied. The record point accounts for the entire net gain. [Paired games, score timing and identical action prefixes](within-13-24.json.gz)

## Carrying the learning to other releases

The 48 practice games and 24 assessments were assigned before play. Measured openings are disjoint between the two groups and the opponent received fresh opening and reply random seeds. Each frozen checkpoint faces exactly the same 24 assessment games with the original full opponent target ranges. Each ball legally returned by the opponent to the arm counts as an opportunity, including those the arm never touched. All 24 games count in the total.

| Direct motor practice games completed | Legal arm returns | Actual opportunities | Median game | Longest game | Games with at least 10 arm returns |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 102 | 123 | 3.5 | 12 | 5 |
| 12 | 135 | 156 | 4 | 22 | 5 |
| 24 | 87 | 108 | 2.5 | 19 | 2 |
| 36 | 124 | 145 | 3 | 22 | 3 |
| 48 | 150 | 171 | 3.5 | 30 | 7 |

The 12-game practice gain disappeared at game 24 and recovered by game 48. Compared with the starting checkpoint, the final checkpoint improved nine assignments, shortened six and tied nine. The paired game bootstrap interval for the mean gain spans −0.33 to +4.88 returns per launch. [Whole-game scores, coverage and uncertainty](retention-48-report.json) The number of opportunities depends on how long rallies survive. Training on zero games starts from the carried teacher and prior short motor fragments. A full-game score includes misses on the opening ball. [Preassigned course](direct-course.json) · [Initial evaluation](no-more-practice.json.gz) · [After 12 games](after12.json.gz) · [After 24 games](after24.json.gz)

## Which alternative movements were available

On 145 earlier practice states, we issued alternatives from three versions of the geometric motor library from the same native physical state. Choosing each library’s top-ranked movement landed 90, 100 and 102 balls. An oracle selecting among those three top actions could land 104, or 108 if allowed to select any of the 75 proposals. This ceiling applies to those proposals at the sampled decision times. A learned selector tested on disjoint release groups landed 100. Improving the motion proposals and getting into position earlier offer more scope than switching among those three existing libraries. [Practice measurement](portfolio-practice.json) · [Grouped validation](portfolio-validation.json)

A separately trained within-state action advantage selector made 38 returns on twelve geometric games compared with 36 originally. On the older direct-joint comparison, substituting learned camera uncertainty produced 101 returns and rechecking cached alternatives on a second camera exposure produced 54, compared with 109 for the original. These datasets and actors differ from the 24-game checkpoint comparison above. [Physical advantage test](advantage-games.json.gz) · [Optical uncertainty](vision-risk.json.gz) · [Second exposure](contingent.json.gz)

The locally fitted motor-response correction was stopped after 13 paired games. It issued the same commands as the unmodified controller throughout all 13 and made 77 returns in both arms of the comparison. No proposed correction met the acceptance conditions. A separate multimodal readout produced 393 simulated contacts among 1,226 extra trial movements on twelve other games, and likewise changed no issued command. A virtual touch alone was a poor proxy for a legal landing. [Local response tape](local-motor-response.json.gz) · [Multimodal control and independent replay](../tt43_motor_modes/README.md)

## Does correct credit timing matter?

Across the same twelve starting memories, aligned live updates made 69 returns, temporally misassigned updates made 47 and frozen weights made 37. The 38-return live point became 13 with scrambled credit and six with updates frozen. Both update arms retained the same executed movements per eligible window, so allowing the memory to grow explains only part of the difference. A change in the learner's departure can change what physical ball comes back on the same next random draw. [Sham-credit paired actions and assignment](misaligned-13-24.json.gz)

## Does narrowing the target during practice transfer?

From the same starting policy, 18 graded practice games earned 129 returns, and 18 full-width practice games earned 104. The graded learner reached the full opponent range during practice. With both learners frozen on 24 full-width games, graded practice produced 131 returns, a median of 3.5 and a peak of 24. Full-width practice produced 122 returns, a median of 3 and a peak of 17. Neither matched the 150 from 48 full-width practice games, which took a larger practice budget. [Graded training](graded-practice.json.gz) · [Equal-game control](uniform-practice.json.gz) · [Matched assessment](graded-report.json)

Screening every previously discarded candidate on eight full games added 966 full-flight simulations after weak stereo-guided decisions. Zero additional joint movements earned a legal simulated landing. Both branches issued identical commands and earned the same 52 returns. [Additional candidate rollouts](loss-screen.json.gz)

On eight prespecified deliveries of the 38-return point, we replayed all previous commands and updates, then removed only the memory gained since point entry before receiving that particular ball. The same physical far contact and near bounce were verified in each fork. The learner returned deliveries 2, 4, 8, 16, 24, 32 and 38 with either memory. Both missed delivery 39. Among these offers only delivery 38 changed command, and both versions landed legally. Learning made the full point longer by changing its path earlier. This test found no increase in independent coverage on the eight sampled balls. [Matched individual deliveries](record-deliveries.json.gz)

Preserving the original teacher motions and evicting locally redundant scored movements at capacity earned 294 returns in 48 real practice games. The ordinary 768-slot sliding window earned 282. Frozen on the same 24 different launches, the diversity-preserving memory made 176 of 197 incoming balls, versus 150 of 171 for the ordinary memory. Ten games improved, seven shortened and seven tied. Both medians were 3.5 and both produced seven games of at least ten returns. [Retention comparison](diverse-retention.json) · [Learned checkpoint](diverse-learning-48.npz)
