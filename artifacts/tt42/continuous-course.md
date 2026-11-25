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

The 12-game practice gain failed to survive the next 12 games. The number of opportunities depends on how long rallies survive. Training on zero games starts from the carried teacher and prior short motor fragments. A full-game score includes misses on the opening ball. [Preassigned course](direct-course.json) · [Initial evaluation](no-more-practice.json.gz) · [After 12 games](after12.json.gz) · [After 24 games](after24.json.gz)

## Which alternative movements were available

On 145 earlier practice states, we issued alternatives from three versions of the geometric motor library from the same native physical state. Choosing each library’s top-ranked movement landed 90, 100 and 102 balls. An oracle selecting among those three top actions could land 104, or 108 if allowed to select any of the 75 proposals. This ceiling applies to those proposals at the sampled decision times. A learned selector tested on disjoint release groups landed 100. Improving the motion proposals and getting into position earlier offer more scope than switching among those three existing libraries. [Practice measurement](portfolio-practice.json) · [Grouped validation](portfolio-validation.json)

A separately trained within-state action advantage selector made 38 returns on twelve geometric games compared with 36 originally. On the older direct-joint comparison, substituting learned camera uncertainty produced 101 returns and rechecking cached alternatives on a second camera exposure produced 54, compared with 109 for the original. These datasets and actors differ from the 24-game checkpoint comparison above. [Physical advantage test](advantage-games.json.gz) · [Optical uncertainty](vision-risk.json.gz) · [Second exposure](contingent.json.gz)
