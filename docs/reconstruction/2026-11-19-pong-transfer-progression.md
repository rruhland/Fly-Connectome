# Carried Pong learning across changes

The starting checkpoint could intercept an incoming ball and learn a new button arrangement over several points. This progression added three decisions at different timescales. The controller estimates what its buttons do from visible paddle motion, estimates when a request will take effect, and chooses where to meet an incoming ball using the outcome of earlier rallies. Its camera, object tracker and inherited interception policy are shared across the stages. All new updates use numerical filtering or score averages without backpropagation.

The three individual milestones and the pooled four-state course passed. The final learner won 56/96 scorable points after an own return versus 7/96 for center aim. In the toughest combined state it saved 86/95 later balls versus 94/101 centered.

![Comparisons for each skill and for continuous play](assets/2026-11-19-pong-transfer-progression.png)

## Recovering controls during a rally

On 30 fresh matched starts covering five hidden button permutations and three switch times, the learner returned 27 first balls versus 12 for the shorter calibration and 13 for the frozen motor map. It returned 95/95 later balls. On ordinary controls its requests and observations matched the reference throughout. In two of the rescues, replaying the same start with partial identification disabled lost the ball. An independent continuous-gain learner also recovered 20/24 first balls after unequal actuator-strength changes versus 9/24 with shorter calibration. That learner was evaluated as a separate architecture. [Primary comparison](2026-11-19-rapid-recovery-qualification.md) · [Continuous actuator effects](2026-11-19-continuous-motor-effects.md)

## Learning when to move

A bank of visual movement models estimates the lag between a request and its visible effect. The controller projects motion already in the command queue when deciding when to stop. The first version did well with newly introduced lag but held the old estimate too long when normal controls returned. The revision uses visible motion earlier than the old model predicts to reconsider that estimate.

On 16 new three-point streams with hidden delays of 240 or 420 ms, the revised learner returned 21/32 first balls and 71/75 later balls. A controller with timing updates frozen returned 6/32 and 3/8. Among the 16 carried-delay openings, five were outside physical reach even under immediate maximum drive once the queue emptied. The learner saved all 11 reachable openings. After ordinary timing returned, it saved 16/16 openings and 16/16 first balls after another hidden remap. Within the 15 mid-rally games where it identified the exact delay, median identification took 1.03 seconds of game time. [Independent qualification and reach analysis](2026-11-19-timing-recovery-qualification.md)

## Learning shot placement from score

The learner practices on a proportional tracking opponent capped at 0.35 arena units per second. It can choose between centering the paddle and six signed offsets. These are available actions. Their values are estimated from terminal scores. The model shares evidence between visually similar arrival heights. No chosen-angle label, opponent miss label, or backpropagation enters the score update. A faster 0.65 opponent tests return reliability. Earlier native trajectory searches explain why these are separate benchmarks. [Practice design and opponent feasibility](2026-11-19-pong-score-placement-protocol.md)

The original scoring policy practiced for 96 chronological points. It made 141/143 returns and won 53 points after an own return, and 18 serves scored before the paddle hit the ball. Its 78 score updates excluded all 18 uncontested serves. On 32 independently prepared starts, the frozen learned values won 24 points after an own return, versus three for center and 14 for untrained fixed exploration. On the stronger opponent it returned 19/20 first and 30/33 later balls, below the 95% later-return target. [Original score qualification](2026-11-19-score-placement-qualification.md)

The following replay shows two frozen policies on the same serve against the same 0.35 opponent. It illustrates a scoring gain. The learned policy hits once and scores at 3.57 seconds. The center policy hits three times and reaches the 12-second cap. The replay reproduced the recorded action hashes for both policies.

![Matched Pong replay of center aim and learned shot placement](assets/2026-11-19-pong-shot-comparison.gif)

The original checkpoint also played 12 fresh paired starts at each of three untrained opponent speed limits. At 0.33, 0.37 and 0.40 it won 8, 5 and 2 points after an own return, compared with 1, 0 and 0 for center. Both returned every incoming ball on these starts. The opponent sent back 0/8, 9/14 and 18/20 of the trained shots, respectively. The two gains at 0.40 are too few to establish a reliable advantage there.

## Continuous play across states

Each arm plays four successive blocks. The learner carries its numerical motor-order probabilities, selected delay and shot values across points. The arena resets the ball, camera history and unexecuted commands for each serve. It alone knows which opponent, button map and queue are active. The center comparison keeps motor and timing adaptation live.

On the original 12-point-per-block course, the scorer won 10, 9 and 10 points after an own return in ordinary play, after a hidden remap plus delay, and on returning to ordinary play. Center won 2, 0 and 2. On the intervening strong-opponent block, both saved 7/12 openings, then the scorer returned 8/14 later balls versus 34/35 with center. This exposed a failure that separate scoring and delay tests could miss. [Mixed-course runner](../../scripts/experimental/run_pong_mixed_course.py)

The safer revision reads the visible opponent's pursuit speed and the stability of the current crossing forecast before committing to a learned offset. The score table is byte-identical to the original 96-point endpoint. The same safety rule applies to trained selection, center and fixed exploration. It is an engineered rule for when to trust a shot. The learned part is which offset to choose.

On another 32 independently selected starts, the revision won 18 after an own return. Center won zero and fixed exploration won nine. Five points in each arm were uncontested serve wins and stayed in the 32-point denominator. Against the strong opponent it saved 32/32 first and 65/66 later balls, versus 32/32 and 76/76 centered. This met all scoring and retention criteria. On the earlier 0.33, 0.37 and 0.40 opponents the revision reproduced all 72 original trained and center action traces. [Independent confirmation](2026-11-19-score-placement-qualification.md) · [Safer placement mechanism](2026-11-19-pong-risk-aware-placement.md)

On a separate continuous course of 16 new points per phase, the same revision won 9, 5 and 11 points after an own return in the three scorable phases. It returned 36/44 later balls during the fast-opponent block with delay and remapping still active. Center returned 67/69. The visual safety rule reset its estimate of opponent pursuit between points. Replays also found that successfully reaching the intended edge shot could create a difficult ball to return on the next exchange.

## Final continuous controller

The final controller retains an estimate of how fast the opposing paddle has visibly pursued the ball. It also remembers whether attempted shots under an inferred delay recently produced points. As their estimated value falls, it returns toward center and periodically retries. The shot-value update receives the terminal game score. The pursuit estimate comes from the camera. Neither mechanism receives the game phase, intervention, opponent speed, or a score label for an individual contact. The inherited controller still accepts the arena's ordinary hit notification.

A development comparison illustrates the tradeoff. Requiring positive visual evidence of weak opponent coverage before every delayed attack gave the best defense on that sample, 43/43 later balls against the faster opponent. On the slower opponent it won just 1/12 points after a return, versus 8/12 with the point-local safety rule. Two other development variants retained either visual pursuit or recent delayed-shot value across points. Their evidence motivated carrying both quantities into fresh games. [Isolated pursuit comparison](2026-11-19-pong-carried-pursuit.md) · [Portable game receipts, comparison code and independent audit](../../artifacts/m1b/2026-11-19-pong-mixed-courses.zip)

Each fresh course ran for 64 points per arm. The scorer started with the 96-point practice checkpoint. Both arms kept updating motor and timing estimates. The scoring arm continued to update shot values and safety evidence between returns and between points. The independent audit verified all 63 state boundaries per arm per course and all 128 paired first-contact clocks. The two courses used 155,952 camera transitions across 256 played points including controls.

| State, in chronological order | Wins after an own return, learned / center | Opening returns, learned / center | Later returns, learned / center |
|---|---:|---:|---:|
| Ordinary, scorable opponent | 21/32 vs 2/32 | 28/28 vs 27/28 | 34/36 vs 116/117 |
| Same opponent, hidden remap and delay | 11/32 vs 2/32 | 22/28 vs 23/28 | 45/50 vs 97/100 |
| Faster opponent, remap and delay persist | 0/32 vs 0/32 | 24/32 vs 24/32 | 86/95 vs 94/101 |
| Original opponent and controls restored | 24/32 vs 3/32 | 31/31 vs 31/31 | 27/29 vs 141/141 |

All prespecified pooled conditions passed. Across the three scorable phases the learner won 56/96 points after an own return versus 7/96 centered. Nine natural serve wins per arm stayed in the denominator and earned zero credit. Earlier scoring ends a rally, so later-return opportunities differ across arms. In the difficult phase the learned later-return rate was 90.5% versus 93.1% centered. Course A alone returned 44/50, below 90%, and Course B returned 42/45. Both arms won zero against this fast opponent. The controller can still lose the first incoming ball while it adapts or while the physical command queue prevents it from getting there in time.

## Transfer to Breakout

The first proposed milestone is to discover the controlled object and motion axis in rotated Pong with static distractors. The current visual binding expects three visible objects, and the shot guard uses the remaining object as the opponent. The inherited motor and arrival knowledge is a candidate for transfer. Role binding and opponent-specific aiming are the assumptions most likely to break.

Then start the saved controller and a fresh controller on the same unseen Breakout serves. Compare the first five incoming balls, frames to sustained returns, and the effects of carrying motor/arrival knowledge alone versus also carrying Pong shot values. Finally, test brick scoring and new layouts, then alternate Pong and Breakout without a game label to measure retention. The two checkpoints below precede any Breakout play. [Transfer plan](../plans/2026-11-19-pong-skill-progression.md) · [Loading both checkpoints](../../artifacts/m1b/README.md)
