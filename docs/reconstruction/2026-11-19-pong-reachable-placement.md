# Public camera forecasts for reachable Pong placements

The scorer can aim a shot by reaching an off-center paddle position when the ball arrives. The observer estimates this position from public camera tracks and the carried motor model. It rolls requests already in the queue forward, then evaluates the positions reached by holding each extreme movement command for the remaining interval. Camera prediction disagreement narrows the eligible offsets. The controller retains the preceding timed interception policy, which turns the selected contact target into actual motor requests.

Eight separate development recordings supplied timed public tracks for the ball and both paddles. Eight visually resolved contacts and 679 unobstructed flight windows gave a fitted ball speed of 0.552 image units per second, a ball-contact plane at −0.4372 and a dependence of outgoing angle on offset near 9.43 radians per normalized image unit. The observed paddle center was at −0.4531. On a steep approach that plane difference can shift the required contact height by about 0.02, comparable to the margin around a sharp shot. The learner received public coordinates and elapsed camera intervals. It used a symmetric opposing contact plane and a bounded pursuit model fitted from 1,952 opponent motion windows.

## Predicting opponent clearance

The camera-only planner rolls ball flight and candidate opponent pursuit forward from the observed scene. It chooses the feasible offset with the largest predicted gap between ball and opponent at crossing. This flight and pursuit fit uses camera trajectories alone. The comparison used the same fitted contact plane and actuator state in both arms. The centered arm aimed directly at the fitted crossing.

| Opponent speed | Own-hit wins, aimed / centered | First returns, aimed / centered | All returns, aimed / centered |
|---|---:|---:|---:|
| .35 | 6/8 · 1/8 | 6/6 · 6/6 | 9/9 · 21/21 |
| .65 | 0/8 · 0/8 | 8/8 · 8/8 | 23/24 · 28/28 |

Two games per .35 arm ended with direct serve wins. The 32-game comparison took 200 seconds on one worker and used 14,787 transition cameras. Aimed rallies ended sooner at .35, leaving fewer later opportunities.

## Using the frozen score ranking

The guarded score controller uses the same reach computation and learned contact plane with the frozen Stage 3 score ranking. It keeps the existing action repertoire and checks the placement against the newly computed feasible interval. In the centered counterpart the score state and all motor and interception behavior are shared, and the rank table is left unused. Each pair starts from the same timing checkpoint. The delayed remap starts used the center stream's actual timing state after its preceding combined remap and delay phase. Both arms continued ordinary motor and timing updates from their own camera history.

| Condition | Own-hit wins, aimed / centered | First returns, aimed / centered | Later returns, aimed / centered |
|---|---:|---:|---:|
| Ordinary .35 | 2/4 · 0/4 | 2/2 · 2/2 | 1/1 · 4/4 |
| Ordinary .65 | 0/4 · 0/4 | 3/4 · 4/4 | 3/3 · 5/5 |
| .65 with remap and 12 queued requests | 0/6 · 0/6 | 4/6 · 4/6 | 1/4 · 4/4 |

Two ordinary .35 games per arm ended with direct serve wins. The 28-game comparison took 79 seconds and used 7,570 transition cameras on one worker. All starts in both comparisons were drawn outside the prior qualification and mixed-course schedules. These are bounded development samples.

Sharp score-ranked placements cost first and later returns against the .65 opponent even after the motor reach check. A feasible terminal position still has to be reached by the timed feedback law with enough precision to make contact. The sampled defensive fallback was `GuardedScorePlacement(..., aimed=False)`, which keeps the fitted plane and targets the center. Identifying when to switch between aimed and centered play from visual evidence remains a separate test.

A separate evaluator-only replay reproduced all four selected aimed misses with identical action hashes and contact clocks. At the missed ordinary opening, the ball and paddle center differed by 0.114 image units. The three delayed-remap misses were later arrivals with gaps of 0.140, 0.217 and 0.212. In the latter two cases the preceding shot had been aimed and the missed approach had no new aim commitment. The resulting opponent return can raise the cost of the next interception, independently of whether the current shot was itself within reach. These private contact measurements served only the replay audit.

The archive includes the fit, exact ordinary and carried timing checkpoints, the frozen score state, condition schedules, action hashes and compact receipts. [Development archive](../../artifacts/m1b/2026-11-19-pong-reachable-placement.zip)
