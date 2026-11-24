# What the SO-101 sees and why it can stay still

On eight continuously varied points, the physical opponent delivered 14 balls that crossed the line 1.10 m on the near side. Replaying the original stereo cameras and commands gave the first incoming crossing forecast a median 124 ms after the opponent’s hit. The forecast used the observed orange ball and the nominal flight model. The opponent’s requests and physical crossing were available only to the evaluator. Replays matched the original referee events. [Per-ball forecasts](tracking.json) · [Sensor replay](../../scripts/experimental/tt32_tracking.py)

| Time before crossing | Incoming balls with a forecast | Median lateral error | Median error predicting the center |
| --- | ---: | ---: | ---: |
| 400 ms | 11/14 | 7.6 mm | 151 mm |
| 300 ms | 13/14 | 4.4 mm | 150 mm |
| 200 ms | 14/14 | 1.7 mm | 126 mm |

On four completely stationary approaches the arm’s planner started trying 77–83 ms after the opponent’s hit. It rejected all 29 attempts. The rejection happened at different stages, including fitting a feasible posture and forecasting a legal return under the motor and collision limits. Extending the possible contact window to include deeper strikes raised the arm from 6 to 8 legal returns on the original eight games and from 4 to 6 on eight new games. Three previously stationary openings yielded actual returns. Three centrally arriving balls still elicited no useful movement. Using configurations from rewarded trajectories as alternative starting guesses generated 2,049 extra candidate movements across the original eight games. None passed verification. These results identify specific limits in the search and in the arm’s coverage. [Idle traces and deeper contacts](idle.md) · [Alternative joint configurations](body-recall.json.gz) · [New course before consulting a learner](fresh-plan.json.gz) · [Fixed controller on that course](fresh-fixed.json.gz)

## What the opponent shows before impact

A separate test asked whether the far paddle's visible motion predicted the shot sooner than observing the incoming ball. A ridge regressor trained on four measured launches with different opponent seeds. Its target was the later crossing estimated from stereo images. Four other launches provided the held examples. The observer captured two snapshots when the outgoing ball passed two fixed depths. On the 31 usable test examples at each depth, all features preceded the far hit. Training and assessment used continuously varied opponent requests. Of 96 episodes, 90 had both a physical and an optical crossing label. Their median lateral difference was 0.31 mm. [Optical observations](opponent-frames.json.gz) · [Pre-impact scores](opponent-results.json) · [Method](../../scripts/experimental/tt32_opponent.py)

| Inputs | Mean absolute error 256 ms before far impact | Mean absolute error 156 ms before far impact |
| --- | ---: | ---: |
| Mean training crossing | 12.9 cm | 12.9 cm |
| Outgoing ball position and velocity | 12.4 cm | 15.6 cm |
| Outgoing ball plus far paddle silhouette | 17.8 cm | 21.4 cm |

This small, calibrated silhouette model generalized poorly before impact. It did not drive the arm. The earlier robot table-tennis literature also tests opponent-racket motion for anticipation. Our rail-mounted paddle, image resolution and snapshot features differ from human stroke video. The post-impact stereo result above supports acting from the observed return as soon as possible. [Learning anticipation policies for robot table tennis](https://ieeexplore.ieee.org/document/6094892/)

## Preparing the next movement

A learner compared three recovery distances on 34 actual following-ball outcomes and chose full recovery on both sides. Smaller movements helped one practice game but did not displace full recovery in either held comparison. A camera-driven adjustment after the opponent’s hit took 6 arm returns to 8 on the first eight games and tied at 4 on the fresh eight. Waiting three ticks without moving matched the one improved point. This identifies fresher camera information as sufficient for the improvement on that ball. [Outcome-based recovery](learn.md) · [Movement and delay controls](anticipate.md)
