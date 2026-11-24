# Learning on continuously random table-tennis returns

The table-clamped SO-101 sustained 32 of its own legal returns in one simulated point using a camera-driven direct-joint controller with a calibrated physics model. A different controller reached 31 on another release. The target is 100 arm returns within one uninterrupted game. Each opposing reception draws four new continuous placement and speed targets. The far paddle reads the incoming physical ball and has to intercept it, redirect it over the net and land it on the table. Its return stream is bounded by the arm benchmark's reach range. The arm sees calibrated outside stereo images, joint encoders and its own commands. Reward comes from the actual game's referee. [Opponent protocol](../tt37/opponent.md) · [32-return command tape](direct-32-tape.json.gz) · [Independent 31-return replay](integrated-31.json)

## What carried between games

The same two 24-opening courses were played with several controllers. The totals count the arm's legal returns after a scored opposing delivery. A longer volley creates additional opportunities. The rows within each column have matched initial conditions and far-target seeds. Action choices change the ball that actually reaches the opponent.

| Camera and encoder policy | First course, arm returns / delivered | Replication, arm returns / delivered | Replication's longest point |
| --- | ---: | ---: | ---: |
| Geometric controller, empty motor and outcome memories frozen | 43 / 63 | 24 / 47 | 4 |
| Same controller, both memories learn from empty | 68 / 89 | 73 / 96 | 13 |
| Carry the first course's learned memories and keep updating | — | 105 / 126 | 14 |
| Learned early preparation, longer-horizon reward and late joint corrections together | 121 / 143 | 134 / 156 | 31 |

The replication cohort had scores of five or more in 11 games with the carried motor and outcome memories. The combined controller achieved ten or more in four of 24 games. Its 31-return game had 63 alternating legal returns, one launch and 51.8 seconds of physical play. The simulator replayed every referee event and checked 24,852 CAD configurations without interpenetration. The combined controller uses a geometric stroke generator. It trained on earlier demonstrated and scored movements. The comparison of whole configurations does not isolate each component. [Replication and survival curves](integrated-repeat.json) · [First course](integrated-school.json) · [31-return audit](integrated-31.json)

## Direct joint learning and its cost

On the same replication course the planner-free policy, which issues a joint reference on each control tick, made 47 arm returns without the added action-value learner. A frozen action-value readout made 38. Updating it from observed movements and actual successes and misses raised that to 69, with 15 in the longest point. All three took about 0.95 CPU seconds per simulated second, including the arena and the stereo images. On a third group of twelve disjoint releases, carrying the first 24 games' learned commands produced 15 returns versus 11 for resetting that memory while keeping the same pretrained model. Seven came from one game in the carried condition. [Planner-free comparison](direct-action-comparison.json) · [Carried policy](direct-third-carry.json.gz) · [Matched reset](direct-third-fresh.json.gz)

The 32-return controller independently evaluates short sequences of joint references in a separate model calibrated to the robot and court. On twelve games its revised reward-credit window produced 81 arm returns versus 78 with the original window. On the selected longest release it made 32, versus 26 with within-point memory frozen and 13 with the original live update. The matched control using the old visual reversal time with the revised interval and sampling density also made 32 on this point. Better timing alone cannot explain that rally. The earlier camera method could assign credit roughly 73 ms late, which matters when interpreting the learned movement, even on points for which it did not change the score. [Direct search](../tt38/joint-mpc.md) · [Camera timing audit](../tt37/camera-time-school-audit.json) · [Matched late-clock control](credit-observed.json.gz)

## What ends the rally

We compared realized incoming trajectories after the game. The height bands use the ball's measured state at one plane, projected under gravity to the same second plane for every shot. These evaluator measurements never enter the learner. The direct-preparation controller touched 14 of the 15 lowest incoming balls on the first course and landed two. It landed every ball in the 1.04–1.09 m band. The combined controller still loses most low arrivals even though it converts almost all higher balls.

| Projected arrival height | Direct preparation, first course | Combined controller, first course | Combined controller, replication |
| --- | ---: | ---: | ---: |
| Below 0.99 m | 2 / 15 | 6 / 18 | 5 / 16 |
| 0.99–1.04 m | 27 / 35 | 42 / 47 | 41 / 51 |
| 1.04–1.09 m | 50 / 50 | 47 / 49 | 55 / 55 |
| At least 1.09 m | 25 / 26 | 26 / 29 | 33 / 34 |

The first priority is turning the already frequent low contacts into legal, answerable shots. Across 839 preceding arm shots from 58 training releases, the private predictions of the preceding outgoing shot barely predicted the next arrival height. Prediction error dropped from 50.7 mm for a constant baseline to 49.9 mm. There was no monotonic separation of low-ball risk across the forecast quartiles. Continue learning how to handle the low arrivals themselves. [Measured opportunities and touches](incoming-height.json) · [Forecast check](next-ball.json)
