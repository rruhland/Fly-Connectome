# Using a learned motor map to recover usable stroke travel

On a hidden cross-coupled simulated gantry, moving the travel constraint into the learned arm coordinates turned a 3–11 loss into an 11–9 win. The player then resumed from that checkpoint without calibration and lost a fresh game 10–12, with 16 legal returns. We have a working game and a transferable motor fit. Sustained winning on this installation is still open. [Watch the 67-second game](gantry-coupled-full-game.mp4) · [Winning checkpoint](motor-gantry-coupling-travel-player.json.gz) · [Continued checkpoint](resume-travel-257341089/motor-gantry-coupling-travel-player.json.gz)

Each cell gives the final score, legal returns and points with at least two legal returns. Columns used identical learned maps, opponent schedules, camera observations during calibration and startup cost. Full games have different numbers of scored points when outcomes diverge. Controller variants changed neither the camera system nor the arm physics. [Audit and source hashes](allocation-summary.json.gz)

| Altered installation | Original travel limit | Limit in learned coordinates | Native planner and rate allocation |
| --- | --- | --- | --- |
| Gantry, coupled | 3–11 / 14 / 3 | **11–9 / 22 / 6** | 3–11 / 10 / 1 |
| Gantry, weakened | 3–11 / 8 / 2 | 9–11 / 14 / 2 | **12–10 / 23 / 4** |
| Five-axis, coupled | 3–11 / 7 / 2 | 8–11 / 7 / 1 | 7–11 / 5 / 0 |
| Five-axis, weakened | 2–11 / 3 / 1 | 2–11 / 3 / 1 | 1–11 / 1 / 0 |

The two additional TT14 actuator matrices had fresh game seeds and had not been used to choose these controller changes. On repeated gantry coupling the three scores in column order were 7–11, 3–11 and 5–11, with seven, eight and six returns. On repeated five-axis attenuation they were 1–11, 0–11 and 3–11, with six, five and one return. Neither gantry win replicated across those other hardware tests. Picking the best controller after seeing each fixture would overstate what a single player can do. [Gantry repeat](motor-gantry-coupling-tt14-travel.json.gz) · [Five-axis repeat](motor-five-gain-tt14-allocate.json.gz)

The original controller clipped accumulated requests separately on each public port. After weakening, a port at the software limit could still leave substantial movement available in the fitted arm coordinates. The revised controller keeps the same predictive stroke planner and the same ±1 public rate limit. It integrates the requests through the learned response matrix and limits travel in that coordinate system. The alternative first computes a stroke with the native motor planner and uses bounded least squares to allocate feasible port rates. Both are controller design choices. The response matrix is learned from images. The fitted envelope is an approximation in this simulator. [Implementation](../../scripts/experimental/tt17_allocation.py) · [Background on constrained control allocation](https://doi.org/10.1016/j.automatica.2013.01.035)

Across the four games, changing only the travel limit improved points won from 11/55 to 30/72, legal returns from 32 to 46 and multiple-return points from eight to ten. Own-service points won rose from 2/24 to 14/35. The five-axis weakened arm still delivered zero legal serves in six attempts. Both methods spent 68 frames on camera setup and 24 on changed controls. The original controller accepted zero in-game motor updates in these fixtures. Stable installations retain the original controller by construction when the fitted map is the identity. Both original stable game tapes were verified in the motor comparison. [Calibration and independent controls](motor.md)

The continuation forked the exact 11–9 endpoint onto a fresh point schedule with the same hardware. The learned-coordinate limit made 16 returns and reached two returns on two points in a 10–12 loss. Restoring only the old travel constraint gave 12 returns and one multiple-return point in a 9–11 loss. Both began with the same stroke memories and motor fit and spent zero frames on further calibration. Selecting the original winner for this test makes this a follow-up on one installation. [Resumed game](resume-travel-257341089/motor-gantry-coupling-travel.json.gz) · [Matched control](resume-travel-257341089/motor-gantry-coupling-bounded.json.gz)

To play another game from the winning first-game endpoint, run the following in the repository's table-tennis simulation environment. The seed selects new ball trajectories and disturbance phases. Use a new seed to save a new continuation.

```sh
python -m scripts.experimental.tt17_allocation_resume artifacts/tt17 gantry-coupling 257341090 travel
```
