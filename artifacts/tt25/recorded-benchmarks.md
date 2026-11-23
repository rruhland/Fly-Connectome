# How closely does the simulator reproduce real table tennis?

We compared the existing simulator with ball measurements from real human–robot matches. We also replayed the SO-101's actual camera-policy games and checked whether its moving links could have made those motions above a solid table.

## Flight on unseen matches

The Sony Ace release includes positions, velocities, spins and timestamps at racket, table and net events. We fitted two free-flight coefficients on 794 non-service shots from matches against five elite players. All 417 eligible shots from matches against the two professional players were reserved for testing. The 417 shots come from 134 rallies. Repeated shots within a rally are correlated. The release contains estimated physical states and event timestamps. The evaluator sees these states at launch. The arm's policy receives its usual camera observations and encoders. [Dataset and coordinate conventions](https://github.com/SonyResearch/ace_public/blob/57ac16a42c25c47bde24cd2a592804f4b800085b/data/Readme.md)

| Prediction on the reserved players | Gravity only | Previous flight law | Match-calibrated flight law |
|---|---:|---:|---:|
| Median distance from actual first bounce | 105.1 cm | 38.5 cm | 16.2 cm |
| 90th percentile distance | 323.1 cm | 84.1 cm | 34.5 cm |
| Median absolute error in first-bounce time | 90 ms | 74 ms | 24 ms |
| Median 3-D error at the recorded bounce time | 54.0 cm | 21.1 cm | 8.7 cm |

The first three rows predict the downward crossing of a plane one ball radius above table height. They include shots whose predicted landing is outside the table. The last row tests position when supplied with the recorded time. Both held players individually reached a 16.2 cm median landing error. The two fitted coefficients are air drag and Magnus acceleration. The fit uses numerical least squares on development flights and requires no learned motor action or backpropagation. The simulator still has errors comparable to the width of a blade. [Scores, splits and coefficients](recorded-flight.json)

We also initialized a physical MuJoCo ball once per real shot and allowed net and table collisions. We checked where it was when the next player hit the real ball. On 383 measured shot–bounce–reply sequences the median error fell from 32.1 cm to 19.6 cm, including simulations that missed the tabletop. The new model contacted the table on 340/383 trajectories versus 282/383 for the previous one. It contacted the net on 56 trajectories versus 51. This test used the existing table material and omitted arms to isolate ball travel. A separate regression check on the original 37 SpinDOE short-flight trajectories moved the median error from 3.63 cm to 4.46 cm. Use the new calibration as a match profile while retaining the legacy benchmark. [MuJoCo and SpinDOE comparison](recorded-exchange.json) · [SpinDOE](https://github.com/cogsys-tuebingen/spindoe)

## What a real practice feed looks like

| Outgoing, after the serve | Shots | Median speed | 10th–90th percentile speed | Median spin |
|---|---:|---:|---:|---:|
| Held SO-101 compact games, both rackets | 37 | 3.84 m/s | 2.34–4.35 m/s | 59 rad/s |
| DeepMind real rally states | 13,088 | 5.50 m/s | 4.22–7.18 m/s | 54 rad/s |
| Sony matches against the two professional players, both rackets | 541 | 9.49 m/s | 5.36–17.96 m/s | 267 rad/s |

The compact policy's median spin was close to the DeepMind reference. Its median shot speed was below even the 10th percentile there. Between alternating racket hits, the compact games had a median time of 0.626 s and the Sony matches 0.541 s. Similar timing on a shorter court can hide substantial differences in shot speed and reach. These are distributions of shots, weighted by how many were recorded in each rally. Neither external collection represents the full range of recreational play. The DeepMind dataset contains initial states for physical shots and does not give subsequent landings. [DeepMind data, CC BY 4.0](https://github.com/google-deepmind/competitive_robot_table_tennis/tree/54cf18e41a8f134d98ad38710001ed50aabb8ebc) · [Reference statistics](recorded-secondary-reference.json) · [Comparison figure](recorded-comparison.png)

## Making the body physically accountable

The stock SO-101 import kept its CAD meshes visible but removed their collision copies. We replayed the exact commands from the 16 prior compact points with the inspected motor maps. Moving link meshes intersected the tabletop by more than 5 mm in 10 points and 53 of 1,026 control ticks. The deepest overlap was 5.59 cm at the gripper. We enabled contact with the existing convex link meshes on the opt-in court. The prior simulator and policy remain available for paired tests. With identical motor tapes, all 16 corrected points terminated, all six previously successful starts still produced a legal return, and the total changed from 11 to 9. Worst sampled penetration was 2.5 mm under compliant contact, with zero ticks deeper than 5 mm. Queries were made at control ticks, not every solver substep. They are geometric checks on the simulated CAD shapes. The audit does not calibrate servo latency, grip slip or on-hardware contact. [Original tapes and clearance](recorded-arm-audit.json) · [Corrected clearance and outcomes](recorded-arm-solid.json) · [Table contact filtering](../../scripts/experimental/tt25_body_contacts.py)

In a separate closed-loop check the same camera-driven player and active opponent played all 16 starts on the corrected court. The near arm made nine legal returns over six starts, and the opponent also made nine. The longest point contained six alternating returns. All 16 games replayed exactly. The replay video shows the longest point at its original speed. [Watch the compact point](recorded-arm-live.mp4) · [Full live actions and referee events](recorded-arm-live.json.gz)

## Teaching choices

1. **Change practice difficulty, keep the test fixed.** Begin with measured feeds a fixed base can physically reach. Keep gravity, contacts, spin and the referee active. Widen placement, pace and spin as the learner makes legal returns. Count repeated hits, uncoached first returns, and how many target balls it took. Publish both the reachable-lane score and the complete feed distribution. Returning balls into the other player's reach is good cooperative practice. Match wins can be scored separately.
2. **Use denser reward as temporary feedback.** A legal landing or continued rally gives immediate feedback for practice. A camera-derived estimate of distance or contact can be compared against sparse point reward under equal practice. A referee-supplied contact label is a teaching aid. Freeze any shaping at evaluation, and publish results from previously unseen points under the common referee. Reward for merely surviving could reward stalling or a motionless arm.
3. **Use recordings for what they measure.** Human match video plus estimated poses provides evidence for timing, paddle motion and ball outcomes. Joint angles from a human skeleton do not specify the actuator commands on an SO-101. A demonstration on the actual robot can record synchronized wrist and outside images, issued targets, measured joints and timestamps. With those data the existing non-backprop readout can try a demonstrated action in a similar observation, then keep or revise it through actual play. Split recordings by session or opponent. Compare equal robot practice with and without demonstrations and report teacher exposure. [Recording cameras and joint positions with LeRobot](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot)
4. **Test the two transfer directions separately.** Compare a native Pong checkpoint with a shuffled-source and cold control using the same arm model, same target demonstrations and same feeds. Then give the pre- and post-table-tennis checkpoints the same short wrist-only solo-bounce course. Reward quicker first successes and retained long streaks on held releases. A better motor map can help both games, but transferring when the motor map is already equal would support a stronger task-level claim.

## Reproduction

Run from the repository root in its MuJoCo environment. Retrieve the pinned Sony CSV, the DeepMind rallies JSON and the SpinDOE trajectories archive from the source repositories above. The analysis verifies file digests. These files are not redistributed here.

```bash
python -m scripts.experimental.tt25_recorded_flight /path/to/match_data.csv
python -m scripts.experimental.tt25_recorded_exchange /path/to/match_data.csv /path/to/trajectories.zip
python -m scripts.experimental.tt25_recorded_figure /path/to/match_data.csv /path/to/rallies.json
python -m scripts.experimental.tt25_arm_replay
python -m scripts.experimental.tt25_solid_play
python -m scripts.experimental.tt25_solid_play --movie
```
