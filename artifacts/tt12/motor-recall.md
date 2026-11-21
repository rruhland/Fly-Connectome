# Recovering a learned motor map after rewiring

On the gantry that lost 0–11 in TT11, the arm used two 34-frame RGB motor probes to identify how its new motor ports corresponded to previously learned arm models. Retaining the registered serving policy and changing the calibrated motor map raised legal serves from **1/16 to 14/16** on new throws. The arm won eight of those 16 service points. Two full matches told a different story: it made more legal returns and sustained more rallies, but scored five points in total, the same as the usual calibration. [Side-by-side replay of the first two new tosses](gantry-motor-recall.mp4) · [Usual calibration](gantry-serves.json.gz) · [Two-probe transfer](gantry-recalled-serves.json.gz)

## What fixed the serve

We first replayed the full 11-point history that included the five failed serves. The standard control reproduced every scorer result and every command to within 8.3 × 10⁻¹⁴. We then tested each version on 16 new own-service points, with four each under ordinary, softened, delayed and mixed conditions. The controls saw the same tosses and disturbance streams. Their internal learning continued between points. Every played tape reproduced its scored physical result. This was a service test with an opponent that played out every legal point.

| Calibration or controller change | Original tosses, legal / 5 | New tosses, legal / 16 | New service points won / 16 |
|---|---:|---:|---:|
| Usual calibration and registered serve | 0 | 1 | 0 |
| Same calibration, fixed forward drive | 0 | 1 | 1 |
| Same calibration, stroke from the earlier 11–6 gantry win | 0 | 0 | 0 |
| Same calibration, constrain all future commands in the stroke plan | 0 | 1 | 0 |
| Change only the visual tracker reference | 0 | 1 | 0 |
| Change only the learned actuator timing | 0 | 1 | 0 |
| Change only the learned spatial motor map | 2 | 12 | 3 |
| Change geometry and timing; retain original optics | 4 | 14 | 8 |
| Two-probe calibration, including its own optics | 4 | 14 | 8 |

Here geometry denotes the coefficients that predict the blade's position and inclination from effective motor commands. It includes linear and quadratic terms. The geometry-only comparison used the original tracker, timing, and serving policy. The full-horizon constraint comparison corrected an approximation in the original planner, which solves for a series of future commands and clips only the first. That change produced the same number of legal serves. [Geometry](gantry-geometry-serves.json.gz) · [Timing](gantry-timing-serves.json.gz) · [Joint change](gantry-both-serves.json.gz) · [Optics](gantry-optics-serves.json.gz) · [Constrained planning](gantry-feasible-serves.json.gz)

## What the arm learned from the probes

The TT9 donor already had motor maps from earlier installations. For each stored map with the observed number of motor ports, the new player matched old and new response directions, then tried signed port swaps against measured stereo motion. It chose by prediction error on the two probes. The learner received commands and RGB sightings of its blade. The evaluator retained the actual wiring, contacts and joint state. Both the transferred player and the fresh two-probe fit built their visual reference from the same first two episodes. This procedure assumes that some previously learned model is useful up to a change in port signs and order. It does not discover an arbitrary new mechanism from two episodes. [Motor transformation](../../scripts/experimental/tt12_motor_recall.py) · [Probe and checkpoint construction](../../scripts/experimental/tt12_motor_probe.py)

Three further probe sequences were withheld from every fit. The reported error is a clipped pose norm, using translation in metres and a 0.12 weight on the pitch residual. The normal commissioning procedure used five probe episodes on the gantry and eight on C and D. Short calibration consumed 68 commanded frames (2.27 simulated seconds) per installation.

| Held pose error, cm | Carried and remapped | Fresh two-probe fit | Usual commissioning | Carried with original wiring assumed |
|---|---:|---:|---:|---:|
| Four-axis gantry | 3.9 | 8.4 | 5.6 | 19.8 |
| Five-axis C | 4.9 | 11.3 | 5.0 | 13.9 |
| Five-axis D | 4.5 | 6.5 | 4.3 | 19.2 |

On the common complete-game fixture, the transferred models scored 2–11, 7–11 and 9–11 on gantry, C and D, versus 0–11, 5–11 and 0–11 when fitting from the same two probes without the old maps. All three used identical time and motor allowances. The full-calibration fixed table models on that fixture scored 4–11, 11–8 and 10–12. On a second gantry fixture, recall lost 3–11, full calibration lost 1–11 and the fresh short fit lost 0–11. Across the two gantry games, motor recall made 21 legal returns and sustained six rallies versus 11 and two for full calibration. All three started afresh for the second fixture and kept learning within the game. [Gantry probe evidence](motor-gantry-probe.json) · [C probe evidence](motor-c-probe.json) · [D probe evidence](motor-d-probe.json) · [First gantry game](motor-gantry-recall-game.json.gz) · [Second gantry game](gantry-recall-replication.json.gz)

## Does match experience improve the next game?

The gantry first played the 2–11 loss above. We preserved that player's ordinary online updates and also tried learning a further correction to its motor map from its own stereo blade tracks and issued commands. No joint positions or true delays entered the fit. For each played point the estimator selected a command delay using its RGB prediction residual. A robust ridge fit adjusted only translation coefficients. It trained on the first two thirds of the points and selected the correction strength by optical error on the remaining points before updating from all practice points. The next fixtures were disjoint from the practice game. [Estimator](../../scripts/experimental/tt12_match_fit.py) · [Gantry fit](playfit-gantry-fit.json) · [Five-axis C](playfit-c-fit.json) · [Five-axis D](playfit-d-fit.json)

| Gantry start | Next game | Legal returns / sustained rallies | Independent repeat | Legal returns / sustained rallies |
|---|---:|---:|---:|---:|
| Original two-probe checkpoint, previous game discarded | 11–5 | 31 / 7 | 7–11 | 18 / 5 |
| Carry ordinary updates from the previous game | 11–6 | 26 / 8 | 11–8 | 24 / 6 |
| Carry those updates and re-fit the motor map | 11–9 | 15 / 8 | 12–10 | 27 / 11 |

Keeping the ordinary match experience changed the repeat from a loss to a win and gave two wins in two fixtures. The reset player also won the first fixture, with more legal returns. The additional RGB correction increased sustained rallies in the repeat but reduced first-fixture returns. On C it changed 7–11 to 8–11 (10 to 13 returns); on D it changed 8–11 to 7–11 (17 to 10 returns). All three corrections improved held practice prediction, so that metric alone cannot select a consistently better match player. The gantry checkpoint following the 11–9 win preserved all 180 native Pong action tapes and 41 points. [Gantry carried fixture](playfit-gantry-base-game.json.gz) · [Gantry carried repeat](playfit-gantry-base-repeat.json.gz) · [Reset repeat](playfit-gantry-reset-repeat.json.gz) · [Updated repeat with 27 returns](playfit-gantry-updated-repeat.json.gz) · [Winning checkpoint](playfit-gantry-player.json.gz) · [Pong retention](motor-gantry-pong.json.gz)

## Transfer to a new task

We moved the newly commissioned maps into a 3D tracking game. It rewards time within 4.5 cm of a moving goal. The optical tracking learner can update its own geometry throughout play. Each installation saw eight new paths, once in familiar axes and once with randomly rotated views. The target-only player explored in the game. The table-trained players could use their priors from the first command. Across 9,216 scored frames each, the transferred, two-probe fitted, normally commissioned and target-only learners spent 4,755, 4,724, 4,772 and 3,619 frames on target, respectively. The two short table calibrations helped relative to starting the new game empty. Recovering the old port map gave little extra time on target over learning a new map from the same probes. Each action tape replayed in the physical arm simulation. [Gantry](tracking-gantry.json.gz) · [C](tracking-c.json.gz) · [D](tracking-d.json.gz)

## Reproduction

Run in the hall environment, from the repository root. The first command reconstructs the original gantry start. The second also creates the fresh two-probe control beside its transferred checkpoint. For C and D substitute the corresponding TT11 inherited starts. Use game seeds 182395613 and 24810795 for the main and repeat gantry fixtures.

```sh
python -m scripts.experimental.tt10_series_trial /tmp/gantry.json.gz --prepare --body gantry --installation-seed 780847903 --calibration-seed 1500752929 --probe-cache /tmp/tt12-cache
python -m scripts.experimental.tt12_motor_probe /tmp/gantry.json.gz artifacts/tt9/precontact-match-b-player.json.gz /tmp/probe.json /tmp/recalled.json.gz --cache /tmp/tt12-cache
python -m scripts.experimental.tt12_gantry_trial /tmp/gantry.json.gz /tmp/control.json.gz --historical-control
python -m scripts.experimental.tt12_gantry_trial /tmp/recalled.json.gz /tmp/new.json.gz --modes registered
python -m scripts.experimental.tt12_motor_movie artifacts/tt12 /tmp/first-two-tosses.mp4
python -m scripts.experimental.tt12_match_fit artifacts/tt12/motor-gantry-recall-player.json.gz artifacts/tt12/motor-gantry-recall-game.json.gz /tmp/match-fit.json.gz /tmp/match-fit.json
python -m scripts.experimental.tt10_series_trial /tmp/updated-repeat.json.gz --resume /tmp/match-fit.json.gz --seed 134701923 --games 1
```
