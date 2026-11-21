# From table tennis to precision reaching

The arm had to hold its tip within 4.5 cm of eight successive targets, for two consecutive simulation frames per target. Each session lasted 6.4 simulated seconds. On familiar motor wiring, the map learned in table tennis let it reach the very first target in 36 of 42 sessions. Starting with an empty map and the same update and exploration rules yielded 4 of 42. The identical wiring and physical body carried over, while the ball, opponent, scoring rule and action-selection policy changed. [First course](target-transfer.json.gz) · [Independent repeat](target-confirm.json.gz)

| Starting motor map and cursor procedure | Familiar wiring, targets / 336 | First targets / 42 | Secretly rewired, targets / 336 |
|---|---:|---:|---:|
| Table tennis, adaptive with evidence-triggered probes | 297 | 36 | 209 |
| Empty, identical adaptation and probes | 180 | 4 | 199 |
| Table tennis, frozen map with evidence-triggered probes | 284 | 34 | 1 |
| Empty, scheduled exploration | 220 | 2 | 206 |
| Empty, probes first | 209 | 2 | 222 |
| Sign-reversed table map, evidence-triggered probes | 194 | 2 | 203 |

The main transfer effect repeated on both courses (168/192 and 129/144). With familiar wiring the adaptive player reached 111/112 and 107/112 targets on the two five-axis installations, and 79/112 on the six-joint arm. The gap between 284 and 297 measures the additional benefit of updating the carried map in the new game. A secret change to the motor connections erased the overall advantage over the strongest target-only method.

## What the learner received

The cursor game runs the original physical MuJoCo joints and servos. Two calibrated orthographic RGB displays show the arm tip and goal in XY and XZ. Marker identities, shared metric axes, the reaching objective, and the controller that moves toward the goal were supplied. The transferred quantity was the local relationship between motor commands and 3D tip displacement, estimated previously by visual probes in table tennis. Online least squares updates use requested commands and observed marker motion. No backpropagation was used. Wiring, target-generating actions and joint coordinates stayed in the environment. The first two targets used immediate actions, then the simulator introduced noise and two frames of delay; the delay later fell to one frame. Each option received the same target sequence, motor opportunities and disturbances. All 756 session action tapes reproduced their physical scores. [Environment and learner](../../scripts/experimental/tt10_target.py) · [Evaluation](../../scripts/experimental/tt10_target_trial.py)

## Reconnecting the motor channels during play

In separate 16-target sessions, the arm played four targets with the original wiring, six after an unannounced change, and six after reconnection. The arm, pending commands and learner continued through both switches. On an initial set, ordinary online updates reached 58/64 before the switches, 11/96 during the change and 8/96 upon reconnection. Restoring a cursor learner checkpoint halfway through each of two example sessions reproduced all subsequent commands, outcomes and final state. [Five-axis](target-switch-offset.json.gz) · [Six-joint](target-switch-rotary.json.gz) · [Checkpoint](target-switch-offset-player.json.gz)

On 20 fresh sessions, a learner re-probed each motor channel when the observed motion repeatedly disagreed with its forecast. Each probe consumed ordinary playing time.

| Installation and method | Original, / 40 | Rewired, / 60 | Original restored, / 60 |
|---|---:|---:|---:|
| Five-axis B, ordinary updates | 39 | 5 | 4 |
| Five-axis B, re-probe | 39 | 20 | 26 |
| Six-joint, ordinary updates | 26 | 5 | 9 |
| Six-joint, re-probe | 26 | 17 | 7 |

The benefit came from learning the current directions again. Adding a bank of past maps produced identical scores and never selected a stored map. An independent test of estimating a signed channel permutation earned 126/192 on initially rewired arms versus 119/192 with the same exploration and an empty map. On familiar wiring this method scored 128 versus 167 for direct transfer. The established result is fast reuse when the controls persist. Automatic recognition after a switch remains open. [Five-axis re-probing](target-reidentify-offset.json.gz) · [Six-joint re-probing](target-reidentify-rotary.json.gz) · [Permutation study](target-remap.json.gz)

## Reproduction

Run from the repository root in the MuJoCo environment. The second trial uses the same interfaces on independently seeded sessions. The verifier physically replays all 128 full mid-session wiring trials from saved commands.

```bash
python -m scripts.experimental.tt10_target_trial /tmp/course.json.gz --seed 93825711 --sessions 8
python -m scripts.experimental.tt10_target_trial /tmp/confirm.json.gz --seed 420173597 --sessions 6
python -m scripts.experimental.tt10_target_verify artifacts/tt10/target-{switch,reidentify}-{offset,rotary}.json.gz
```
