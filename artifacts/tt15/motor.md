# Recognizing motor changes from rally motion and repeated swings

Testing a candidate on a separate scored point reduced false switches from five to one in fresh games with unchanged controls. The arm lost both return games 3–11 and failed to recognize either restoration. Repeating a short optical movement detected four physical changes in one or two affected starts and preserved all 38 unchanged action tapes. It made 35 legal returns during changed controls, versus 27 with the earlier selector, and its game scores were mixed. [Game and decision audit](motor-summary.json.gz) · [TT14 starting evidence](../tt14/gain.md)

## Fresh games using rally evidence

The gantry received new coupling across its four motor rates. The five-axis arm received new independent rate attenuation. The change began before the fifth start. The players loaded saved checkpoints for the next game and played its first four starts with the change still active. The original controls returned before its fifth start. Fixed, TT14 selective, and cross-point branches started from the same TT13 checkpoint, ball seeds and disturbance schedule for each fixture. All four opening action tapes matched on both changed fixtures. Games used shaded first-person RGB, physical alternating service and a two-point winning margin. [Gantry corrections](motor-gantry-coupling-guard.json.gz) · [Five-axis corrections](motor-five-gain-guard.json.gz)

| Arm and map | First score | Changed starts | Points won | Legal returns | Return game | Changed starts | Points won | Legal returns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gantry cross-point](motor-gantry-coupling-guard.json.gz) | 8–11 | 15 | 4 | 5 | 3–11 | 4 | 2 | 1 |
| [Gantry TT14 selective](motor-gantry-coupling-hybrid.json.gz) | 5–11 | 12 | 1 | 4 | **11–8** | 4 | 1 | 2 |
| [Gantry fixed](motor-gantry-coupling-fixed.json.gz) | 6–11 | 13 | 2 | 0 | 9–11 | 4 | 2 | 0 |
| [Five-axis cross-point](motor-five-gain-guard.json.gz) | 5–11 | 13 | 3 | 2 | 3–11 | 4 | 1 | 1 |
| [Five-axis TT14 selective](motor-five-gain-hybrid.json.gz) | 5–11 | 12 | 3 | 5 | 5–11 | 4 | 1 | 2 |
| [Five-axis fixed](motor-five-gain-fixed.json.gz) | 3–11 | 10 | 1 | 1 | 8–11 | 4 | 0 | 1 |

The cross-point rule fit gain and coupling candidates using an earlier point and evaluated them on a different point. Both histories had to improve before the model could change. It selected a correction after the second changed start on both arms. TT14 selected a correction after the first. Across complete changed intervals, cross-point confirmation made nine legal returns in 36 starts, TT14 made 13 in 32, and fixed made two in 31. The more conservative rule reduced false changes at the cost of fewer returns and slow or incorrect restoration. [Guard implementation](../../scripts/experimental/tt15_motor_guard.py)

| Unchanged game | Cross-point score / switches | TT14 score / switches | Fixed score |
| --- | ---: | ---: | ---: |
| [Gantry](motor-gantry-stable-guard.json.gz) | 9–11 / 0 | **11–6** / 1 | 9–11 |
| [Five-axis](motor-five-stable-guard.json.gz) | **11–7** / 1 | 7–11 / 4 | 7–11 |

The guarded gantry reproduced all 20 unchanged fixed-map point tapes. On five-axis, the guarded rule switched after point five and reproduced five of the fixed game's 18 action tapes. Both inherited and guarded rules could win an unchanged game after changing their motor estimates. This distinguishes the quality of play from evidence that the controls actually changed. [Stable gantry TT14](motor-gantry-stable-hybrid.json.gz) · [Stable five-axis TT14](motor-five-stable-hybrid.json.gz)

## Independent physical motion

On 16 unseen matched command sequences per body, the guarded maps cut error in predicting the hardware's change in blade motion from 75.6 to 45.2 mm on gantry and from 39.4 to 14.5 mm on five-axis. The corresponding TT14 errors were 46.0 and 19.7 mm. The paired sequences were executed and replayed with both harnesses in MuJoCo. The evaluator computes the actual difference in physical tool displacement, while the frozen models predict it from the requested rates. This confirms that the fresh models captured real changes in an independently observed motion even where game outcomes were mixed. [Gantry pulses](motor-gantry-coupling-pulses.json.gz) · [Five-axis pulses](motor-five-gain-pulses.json.gz)

All 315 scored tapes across the 18 passive-comparison games replayed physical outcomes, service order, applied requests, referee events and optical map choices. The player loaded a saved checkpoint at each return game. [Gantry changed checkpoint](motor-gantry-coupling-guard-changed-player.json.gz) · [Gantry after return](motor-gantry-coupling-guard-restored-player.json.gz) · [Five-axis changed checkpoint](motor-five-gain-guard-changed-player.json.gz) · [Five-axis after return](motor-five-gain-guard-restored-player.json.gz)

## Confirming surprise with a dry movement

The alternative took one reference measurement before play. A large optical error after a scored point triggered the same sequence of paired positive and negative commands. The movements ran in a separate unscored interval on the current arm with ordinary command transport. The controller compared the two RGB trajectories after subtracting each starting blade position. A repeat within 12 mm recovered the original map. A larger discrepancy gave the learner independent evidence for a correction, fitted from the pulse, the scored movement, or their pooled observations. The matched games below revisit the passive fixtures after a pilot on their motor changes. [Pulse procedure](../../scripts/experimental/tt15_motor_probe.py) · [Full game and replay protocol](../../scripts/experimental/tt15_motor_probe_trial.py)

| Arm | First score | Changed starts | Points won | Legal returns | Return game | Changed starts | Points won | Legal returns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gantry coupling](motor-gantry-coupling-probe.json.gz) | 7–11 | 15 | 3 | 13 | 7–11 | 4 | 1 | 0 |
| [Five-axis gain](motor-five-gain-probe.json.gz) | 4–11 | 12 | 2 | 3 | **11–8** | 4 | 1 | 1 |

The repeated movement supported a correction after the first changed point on each arm. The original maps returned after three restored starts on gantry and one on five-axis, and stayed active to the end. Under changed controls the probe-assisted branch made 17 legal returns in 35 starts versus 13 in 32 for TT14. The gantry's additional returns did not produce a match win, while the five-axis arm won its restoration game. The two probe branches in unchanged games preserved all 38 fixed-map action tapes. The startup reference and subsequent measurements used 42 unscored movements in total across the six games, or 1,308 control frames at 30 Hz. The receipt separately replays those movements and all 113 scored points. [Gantry stable](motor-gantry-stable-probe.json.gz) · [Five-axis stable](motor-five-stable-probe.json.gz) · [Gantry after restoration](motor-gantry-coupling-probe-restored-player.json.gz) · [Five-axis after restoration](motor-five-gain-probe-restored-player.json.gz)

The frozen probe-assisted maps lowered independent change-prediction error to 32.5 mm on the gantry and 11.2 mm on five-axis, compared with 46.0 and 19.7 mm for TT14. Motion predictions under a physically changed harness were evaluated on the same unseen command sequences as the passive branches. [Gantry pulses](motor-gantry-coupling-pulses.json.gz) · [Five-axis pulses](motor-five-gain-pulses.json.gz)

## Unseen harnesses

The follow-up used untouched sets of rate gains, cross-axis coefficients, ball trajectories and disturbances. It reversed which arm received each type of change. The dry check selected a correction after two affected starts for the gantry and after one for the five-axis arm. The gantry probe branch came within two points of winning at deuce and made more changed-control returns than either comparison. The five-axis probe branch made one changed-control return and tied the fixed-map score. TT14 made five changed-control returns on that arm. [Gantry dry](motor-gantry-gain-fresh-probe.json.gz) · [Five-axis dry](motor-five-coupling-fresh-probe.json.gz)

| Fresh harness and map | Game score | Changed starts | Points won | Legal returns |
| --- | ---: | ---: | ---: | ---: |
| [Gantry gain, dry check](motor-gantry-gain-fresh-probe.json.gz) | 13–15 | 25 | 11 | 17 |
| [Gantry gain, TT14](motor-gantry-gain-fresh-hybrid.json.gz) | **11–9** | 16 | 10 | 9 |
| [Gantry gain, fixed](motor-gantry-gain-fresh-fixed.json.gz) | 4–11 | 11 | 2 | 2 |
| [Five-axis coupling, dry check](motor-five-coupling-fresh-probe.json.gz) | 2–11 | 10 | 0 | 1 |
| [Five-axis coupling, TT14](motor-five-coupling-fresh-hybrid.json.gz) | 4–11 | 13 | 2 | 5 |
| [Five-axis coupling, fixed](motor-five-coupling-fresh-fixed.json.gz) | 2–11 | 10 | 0 | 0 |

The dry branch made 35 legal returns in 70 changed starts across the matched and fresh harnesses. TT14 made 27 in 61 starts and the fixed maps made four in 52. TT14 won the fresh gantry game and made a false map selection before its hardware switch. The dry branch preserved all eight pre-switch action tapes across the two fresh fixtures. The repeated movement helped identify the motor context. The results leave the stronger map-selection and rally policy open. All 537 valid point-start tapes across 30 full games and all 52 unscored movements reproduced. [Complete audit](motor-summary.json.gz)

## Reproduction

Use the [TT6 simulation environment](../tt6/README.md) and the experimental scripts on `PYTHONPATH`. For fixed and TT14 selective baselines substitute `tt14_gain_trial` and specify `--mode fixed` or `--mode hybrid`. Use the matching output prefixes for the summary command.

```sh
python -m scripts.experimental.tt15_motor_trial \
  artifacts/tt13/live-gantry-remap-player.json.gz artifacts/tt15/motor-gantry-coupling-guard \
  --seed 4257013 --perturbation-seed 18583119 --profile coupling
python -m scripts.experimental.tt15_motor_probe_trial \
  artifacts/tt13/live-gantry-remap-player.json.gz artifacts/tt15/motor-gantry-coupling-probe \
  --seed 4257013 --perturbation-seed 18583119 --profile coupling
python -m scripts.experimental.tt15_motor_pulses artifacts/tt15 gantry coupling --seed 49251753
python -m scripts.experimental.tt15_motor_report artifacts/tt15
```

| Fixture | Game seed | Perturbation seed | Games |
| --- | ---: | ---: | ---: |
| Gantry coupling | 4257013 | 18583119 | 2 |
| Five-axis gain | 52364417 | 48291571 | 2 |
| Gantry stable | 9362581 | 1 | 1 |
| Five-axis stable | 73158697 | 1 | 1 |
| Fresh gantry gain | 317589053 | 184394377 | 1 |
| Fresh five-axis coupling | 723940283 | 53122309 | 1 |

The five-axis pulse audit used seed 18240223. The [development replay](motor-summary.json.gz) applies the guard to requested actions and saved sightings on TT14 fixed-map tapes. It is a discrimination check on existing actions. The fresh games instead play the guard's decisions through the physical arm. [Protocol code](../../scripts/experimental/tt15_motor_trial.py) · [Pulse audit code](../../scripts/experimental/tt15_motor_pulses.py) · [Result summary code](../../scripts/experimental/tt15_motor_report.py)
