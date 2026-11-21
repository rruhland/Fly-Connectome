# Motor gain and coupling during table tennis

The four-axis gantry won 11–4 after an unannounced cross-axis motor change. On the same opening, the retained original map lost 4–11, signed remapping lost 3–11 and a new local fit lost 8–11. While the coupling was active, the adaptive gantry won eight of 11 starts and made six legal returns. The five-axis arm gained two points and one legal return during its changed interval and lost 3–11. [Gantry first-person game](gain-gantry-coupling.mp4) · [Gantry starting player](../tt13/live-gantry-remap-player.json.gz) · [Five-axis starting player](../tt13/live-five-remap-player.json.gz)

## Coupling and restoration

The changed-start columns count only starts played with coupling active, including the four at the start of the return game. A point start may produce multiple legal returns. The game scores use every start, including the original controls. [Machine-readable results and fit costs](gain-summary.json.gz)

| Arm and map | First game | Changed starts | Points won | Legal returns | Return game | Changed starts | Points won | Legal returns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gantry correction](gain-gantry-coupling-hybrid.json.gz) | **11–4** | 11 | 8 | 6 | **11–7** | 4 | 3 | 4 |
| [Gantry fixed](gain-gantry-coupling-fixed.json.gz) | 4–11 | 11 | 1 | 1 | **12–10** | 4 | 2 | 2 |
| [Gantry signed remapping](gain-gantry-coupling-remap.json.gz) | 3–11 | 10 | 0 | 0 | 5–11 | 4 | 2 | 2 |
| [Gantry local fit](gain-gantry-coupling-refit.json.gz) | 8–11 | 15 | 5 | 4 | 3–11 | 4 | 1 | 1 |
| [Five-axis correction](gain-five-coupling-hybrid.json.gz) | 3–11 | 11 | 2 | 1 | **11–8** | 4 | 3 | 1 |
| [Five-axis fixed](gain-five-coupling-fixed.json.gz) | 1–11 | 8 | 0 | 0 | **16–14** | 4 | 0 | 0 |
| [Five-axis signed remapping](gain-five-coupling-remap.json.gz) | 2–11 | 9 | 1 | 0 | 6–11 | 4 | 0 | 0 |
| [Five-axis local fit](gain-five-coupling-refit.json.gz) | 2–11 | 9 | 1 | 0 | 3–11 | 4 | 0 | 0 |

Both adaptive players selected a full correction after the first changed start. In the return games, both selected the exact original map after the first restored start. Later camera evidence caused both players to switch away from that original map again, and both finished with a correction active. The return-game wins show that the full players kept functioning through the change back. The fixed maps also won both return games after their original controls resumed. [Gantry changed checkpoint](gain-gantry-coupling-hybrid-changed-player.json.gz) · [Five-axis changed checkpoint](gain-five-coupling-hybrid-changed-player.json.gz) · [Gantry after restoration](gain-gantry-coupling-hybrid-restored-player.json.gz) · [Five-axis after restoration](gain-five-coupling-hybrid-restored-player.json.gz)

The independent gantry fixture used fresh coupling and a different trajectory sequence. The player selected a correction after the third changed start. It scored four points and made five legal returns in 14 changed starts, finishing 6–11. The new local fit also lost 6–11 after winning four changed points with one legal return in 14 starts. The fixed map lost 4–11, with two points and three returns in 12 changed starts. Signed remapping lost 3–11, with one point and two returns in 11 changed starts. The first gantry win did not repeat. [Correction](gain-gantry-coupling-check-hybrid.json.gz) · [Local fit](gain-gantry-coupling-check-refit.json.gz) · [Fixed](gain-gantry-coupling-check-fixed.json.gz) · [Signed remapping](gain-gantry-coupling-check-remap.json.gz)

## Independent gain changes

| Arm and map | First game | Changed starts | Points won | Legal returns | Return game | Changed starts | Points won | Legal returns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Gantry correction](gain-gantry-gain-hybrid.json.gz) | 9–11 | 18 | 7 | 11 | 9–11 | 4 | 1 | 2 |
| [Gantry fixed](gain-gantry-gain-fixed.json.gz) | 2–11 | 9 | 0 | 0 | 10–12 | 4 | 0 | 0 |
| [Five-axis correction](gain-five-gain-hybrid.json.gz) | 4–11 | 11 | 2 | 10 | **11–7** | 4 | 1 | 2 |
| [Five-axis fixed](gain-five-gain-fixed.json.gz) | 3–11 | 10 | 0 | 0 | 6–11 | 4 | 1 | 1 |

The five-axis arm made eight of its ten changed-control returns in one rally. It ended the first game with a gain-only correction. The gantry retained a full correction even though the hardware only changed gains. It selected the original map on the first restored start. The five-axis arm kept fitting corrections after restoration. A false correction in the five-axis opening also changed two of the four pre-switch action tapes, so that gain fixture starts from the same checkpoint and schedule with some action history already diverging.

## Unchanged controls

The adaptive gantry made four false switches and won 12–10, compared with a fixed-map loss of 8–11. The adaptive five-axis arm made five false switches and lost 5–11, compared with 6–11 fixed. Just two of 19 fixed gantry point tapes and six of 17 fixed five-axis tapes matched. The unchanged-control results make the current switch rule too unstable for general use, and they show that a scoring improvement alone cannot establish motor identification. [Gantry adaptive](gain-gantry-stable-hybrid.json.gz) · [Gantry fixed](gain-gantry-stable-fixed.json.gz) · [Five-axis adaptive](gain-five-stable-hybrid.json.gz) · [Five-axis fixed](gain-five-stable-fixed.json.gz)

## What the arms could use

The experiments used the same shaded first-person stereo, physical tosses, alternating service and points to 11 with a two-point margin as the earlier motor-port trials. Each main series began with four starts on the original controls, changed them before the fifth and played to completion. The second game loaded the saved player, played its first four starts with the change still active, then restored the original controls. Fresh ball and disturbance seeds distinguish the independent gantry challenge from the original game.

The environment multiplied the transport-delayed, noisy rates by a private matrix before applying the original physical ports. Gain changes attenuated individual rates. The coupling challenge also added input from a second port in each output direction. Each coupling row had an absolute sum of .98, which kept the mixed normalized rates inside the simulator's motor input bounds. The positions, forces and impacts still came from MuJoCo. [Hardware implementation](../../scripts/experimental/tt14_gain_hardware.py)

The adaptive player tried the original and previously recalled maps and fit either a diagonal or a full correction around the original camera body. It reused the model's quadratic geometry and each of its per-axis settling curves. The fits used requested rates and RGB estimates of blade position and pitch. Alternating four-frame blocks chose and tested a map using camera observations from the same point. A large improvement permitted a switch after one point. Smaller proposed changes required supporting evidence from the preceding point. The update had no access to the perturbation, the underlying joint state, the applied commands, the incoming ball seed or the referee's evaluation. The learned correction enters the ordinary predictive stroke servo on the next point. The controls keep the original map, test only signed relabelings, or fit new linear directions with a shared settling curve. All branches use the inherited stroke, service and timing updates. [Camera-based learner](../../scripts/experimental/tt14_gain_model.py) · [Motor-aware stroke controller](../../scripts/experimental/tt14_gain_servo.py)

The frozen maps were also tested on 24 fresh paired physical pulse sequences per body. Error in predicting the motion change dropped from 53.9 to 27.7 mm on the gantry and from 78.3 to 32.7 mm on the five-axis arm, compared with keeping the original map. The five-axis motor prediction improved even though its first complete game showed a small gain in scoring. [Independent physical and transfer checks](signal.md)

All 551 point tapes across 32 complete games reproduced physical outcomes and camera-only map decisions. All four opening action digests matched in each coupling fixture and the gantry gain fixture. Across 198 adaptive updates, 140 skipped candidate fitting and took a median 0.87 ms of CPU time. The 58 updates that evaluated candidates took a median 12.1 ms. This measures the map update after each point and excludes the normal camera processing, physics and stroke control. [Game audits and cost measurements](gain-summary.json.gz)

## Reproduction

Use the [TT6 simulation environment](../tt6/README.md) and set the experimental scripts on `PYTHONPATH`. The main gantry example below plays and verifies both games. The changed and restored checkpoints are written alongside the receipt.

```sh
python -m scripts.experimental.tt14_gain_trial \
  artifacts/tt13/live-gantry-remap-player.json.gz artifacts/tt14/gain-gantry-coupling-hybrid \
  --profile coupling --mode hybrid --seed 261051481 --perturbation-seed 58214071
python -m scripts.experimental.tt14_gain_replay \
  artifacts/tt14/gain-gantry-coupling-hybrid.json.gz /tmp/gantry-11-4.mp4
python -m scripts.experimental.tt14_gain_report artifacts/tt14
```

| Fixture | Game seed | Perturbation seed | Profile and games |
| --- | ---: | ---: | --- |
| Gantry | 261051481 | 58214071 | `coupling`, two |
| Five-axis | 45877193 | 51260741 | `coupling`, two |
| Gantry | 21690821 | 9217309 | `gain`, two |
| Five-axis | 257112749 | 94140583 | `gain`, two |
| Gantry | 318629854 | 1 | `stable`, one |
| Five-axis | 71261292 | 1 | `stable`, one |
| Independent gantry challenge | 359128359 | 81437007 | `coupling`, one |

The gain-only and stable fixtures compare `hybrid` and `fixed`. The coupling fixtures also run `remap` and `refit`. The series runner verifies the physical applied tape, referee events, alternating-service score and action digests, then regenerates each map decision from recorded camera sightings and requested commands. [Game and replay code](../../scripts/experimental/tt14_gain_trial.py) · [Summary code](../../scripts/experimental/tt14_gain_report.py)

To reuse a frozen map on the same mechanism, `load_map(checkpoint)` returns its camera body and the learned input transform. The home-position spatial Jacobian in requested-port order is `body.jacobian(np.zeros(body.action_size))[:3] @ transform`. The predictive controller also applies this transform to each outstanding command before using the separate per-axis response curves. [Checkpoint and map loader](../../scripts/experimental/tt14_gain_trial.py)
