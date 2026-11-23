# Transporting the familiar paddle pose across bodies

The carried wrist controller ordinarily reuses the same familiar joint angles on both simulated SO-101 bodies. Here the target is the **familiar paddle center and normal**, in shared world coordinates. A single local solve finds the alternate arm's joint angles for that pose. The new reference sets the servo equilibrium and the decoded spatial origin consistently. The same frozen visual feedback head continues to issue all five normalized joint rates. There is no inverse applied to the individual actions. [Carried wrist bundle](carried-wrist.npz) · [Controller implementation](../../scripts/experimental/tt24_tool_equilibrium.py)

Aligning the altered arm to the stock tool pose reduced its mean longest streak from 10.75 to 1.08. A separate visual workspace adjustment that preserved the altered paddle's familiar height and face angle improved the number of ten-rebound runs from five to eight on 12 new paired drops. It produced fewer 20-rebound runs than the original bundle.

## Coordinate registration and correction

The stock and altered optical fits use the same fixed stereo rigs and CAD mounting convention. The configured focal lengths, principal points, centers and viewing bases match exactly for both stereo cameras in the calibration setup. This checks the common coordinate frame. The two paddle estimates themselves remain calibrated optical models. [Stock calibration](public-body.md) · [Altered calibration](public-morph.md)

At the same familiar donor joint angles, the altered observer places the paddle center relative to stock at **−14.4 mm lateral, −12.2 mm depth, and +19.0 mm vertical**, a net length of 26.8 mm. The estimated outward unit normals differ by about 4.8°. The pose solve changes pan, shoulder lift, elbow flex, wrist flex and wrist roll by **+2.85°, +4.73°, −3.09°, +3.37°, +0.67°**. The moved center and normal match the stock observer's familiar pose within numerical precision. The solve is confined to ±0.24 rad per joint around the familiar configuration.

The archived observers' typical absolute launch coordinates differ by 50.7 mm laterally, 33.2 mm in depth and 45.5 mm in height. Those are earlier public optical estimates from each installation. They motivated an additional workspace branch for the common stock launcher. It starts with the transported pose, then uses three consistent wrist ball estimates from the first ten frames to shift the paddle's horizontal reference toward the observed ball. It retains the stock height, normal and typical ball-to-paddle horizontal offset. The shift is capped at 55.7 mm, 75% of the source optical paddle radius. The pose is solved one more time when the estimate is accepted. The servo's accumulated absolute commands carry across that coordinate change. A missing sighting leaves the transported reference active.

## Paired physical test

Twelve alternate and eight stock receivers start at fresh independent `WidePosture` draws. Every release comes from a common stock dispenser setup with ordinary release jitter and an independent seed. The ball remains held during 14–38 charged visual preparation frames. All conditions see the same independently drawn receiver pose, release position, velocity and spin. Gravity, mass and drag use the standard regime for each body. Trials have a six-second post-release cap.

The original bundle provides the reference comparison. The aligned and reverse controls shift the carrier by opposite joint offsets of exactly the same magnitude. On recalled stock optics both shifts reduce to the original bundle, so those controls share its verified tape. The control mode, prior calibrated maps, wrist RGB and own encoders determine the reference. The simulator's true pose, launcher and release clock stay in the evaluator. The earlier stock and alternate optical fits used eight and four probe drops respectively. This experiment adds zero training or calibration data and no receiving motor reward. [Paired exact command and referee data](tool-equilibrium.json.gz)

| Reference | Alternate mean longest streak | Alternate ≥10 | Stock mean longest streak | Stock ≥10 |
| --- | ---: | ---: | ---: | ---: |
| Original carried bundle | 10.75 | 5/12 | 7.50 | 3/8 |
| Aligned world pose | 1.08 | 0/12 | 7.50 | 3/8 |
| Reversed joint correction | 0.83 | 0/12 | 7.50 | 3/8 |
| Ball-relative workspace around stock height and normal | 1.00 | 0/12 | 9.75 | 4/8 |

The pose-aligned reference lost to the bundle on ten altered-arm drops and tied on two. It still scored a legal rebound at its first paddle contact on ten of 12 drops. The workspace branch did so on 12 of 12 and still scored only one rebound per altered drop. The original bundle scored on the first contact in all 12 and sustained streaks as long as 25. Geometric placement was sufficient to initiate a rebound in most cases. The failure was in sustaining them.

On stock, the workspace branch added one ten-rebound drop, with four paired wins, two ties and two losses. Its launcher estimates used the calibrated color-defined ball sighting and settled on every drop. Median horizontal error was 3.61 mm on stock and 6.67 mm on altered. Sixty-four distinct tapes reproduced their complete physical referee metrics and terminal ticks. The 16 remaining stock controls shared eight identity-transport tapes. [Held commands and full score records](tool-equilibrium.json.gz)

## Independent altered-arm follow-up

The first workspace branch inherits the transported stock height and face angle. The follow-up keeps the recalled body's original visual and motor reference until it sees enough evidence to shift just the horizontal position. It retains the altered body's familiar height and face angle in the pose solve and preserves the target stock horizontal ball-to-paddle offset. The workspace cap, equal servo readout, sensory input and preparation rules remain as above. Twelve new independent receiver and dispenser pairs qualify this follow-up. Every case accepted a visual query. The median realized horizontal reference shift was 45.2 mm, with 0.19 mm median height change and approximately 0.05° median change in the two estimated normal components. Median horizontal sighting error was 4.61 mm.

| Reference on 12 fresh altered drops | Mean longest streak | Maximum | ≥10 | ≥20 | Legal first paddle contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original carried bundle | 9.17 | 21 | 5/12 | 3/12 | 9/12 |
| Ball-relative workspace preserving altered height and normal | 11.25 | 19 | 8/12 | 0/12 | 11/12 |

The workspace branch won six paired cases and lost six. It improved the mean by 2.08 consecutive rebounds and recovered several starts that the original bundle lost early. In this small cohort it reduced the longest achieved runs. All 24 follow-up command tapes reproduced the full physical referee metrics and terminal ticks. [Fresh altered-workspace commands and scores](tool-equilibrium-local-workspace.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_tool_equilibrium --output /tmp/tool-equilibrium.json.gz
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_tool_equilibrium --local-workspace \
  --output /tmp/tool-equilibrium-local-workspace.json.gz
```
