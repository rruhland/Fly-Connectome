# Rare missed replies by the physical far player

The stock broad-random opponent already answers about 99% of achieved arm strokes. Every legal far opportunity in the two archived 31- and 32-return arm games was answered. The far player is a source of occasional truncated points, but these misses did not end either of those long arm rallies. The standard opponent and its draw protocol are unchanged. [Census and source digests](census.json)

| Archive group | Legal far replies | Terminal far misses | Censored after an own legal landing |
| --- | ---: | ---: | ---: |
| Articulated-arm history | 1,081 | 10 | 1 |
| First five declared endurance practice blocks | 398 | 4 | 0 |
| Mirrored physical carriage capacity controls | 440 | 0 | 8 |

An opportunity starts at an earned legal near-arm landing, after a real contact. The installed opening flight, referee-truncated comparisons, and repeated identical physical offers are not counted as missed replies. An actual far terminal miss means the near player won the point after that landing without another legal far return. Every one of the 14 misses was a no-contact miss, rather than a far touch that landed illegally. The aggregate description uses historical records only for prevalence, not for trigger design or motor testing.

## What was detectable in time

The four provably assigned practice misses replayed exactly through stock control. For each, all three ordinary plan stages produced no stroke. Two first bounces were short and outward, so the ball drifted outside the stock ±0.81 m lateral planning box before a useful contact. One was a very low trailing-edge rebound, almost flat below the stock 0.86 m planning floor after passing the back of the table. A fourth was a net dribble that expired before the stock rescue could reach it. This test does not modify that net recovery.

From the live far player's already available ball position, velocity and spin, the opt-in pilot forecasts a slightly longer independent flight when ordinary stage one produces no plan. A narrow trigger flags the two named geometries and leaves the installed opening and any immediate stock plan alone. An audit at observed physical near-paddle separation flagged the three geometric misses 0.85–1.03 simulated seconds before their bounce, and flagged none of the 398 legal stock replies on the first five practice blocks. The geometry also flagged none of 77 legal stock replies in the next numbered practice block after the gate was frozen. Those counts are diagnostic. They are not a guarantee against false activations. [Practice trigger audit](gates.json) · [Next practice block](gates-next.json)

The alternative search enlarges the **contact** region, not the opponent's aim. It reuses the exact independently drawn four-coordinate goal, and calls the existing speed- and acceleration-limited force controller. It requires a table-clear reference path and rejects a stroke whose independently simulated native impact and flight depart too far from the requested receiver location or pace. It cannot redraw or narrow a target, set physical state, or award contact. The near arm simply replays its already-issued commands in the paired test, so this is not evidence about a learned near player's response to the repaired ball.

## Paired physical result

All three triggered practice misses became legal native far returns with the exact original aim. The unflagged net dribble stayed a miss. The near arm's legal score did not change. The five practice control points reproduced all stock referee events exactly, and all 22 previously successful later far replies across the nine pairs remained unchanged. [Paired role, requests, physical referee events and crossings](pilot.json.gz)

| Practice release | Geometric trigger | Far score | Actual miss in requested x / z | Requested → actual pace |
| --- | --- | ---: | ---: | ---: |
| 12596 | Short outward | 8 → 9 | 19.8 / 1.2 mm | 3.79 → 3.68 m/s |
| 14645 | Low trailing edge | 2 → 3 | 8.4 / 26.9 mm | 3.86 → 3.21 m/s |
| 13077 | Short outward | 2 → 3 | <0.1 / 13.7 mm | 3.81 → 4.47 m/s |
| 15035 | None, net dribble | 3 → 3 | — | — |

In 12596 the far player actually hit at x = −0.931 m. In 14645 it hit at z = 0.783 m, behind the far end at y = 1.915 m. All three changed trajectories used the same physical motor forces, with observed peaks of 29.9 N and 3.09 m/s. CAD verification checked every integration state from the first changed force to the measured reception, 2,685 states in total, with zero penetrations. The stock replay exactly reproduced each archived event tape. Two repaired shots missed the requested pace by about 0.65 m/s despite using the original target and closely matching its location. The geometry is promising, but recovery from three calibration misses is not a separate blind test. Keep the repair opt-in until a broader predeclared practice cohort confirms the trigger and the achieved return tolerances.

## Reproduce

Run the following commands from the repository root with the project environment and MuJoCo available.

```sh
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt40_far_same_aim audit
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt40_far_same_aim gates
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt40_far_same_aim pilot
```

For the later frozen-gate check add `gates --through 6 --out /tmp/far-gates-06.json`. The pilot checks a declared set of four already observed practice misses and the first two cards in each corresponding numbered practice block, omitting a repeated target card. No assessment card is or can be a pilot input.
