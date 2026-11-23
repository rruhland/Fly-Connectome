# One carried wrist policy under combined physical changes

One unchanged 30,668-byte inference bundle controls both stock and altered simulated SO-101 arms across a mixed course. It receives native 192 × 144 wrist RGB and own joint angles. The first wrist image selects one of two earlier optical calibrations. An origin-reuse coefficient of 1.0, selected on native Pong, sets both the familiar arm reference and its motor equilibrium. The same frozen donor-trained feedback readout runs across all releases. [Portable bundle](carried-wrist.npz) · [Public policy loader](../../scripts/experimental/tt24_carried_wrist.py)

On every stock drop, the bundle survived both gravity changes and sustained 40–56 consecutive rebounds. On the altered arm, two of eight drops reached the 15 s limit. Their longest streaks were 12 and 28. The optical calibration was recalled correctly on every mixed start, including the altered-arm failures.

The five motor outputs were acquired through physical bounce reward on three hinged donor designs. The donor training used clean simulated state and no teacher commands. It scored 18,000 donor practice drops, used 8,664 donor development trials and 30 motor probes, and first selected the final weights after 5,400 practice drops. Prior stock optics used eight bounded probes with stereo RGB and encoders. Prior altered optics used four such drops, totaling 154 control frames. The reference pose comes from earlier donor starts. Both stationary-camera systems were used during calibration only. This stress qualification adds zero training, optical fitting, or receiving-arm motor reward. [Donor training](tt24_donor.md) · [Stock optical calibration](public-body.md) · [Altered optical calibration](public-morph.md) · [Pong origin selection](tt24_pong_origin.md)

## Paired assessment

Sixteen fresh launches interleave eight stock and eight altered bodies. Each learner begins in a `WidePosture` draw. The separate stock launcher installation uses ordinary independently seeded release jitter and holds the ball in place during 14–38 charged preparation ticks, or 0.23–0.63 s. Each receiving body has its own physical start seed. The altered body retains the alternate links, wrist mount and drive settings. Each trial draws a ball mass and inertia scale from 0.82–1.18 and a drag-acceleration multiplier from 0.82–1.22. The camera and visible ball size stay fixed within each installation.

Gravity starts at nominal. It switches at 3.00–3.70 s and again at 6.00–6.90 s after release, between nominal and two opposite bands. The low band is 0.78–0.88 times nominal and the high band 1.13–1.26. Each body has four high-first and four low-first courses. The schedule is drawn in advance and identical across the three paired branches. Gravity affects the whole simulated fixture. The actor sees the resulting image and encoder changes during play. Physics settings, launcher state and switch times are evaluator information.

The active comparison uses the same five-joint weights with the original stock observer and the current starting pose as its visual and motor reference. It therefore tests the bundle's combined origin reuse and optical recall. The passive arm holds its initial servo targets. All branches use the same 15 s post-release cap. A course that reaches this cap is right-censored for time to ball loss. [Runner](../../scripts/experimental/tt24_integrated_stress.py) · [Scheduled parameters, commands and referee data](integrated-stress.json.gz)

## Held results

| Controller and body | Mean longest streak | ≥10 | ≥20 | ≥30 | Ended at 15 s | Median observed time to loss or cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bundle, stock | 50.00 | 8/8 | 8/8 | 8/8 | 8/8 | 15.00 s |
| Bundle, altered | 6.13 | 2/8 | 1/8 | 0/8 | 2/8 | 1.20 s |
| Bundle, both bodies | 28.06 | 10/16 | 9/16 | 8/16 | 10/16 | 15.00 s |
| Reset origin and stock optics, both bodies | 0.56 | 0/16 | 0/16 | 0/16 | 0/16 | 0.52 s |
| Passive, both bodies | 0.50 | 0/16 | 0/16 | 0/16 | 0/16 | 0.44 s |

All ten bundle runs that reached the first change also reached the second and stayed alive to the 15 s cap. Each scored legal rebounds between changes and again afterward. Their counts after the second change were 21–35. The two altered survivors scored 49 and 54 total qualified rebounds across shorter consecutive sequences. Six altered runs terminated before the first switch. Neither control reached the first switch, and each had a maximum streak of one.

Nineteen of the 20 gravity changes reached by the bundle occurred while the ball was physically airborne. One timed cut coincided with a stock paddle contact, and its paired interventions retained the same hidden clock. All 48 issued command sequences reproduced their exact terminal ticks, complete referee events and recorded ball state at every reached switch under replay. [Per-drop metrics and command tapes](integrated-stress.json.gz)

This is a compound qualification on two pre-calibrated simulated bodies. The comparison combines the carried reference and optical selection, and the common stock dispenser makes the altered arm's result specific to that installation. The bundle has no explicit gravity-estimation or online weight-update mechanism.

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_integrated_stress --output /tmp/integrated-stress.json.gz
```
