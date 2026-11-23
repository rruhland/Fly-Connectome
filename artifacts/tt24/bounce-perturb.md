# Wrist feedback after a flight disturbance

Passive stability can help keep a ball bouncing on a periodic paddle path. The paired test measures whether the learned wrist response adds recovery after an unexpected lateral change. [Human ball-bouncing study](https://pubmed.ncbi.nlm.nih.gov/11304282/)

Before running the experiment, we fixed 24 fresh stock starts and 16 independent alternate-arm starts. Each cohort balances positive and negative changes of 0.16 or 0.24 m/s along the two horizontal axes. The test applies the assigned change to the physical ball after the third consecutive qualifying rebound, while the ball is airborne. Every start reaching this trigger enters the paired comparison. The original wrist pupil has four motor-labeled demonstrations and privileged perception calibration. The alternate-arm pupil retains the stock action fit and uses its previously exported correction from 248 frames of wrist and calibrated fixed-camera RGB with own encoders over four changed-body drops. At test time both pupils see wrist RGB and their own joint angles. [Stock training](bounce-ident.md) · [Public body recalibration](metric-transfer.md)

Each start first runs with the frozen pupil and no kick. The undisturbed motor tape then replays on the same start with exact commands and referee results. The three perturbed branches share that initial state and motor history through the kick. The live controller continues using wrist images. The tape branch repeats the undisturbed joint rates. The hold branch commands zero rates from the kick onward, which holds its existing joint targets. Only the ball velocity changes at the kick. The live and tape rate commands are identical on the kick frame, and a separate replay verifies each disturbed physical trajectory.

All scores use the same time limit within a pair, the earlier of two seconds after the kick or the end of the original tape. The tape ends there without extra commands. Survival means observed time until a physical loss or that limit. Runs that reach the limit are right-censored. The rebound count is the uninterrupted continuation after the third rebound. A later rebound following an unqualified impact is counted separately in the raw receipt. [Experiment](../../scripts/experimental/tt24_bounce_perturb.py)

## Held results

| Body and fixed calibration | Eligible starts | No-kick continuation | Disturbed live | Disturbed tape | Disturbed hold | Live wins / ties / losses vs tape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock | 21/24 | 53 | 47 | 18 | 15 | 11 / 8 / 2 |
| Alternate with public target calibration | 16/16 | 22 | 20 | 9 | 9 | 7 / 8 / 1 |

The four score columns sum the number of immediate consecutive rebounds after the kick time. On stock, live feedback extended mean observed survival from 0.568 to 1.080 seconds and physical losses fell from 18 to 10. On the alternate arm, survival rose from 0.483 to 0.822 seconds and losses fell from 15 to 11. The original tape length set the matched control limit on 12 starts in each cohort. The fixed-rate and held-pose controls still produced rebounds in both cohorts, while live feedback preserved more of the ongoing sequence.

Live rates differed from the tape before the next measured paddle contact or a 20-tick early cutoff on 21/21 stock starts and 15/16 alternate starts. Where they differed, the first change came after a median of one 60 Hz control tick on both arms. All 151 tapes reproduced the physical and referee traces with the same intervention. [Per-start results, commands, and survival records](bounce-perturb.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_bounce_perturb --output /tmp/bounce-perturb-reproduction.json.gz
```
