# Learning what to send one tick early

The regulation encoder probe recovered the compact arm’s signed and permuted motor routing, including an actual one-tick actuator queue. Accurate identification alone left the frozen stereo policy vulnerable to the physical wait. We kept that queue active, used the empirical motor map from the regulation probe, and fit timing readouts from the original TT24 compact demonstrations. The near arm retained stereo RGB, measured joints and its own issued commands as inputs. The far arm planned live, and all branches played from the start of each point. [Motor transfer and timing baseline](compact_motor.md) · [Implementation](../../scripts/experimental/tt27_delay_readout.py)

The direct readout paired the current sensory history with the following nominal command and its resulting target position. Pairs had to share a demonstration release and match the next row’s recorded previous command and encoder displacement. This retained 4,980 adjacent pairs in the 48-release optical fit and 6,229 pairs in the 54-release corrective fit. Both readouts used the same local weighted least-squares procedure as the original policy. The blend gave the one-tick-ahead readout 65% weight and retained 35% of the timing baseline. A separate local fit predicted the next optical feature vector and passed it through the frozen policy. All delayed branches projected the pending motor command into their next target. [Original training data](../tt24/rally-improve.md)

Candidate selection used four explicitly labeled diagnostic retries. It ranked the number of points reaching a second near return, then total near returns, then the rate of abrupt adjacent reversals. The independent held course compared the selected readout with the original motor timing under the same queue, one-frame linear pixel extrapolation under the same queue, and the intact interface. We also ran the command blend as a supplementary branch on the same held releases because it reached second returns in two diagnostic points. The source demonstrations supplied all readout parameters.

## Diagnostic retries

| Four previously observed starts | Near / far returns | Second near return | Longest alternating exchange |
| --- | ---: | ---: | ---: |
| Direct one-tick-ahead commands | 2 / 2 | 0 | 2 |
| Blended one-tick-ahead commands | 7 / 7 | 2 | 6 |
| Learned next optical features | 9 / 9 | 2 | 10 |
| Original timing with correctly mapped queue | 1 / 1 | 0 | 2 |
| One-frame pixel extrapolation | 1 / 1 | 0 | 2 |
| Intact interface | 22 / 22 | 4 | 14 |

The learned optical forecast took two previously observed points from zero near returns under the original delayed timing to five and four. The command blend reached three near returns and three opposing replies on two different starts. These results selected the learned optical forecast as the primary candidate. [All 24 diagnostic tapes](delay_readout-development.json.gz)

## Fresh releases

| Eight fresh starts | Near / far returns | Second near return | Longest exchange | Abrupt reversals per second |
| --- | ---: | ---: | ---: | ---: |
| Learned next optical features | 1 / 1 | 0 | 2 | 12.24 |
| Blended one-tick-ahead commands | 0 / 0 | 0 | 0 | 28.27 |
| Original timing with correctly mapped queue | 0 / 0 | 0 | 0 | 26.66 |
| One-frame pixel extrapolation | 2 / 2 | 0 | 2 | 31.60 |
| Intact interface | 11 / 11 | 2 | 14 | 6.51 |

The sustained diagnostic rallies did not recur on fresh trajectories. The learned optical forecast reduced rapid motor reversals and restored one first return, while linear pixel extrapolation restored two. The command blend made zero returns. The reversal metric counts adjacent normalized rates crossing directly between above +0.1 and below −0.1, pooled across the five applied axes and divided by elapsed play time. Mean absolute per-tick rate changes were 0.186 for the optical forecast, 0.271 for the original delayed timing, 0.315 for pixel extrapolation and 0.126 for the intact interface. These measurements come from different realized trajectories and durations.

All 64 branch tapes reproduced their physical commands and referee events. The eight held launches are disjoint from the diagnostic releases, original training rows and prior compact experiments. [Primary and control tapes](delay_readout-held.json.gz) · [Command blend follow-up](delay_readout-command-held.json.gz)

## Reproduce

```sh
export PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt27_delay_readout --develop
python -m scripts.experimental.tt27_delay_readout
python -m scripts.experimental.tt27_delay_readout --command-only --workers 2
```
