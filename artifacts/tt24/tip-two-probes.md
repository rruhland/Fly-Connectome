# Two encoder probes after native Pong

Two balanced probe directions substantially repaired the five-axis motor map learned during ordinary native Pong on 10 of 12 fresh altered-arm drops. The repair improved gameplay under one wiring and worsened it under the other. It raised the sum of longest streaks from 58 to 75 while reducing the total number of qualifying rebounds from 90 to 75. The paired streak comparison was three wins, three losses, and six ties. These runs establish a short motor-calibration benefit and leave a repeatable gameplay benefit unproven.

## Transfer and matched preparation

Each map came from twelve reward-trained Pong points controlling the stock arm through a dense joystick interface. The two fitting histories amounted to 59.08 and 68.02 seconds of physical commands. The experiment reused the original map and command windows retained by the fit for each encoder. It gave each encoder’s command-window covariance equal total weight and chose the two directions with the least measured coverage. The two weakest normalized singular values were 0.023 and 0.036 for wire 0 and 0.025 and 0.032 for wire 1. The selection used neither target rewards nor the known wire matrix. [Registered sources, probe directions and held releases](tip-two-probes-plan.json)

On the altered arm, each selected direction received a +0.075 rad and −0.075 rad requested excursion, with settling between endpoints and a return to the initial command target. The difference between the encoder endpoints estimates the two measured response columns. The update replaced the prior’s response in that span. The unprobed complement stayed at its source estimate or at the identity for the stale control. The known wire condition used the evaluator’s exact input matrix. The different optical model and motor head came from the existing carried policy.

Six fresh drops per wire included the altered arm and wrist view, independently varied ball mass and drag, and two scheduled changes in gravity. All four conditions shared exact physical probe and normal visual preparation commands through the original source map. They diverged only when their decoder took effect after release. No pose reset occurred between probes and play. Probes cost 96 held ticks or 1.60 seconds per drop. Including optical preparation, each 12-drop condition incurred 23.40 seconds of held time. Live play was capped at ten seconds. Each trajectory reproduced its referee scores, physical clock, and gravity-change events on exact action replay. [Wire 0 tapes](tip-two-probes-0.json.gz) · [Wire 1 tapes](tip-two-probes-1.json.gz) · [Runnable assay](../../scripts/experimental/tt24_tip_two_probes.py)

## Held results

| Motor decoder | Sum of longest streaks | Wire 0 / wire 1 | Total rebounds | 10+ drops | 20+ drops | Mean live seconds |
|---|---:|---:|---:|---:|---:|---:|
| Original source | 58 | 19 / 39 | 90 | 3/12 | 1/12 | 2.69 |
| Source plus two probes | 75 | 12 / 63 | 75 | 2/12 | 2/12 | 2.57 |
| Identity plus the same two probes | 52 | 41 / 11 | 52 | 1/12 | 1/12 | 1.85 |
| Known wire | 67 | 36 / 31 | 74 | 2/12 | 1/12 | 2.26 |

Relative map error fell from 13.32% to 2.36–2.91% on all six wire-0 starts and from 5.29% to 1.64–2.94% on four wire-1 starts. The two wire-1 exceptions had errors of 16.36% and 38.00%. Their positive and negative encoder responses were strongly asymmetric. A referee-side replay found the paddle touched the table during both probe sequences. The most distorted sequence drifted 0.061 rad in joint space on returning to its initial command target. The other moved only 0.00007 rad on return, so closure alone would have missed a contaminated measurement. The paddle touched the table in one other sequence with little endpoint asymmetry. None of the probes clipped actuator command targets.

Removing motor-map error alone did not consistently improve rallies. Wire 0 had accurate patched maps on every start and still lost seven summed streak rebounds versus the unchanged source. The exact wire also allowed only one rebound on five of its six wire-0 starts. These observations constrain the claim to calibrating a local input map. A separate five-direction target-only decoder was outside this assay.
