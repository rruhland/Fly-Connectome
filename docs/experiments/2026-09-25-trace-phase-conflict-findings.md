# Quiet targets suppress trace amplitude without erasing event ranking offline

The [registered frozen-capacity audit](2026-09-25-trace-phase-conflict-protocol.md) fitted one evaluation-only 17×17 local linear readout to traces from 192 training-shape interruptions spanning two-, three-, and four-frame gaps. It first used only last-blank-frame → reappearance targets, then added an equal number of quiet targets from each episode. The four-phase arm was skipped under the preregistered stop rule when the second arm's training gap-exit F1 fell below 0.20. No offline weight entered online M1A learning.

| Offline shared readout | Training gap-exit F1 | Held-out gap-exit F1 | Training quiet false-positive pixels |
| --- | ---: | ---: | ---: |
| Gap exit only | 0.258 | 0.176 | 760 |
| Gap exit + quiet | **0.000** | **0.000** | 0 |

The exit-only arm already fits mixed gap durations much worse than the previous three-frame-only fit (0.258 versus 0.768 training F1). Its late four-frame training F1 is 0.088, indicating one scalar local field has difficulty with timing variation before quiet targets are added. With quiet targets, **all** 1,968 training gap-exit event pixels fall below the fixed 0.5 threshold. This reproduces the online learner's zero gap-exit F1, but it does not establish that quiet and event examples are inseparable.

An exploratory threshold-free audit on held-out exit events versus reachable quiet-frame pixels found almost unchanged ranking: area under the ROC curve **0.908** for exit-only and **0.907** for exit + quiet. Mean held-out event score fell from 0.268 to 0.106, while the quiet score's 99th percentile fell from 0.389 to 0.155. The quiet target scales down both distributions; the fixed 0.5 decision threshold hides remaining separation. The registered fixed-threshold gate fails, but calling this a representation failure would overstate the evidence.

The next decisive diagnostic is to save the **online** continuous-trace learner and measure its threshold-free event/quiet ranking on the same frozen held-outs. If online ranking remains high, test one locally computed event-rate calibration or balanced local plasticity rule, not a global hand-tuned threshold. If online ranking is poor, its shared eligibility/update rule destroys useful trace information and should be redesigned before adding recurrent complexity. The current model is not promotable; production M1A remains unchanged.
