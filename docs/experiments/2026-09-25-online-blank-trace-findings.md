# A continuously active trace readout underfits online despite frozen capacity

The [registered opt-in learner](2026-09-25-online-blank-trace-protocol.md) used the saved learned motion code and trace, a zero-initialized 17×17 local ON/OFF emission field, and the same delayed local event-error update as earlier readouts. One pass covered 96 clean episodes and 64 generic interruptions at each of two-, three-, and four-frame gaps. No offline decoder weights, labels, gradients, or production files entered training. Training took **58 seconds**; the full run including controls and held-out evaluation took 211 seconds.

The learner scored **zero true positives on the last-blank-frame → reappearance transition in all three training gap groups** (656 target-event pixels each). It also scored zero in every held-out gap-exit direction and duration. A trace reset removed its scattered gap-exit false alarms, but there were no correct events to preserve. The focused readout tests show that it can emit during blank input and that a future event correctly credits a previously active local trace source; the failure is not a missing code path for delayed updates.

| Full held-out next-event F1 | Fixed raw control | Online trace alone | Fixed + online trace |
| --- | ---: | ---: | ---: |
| Unseen single patterns | 0.971 | 0.034 | 0.900 |
| Independent movers | 0.977 | 0.061 | 0.914 |
| Crossings | 0.817 | 0.080 | 0.725 |
| Three-frame occlusion | 0.716 | 0.032 | 0.696 |
| Speed change | 0.893 | 0.073 | 0.831 |
| Noise | 0.798 | 0.048 | 0.757 |

The online candidate fails every gap-exit and non-regression gate and must not be promoted. This is especially informative beside the frozen supervised capacity bound of 0.768 training / 0.516 unseen three-frame F1 from the **same trace**: useful information is available, but this shared online local objective does not extract it. The broad 17×17 weights receive error from quiet, visible-motion, disappearance, and reappearance frames through overlapping trace sources; that is a plausible credit-interference mechanism, not yet proven by this result alone.

The next decisive diagnostic should fit evaluation-only local readouts from the same training traces with (1) gap-exit targets only, (2) gap-exit plus quiet targets, and (3) those plus ordinary visible/post-gap targets. If even the offline shared readout collapses when phases are combined, the representation needs separate local predictive phases or degrees of freedom. If it retains capacity, the online eligibility/update rule is the bottleneck. Do not tune the failed learner's learning rate or threshold through a narrow sweep. Production M1A remains unchanged.
