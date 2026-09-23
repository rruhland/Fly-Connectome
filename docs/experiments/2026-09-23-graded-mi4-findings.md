# Opt-in graded Mi4 transmission stops at the flash calibration

The experimental [graded-release kernel](../../scripts/graded_mi4.py) acts only on the 4,228 measured feedforward Mi4→T4 edges. It preserves their inhibitory transmitter sign, one-tick measured delay, existing weights, and all other network behavior. A presynaptic Mi4 membrane hyperpolarization reduces a bounded, nonnegative release level; release never becomes excitation. The in-place model and plasticity were unchanged. The [frozen timing audit](2026-09-22-motion-stage-timing-findings.md) set the voltage normalization to 0.0042 **before** motion tests.

The analytic no-event estimate of tonic release underestimated network feedback: all three initial candidates exceeded the predeclared 0.05 p99 added-T4-current bound. We corrected each candidate's release baseline using only no-event settling, targeting 0.045 for headroom, then repeated matched ON flashes at x=18 and x=46. No motion probe, direction label, or Pong score informed calibration. The final [machine-readable calibration](2026-09-23-graded-mi4-calibration.json) shows:

| Release fraction of flash scale | Blank p99 T4 current | Peak local mean absolute flash-current change x=18 / x=46 |
| ---: | ---: | ---: |
| 0.25 | 0.045 | 0.00187 / 0.00302 |
| 0.50 | 0.045 | 0.00313 / 0.00448 |
| 1.00 | 0.045 | 0.00425 / 0.00522 |

All states were finite. The preregistered minimum was **0.005 at both locations** under a blank p99 of at most 0.05. No candidate passed; the x=18 response was short even at the largest permitted fraction. Per the protocol, no release fraction was selected, and no graded moving-bar, shuffled-voltage, held-out, or Pong run was made. This is a failure of this *bounded additive graded-release approximation*, not evidence that graded Mi4 transmission is biologically absent. The point-neuron current rule, small and delayed Mi4 voltage change, and other active T4 inputs remain distinct explanations. The next opt-in test isolates the active-input conductance/compartment hypothesis without relaxing this gate.
