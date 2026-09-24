# The learned context dictionary collapses opposite histories

The [registered frozen-code capacity check](2026-09-24-context-trace-capacity-protocol.md) fit an evaluation-only 5×5 local linear readout at reappearance on 64 generic interrupted sequences from four training shapes, four directions, two speeds, and both contrasts. It tested eight held-out disappearance cases from unseen diamond/ring shapes. The same two locally trained context populations were recreated with the prior seeds and data; no offline weights entered M1A or production code. This run took 91 seconds.

| Exact t=10→11 offline F1 | Learned trace code | Learned zero-trace code | Trace code with history reset |
| --- | ---: | ---: | ---: |
| Readout training cases | 0.480 | 0.464 | 0.473 |
| Held-out unseen shapes | **0.458** | 0.467 | 0.437 |
| Matched opposite-motion pair | 0.556 | 0.500 | 0.556 |

All held-out target events were reachable from each code's 5×5 readout fields, so the result is not spatial reach. The trace code did **not** beat its zero-trace control on held-out shapes, much less reach the registered 0.15 margin. More decisively, the matched pair gave the learned trace population **identical winner sources** for both opposite histories (zero differing sources), despite the trace-bearing input codes being different in the previous experiment. Its offline forecasts were therefore identical as well (zero total absolute forecast difference), and its 0.556 pair F1 does not indicate direction-sensitive prediction. Resetting the trace at reappearance did not change the pair's selected sources.

This separates two losses: the low-level learned motion trace carries history, but the current eight-unit concatenated raw+trace competition discards the distinction before it reaches predictive synapses. Replacing only the scalar predictive update cannot recover a difference that is absent from the latent sources. Do not tune this dictionary's unit count, decay, radius, or threshold through a sweep. The next architecture experiment should keep raw events as a **gate** and train history-selective local units on the trace itself, then require opposite histories to select different units before spending another full forecast-training run. That is a substantive change to the opt-in temporal representation, not a production revision.
