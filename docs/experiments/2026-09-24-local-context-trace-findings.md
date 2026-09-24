# Local context is present, but online credit still ignores it

The [registered opt-in trace test](2026-09-24-local-context-trace-protocol.md) retained the 24-unit learned coincidence population and added an eight-unit locally learned context population. Its input combined raw first-sighting events with a decaying, spatially pooled trace of learned motion activity. The context dictionary received two local Hebbian passes and its predictive synapses one local delayed-error pass over 96 clean plus 32 generic interruption/noise episodes. The spatial reach radius was fixed before training. The test took 154 seconds; production code did not change.

| Held-out online next-event F1 | Archived learner | Trace context | Zero-trace control | Trace reset at t=10 |
| --- | ---: | ---: | ---: | ---: |
| Unseen single patterns | 0.669 | **0.668** | 0.668 | 0.668 |
| Independent movers | 0.681 | **0.680** | 0.680 | 0.680 |
| Crossings | 0.583 | **0.583** | 0.582 | 0.583 |
| Disappearance, all frames | 0.430 | **0.430** | 0.424 | 0.429 |
| Speed changes | 0.634 | **0.633** | 0.634 | 0.633 |
| Sensor-bit noise | 0.487 | **0.487** | 0.487 | 0.487 |
| Exact reappearance t=10→11 | 0.000 | **0.103** | 0.048 | 0.093 |

The trace-bearing context input was present at all **116** raw reappearance event sites. Clean prediction and quiet-target false alarms were preserved: 98 false-positive pixels versus 97 for the archived learner. The trace candidate predicted 9 of 124 exact reappearance target events, with 42 false positives. The zero-trace and reset controls predicted 4 and 8, respectively. The trace-versus-reset margin was only **0.010 F1**, far below the registered 0.10 gate.

The matched counterfactual makes the learning bottleneck sharper. Opposite pre-gap motions gave the trace pathway substantially different local input codes (92.256 total absolute difference), yet its next-event forecasts were **identical**. The zero-trace and reset controls were identical too. Thus temporal context reaches the model, but these locally trained predictive weights do not express it. This experiment does not prove the code is sufficient to predict the future; code difference alone may be irrelevant. A targeted frozen-code readout trained on generic interrupted examples and tested on held-out shapes can resolve that remaining question. If the code is decodable, change the local credit/objective; if not, change the temporal representation. Do not sweep decay, radius, unit count, or threshold for this failed variant, and do not promote it.
