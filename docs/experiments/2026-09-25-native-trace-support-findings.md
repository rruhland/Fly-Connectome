# Local event trace wakes native sensory code but does not transfer the transition

**Status:** Opt-in frozen diagnostic under the [trace protocol](2026-09-25-native-trace-support-protocol.md), with [results](2026-09-25-native-trace-support-results.json). No production change.

On the same four held-out 120 Hz Pong camera streams, a four-frame exponentially decaying local ON/OFF trace replaced the immediate previous-event frame in the existing one-/two-pixel coincidence channels. The trained 24-unit dictionary and recurrent transition remained frozen. Neither the input nor the model received Pong state or labels.

| Native-rate input | Code-active frames | Active-frame target sites | Learned next-latent F1 | Persistence F1 |
| --- | ---: | ---: | ---: | ---: |
| Adjacent frame | 33/464 (7.1%) | 33 | 0.000 | 0.000 |
| Four-frame local trace | 287/464 (61.9%) | 509 | **0.035** | **0.073** |

The trace confirms that the native front end lacked temporal support: it increases active code frames almost ninefold and raises average hidden activity on camera-quiet frames from 0.25 to 2.62 sites. But the unchanged transition does not forecast the newly available code; it scores below simply holding the last observed latent state. More input memory is therefore not a complete fix. The model was trained to advance on one-/two-pixel-per-frame generic trajectories, whereas native camera events often arrive after several subpixel physics frames.

Next test a genuinely event-indexed forecast using the same frozen model and trace input: advance the state only when a local code event exists and ask it to predict the next local code event. This changes the opt-in clock and the forecast target, so its F1 must be reported separately from frame-indexed F1. If that still fails, train local dictionary/transition dynamics on generic variable-cadence experience rather than making another threshold adjustment. No production architecture should be promoted yet.
