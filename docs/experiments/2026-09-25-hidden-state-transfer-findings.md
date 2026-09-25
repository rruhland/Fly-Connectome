# Hidden motion state transfers across patterns but lags abrupt speed changes

**Status:** Opt-in frozen evaluation under the [registered transfer protocol](2026-09-25-hidden-state-transfer-protocol.md). [Machine-readable results](2026-09-25-hidden-state-transfer-results.json). Production architecture unchanged.

The same one-pass local transition trained on 96 clean single-pattern episodes was evaluated without further learning on 48 generic robustness scenes. Next-frame target was the actually observed 24-unit latent code; the fixed persistence control held the previous observed code. No scene metadata entered the model.

| Held-out scene | Learned next-state occupancy F1 | Persistence F1 | Learned/target active sites |
| --- | ---: | ---: | ---: |
| Two independent patterns | **0.749** | 0.342 | 1.39× |
| Crossing patterns | **0.683** | 0.306 | 1.21× |
| Abrupt speed change, full sequence | **0.755** | 0.229 | 1.13× |
| Sensor noise | **0.568** | 0.345 | 1.98× |

Frozen zero-transition and episode-shuffled credit made no next-state predictions at the 0.5 threshold. The transition therefore learns reusable spatial motion at this controlled sampling rate, including two simultaneous patterns and crossings. Noise nearly doubles active sites, so its F1 gain should not be read as clean robustness.

The speed-change frames are less favorable than the full-sequence average: learned F1 is **0.878** at the change frame, **0.554** one frame after, and **0.443** two frames after (persistence **0.231 / 0.154 / 0.231**). It remains better than persistence but does not recover to its prior accuracy within the protocol's two-frame window. On eight paired disappearance scenes, the hidden-state occupancy F1 during the three gap frames is **0.541 / 0.415 / 0.297**, versus **0.301 / 0.158 / 0.158** for a static pre-gap hold. The first blank contains a disappearance observation, so frozen/shuffled controls can score there; only the learned transition continues afterward.

This passes the independent/crossing transfer thresholds but fails the complete generic-world-state gate because rapid speed-change recovery is unproven. Do not promote it yet. The next high-value test is zero-shot transfer to real Pong event-camera streams at the native frame rate and a coarser sampling control. The model must receive only the camera events; sampling rate is an evaluation condition. If native-rate correlations are too sparse, that is a front-end temporal-support issue to solve before another visibility head or production revision.
