# Frozen learned visual state transfers to multiple movers and rate changes

The robustness helper retrained the opt-in learned correlation-input circuit from the same seed and reproduced its original held-out event F1 **exactly: 0.669273**. It then froze all weights and evaluated 48 new generic scenes, with no new labels or learning. Results use the same next-frame target and threshold 0.5.

| Scene family | Learned local state | Random dictionary + local credit | Fixed sensory correlation |
| --- | ---: | ---: | ---: |
| Two independent movers | **0.681** | 0.238 | 0.977 |
| Crossing movers | **0.583** | 0.226 | 0.817 |
| Speed change | **0.634** | 0.210 | 0.893 |
| Brief disappearance | **0.430** | 0.147 | 0.716 |
| Sensor-bit noise | **0.487** | 0.147 | 0.798 |

The registered gates for independent movers and speed changes pass, each by more than a 0.42 margin over the matched random dictionary. At the rate-switch frame alone, the learned forecast scored **0.526** versus 0.161 random and 0.615 fixed. At disappearance/reappearance frames it scored **0.288** versus 0.155 random and 0.348 fixed. The whole suite completed in 32 seconds after training from scratch on this host.

The learned local representation is no longer confined to one shape at one speed, and its predictive synapses have a causal advantage over shuffled credit from the preceding experiment. Yet the fixed low-level sensory forecaster remains substantially better at next-event F1 in every family. The weakest conditions are exactly where a code selected only at current correlation-event sites loses information: hidden evidence and noise. The current predictive output does not feed back into latent inference, so it cannot maintain a coherent local state while inputs are missing. Test an opt-in recurrent state that can persist and be corrected by new evidence, using the existing learned code and local prediction error, before production promotion. Evaluate whether it improves disappearance/noise **without reducing** independent-mover and speed-change performance. No production M1A code changed.
