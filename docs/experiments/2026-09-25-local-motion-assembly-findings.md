# Greedy local assembly linking did not learn a reusable motion state

**Status:** One-shot opt-in M1A architecture experiment under the [registered protocol](2026-09-25-local-motion-assembly-protocol.md). Production architecture, measured optic-lobe graph, and motor learning remain unchanged. No direction/object labels or backpropagation entered sensory coding, local Hebbian association, or assembly inference.

The model converted full event-camera frames into locally competing candidates from raw signed events plus the previously learned *observed* 24-unit sensory code. A shared 7×7 transition kernel learned local feature-pair/displacement co-occurrences across meaningful event frames from 64 unlabeled generic scenes. Sparse token assemblies retained their last feature and position through up to eight quiet frames and competed to claim new candidates. The same candidate streams fed a fixed proximity-only tracker and a temporally shuffled-credit kernel. No object regions were supplied during inference; synthetic paths were used only after inference to match and score output tracks.

| Held-out set | Learned direction / matched | Proximity direction / matched | Shuffled direction / matched |
| --- | ---: | ---: | ---: |
| Position, 16 objects | 16 / 16 | 16 / 16 | 16 / 16 |
| Speed ×2, 16 objects | **0 / 10** | **0 / 16** | **0 / 14** |
| Plus shape, 16 objects | 16 / 16 | 16 / 16 | 16 / 16 |
| Separated movers, 16 objects | 16 / 16 | 16 / 16 | 16 / 16 |
| Crossing movers, 4 objects | 4 / 4 | 4 / 4 | 4 / 4 |

The numerator is correct four-way direction among *all* objects; the denominator is objects with a matched track. At two pixels/frame, learned assemblies averaged only .292 path coverage, proximity .333, and shuffled .319. Many short tokens formed (18–20 per single-object trial) instead of one persistent moving assembly. All three rules handled the easy one-pixel/frame cases equally, so their perfect scores do not demonstrate learned association. Even the four crossing examples were solved by proximity and shuffled credit. Aligned Hebbian updates did find more local pairs than shuffled credit (4,404.5 versus 1,904.5 weighted pair mass), but that signal did not create a useful speed-transfer advantage.

**Decision:** Reject this greedy one-candidate/one-token implementation; do not tune its radius, persistence, or affinity gain. It has not learned a reusable motion representation, and it does not meet the protocol's requirement to beat controls on multi-entity cases. The central failure is assignment ambiguity: nearby leading/trailing event fragments can repeatedly claim a token, so local pair affinity and nearest distance both produce fragmented or stationary histories at faster motion. The previous oracle-localized audit showed that the sensory code *does* contain direction at speed two; the loss occurs while binding evidence into persistent assemblies.

The next architecture should let an assembly maintain **multiple local evidence hypotheses** over a short interval and learn to integrate their collective spatiotemporal pattern before committing to a single trajectory. Competitive assemblies could use local recurrent excitation/inhibition and delayed confirmation from future sensory evidence, with learned coherence across neighboring event fragments. This is a different representation problem from changing a scalar matching weight. A useful next experiment must compare it with fixed proximity and time-shuffled learning on speed transfer and multi-entity scenes, with no oracle regions at inference; only then test zero-shot native streams and consider a production proposal. T4/T5 measured-graph activity remains a potential complementary input after an assembly mechanism works.

Reproduction: `python scripts/run_local_motion_assemblies.py`; [raw results](2026-09-25-local-motion-assembly-results.json) include every object decision, matched coverage, source density, controls, and elapsed time.
