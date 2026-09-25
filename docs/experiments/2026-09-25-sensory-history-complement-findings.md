# Learned history adds missing events but naive full-scene fusion fails

The [registered frozen-weight experiment](2026-09-25-sensory-history-complement-protocol.md) combined a fixed raw-event correlation forecast with the online-learned local history forecast by pixelwise maximum. It reused the saved balanced model without retraining and scored complete unseen generic sequences, plus early two-frame and late four-frame interruptions. The fixed control is a hand-specified low-level sensory primitive used only as a comparator; it is not the proposed learned M1A representation.

| Full next-event sequence F1 | Fixed raw | Fixed + learned history | Fixed + random history | Archived learned correlation + history |
| --- | ---: | ---: | ---: | ---: |
| Unseen single patterns | 0.971 | 0.971 | 0.971 | 0.669 |
| Independent movers | 0.977 | 0.977 | 0.977 | 0.681 |
| Crossings | 0.817 | 0.814 | 0.816 | 0.581 |
| Three-frame occlusion | **0.716** | **0.717** | 0.710 | 0.444 |
| Speed changes | 0.893 | 0.893 | 0.893 | 0.634 |
| Noise | 0.798 | 0.796 | 0.800 | 0.486 |
| Early two-frame gap | 0.752 | 0.755 | 0.749 | 0.519 |
| Late four-frame gap | 0.678 | 0.687 | 0.676 | 0.472 |

At exact reappearance the fixed raw primitive still had zero true positives; fixed plus learned history reached 0.427 F1 after the early gap and 0.380 after the late gap, while fixed plus random reached only 0.154 and 0.105. The learned component therefore carries genuine complementary information. But across complete three-frame occlusion sequences, fusion gained **48 true positives and 81 false positives** over fixed alone (682/124 TP/FP to 730/205). Full-sequence F1 improved by only 0.001, far below the registered +0.05 gate. Early and late gap sequences likewise improved only 0.003 and 0.009. Gap-quiet false-alarm pixels rose from 364 fixed to 430 and 432 fused on the new windows; this remains below the predeclared 1.5× ceiling, but the additional errors cancel much of the extra event coverage. Crossings and noise regress slightly. The candidate fails the principal full-scene gate and must not be promoted.

This rejects a simple additive output merger as the path to M1A. It does **not** reject the locally learned history state: history reset removes its exact post-gap successes, and both changed-gap windows transfer. The next decisive question is whether locally available source/trace/sensory state distinguishes *useful* history emissions from the added false alarms in full scenes. If it does, test a simple locally learned reliability or competition mechanism; if it does not, improve the temporal state and training objective. Avoid hand-coded Pong semantics, hard-coded motion identity, or a threshold sweep. Production M1A is unchanged.
