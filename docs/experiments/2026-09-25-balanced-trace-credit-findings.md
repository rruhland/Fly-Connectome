# Balanced local trace credit improves ranking but floods full scenes

The [registered opt-in experiment](2026-09-25-balanced-trace-credit-protocol.md) kept the saved learned motion/trace code and 17×17 emission field, but separately normalized each source's event-confirming and false-alarm-suppressing local updates. It trained one balanced model and a matched shuffled-credit model for one pass over the same 288 generic episodes. No gradient, direction/object label, offline decoder weight, or production file entered training. Both models trained in 159 seconds; the complete run took 241 seconds.

| Held-out gap exit | Balanced F1 | Balanced event/quiet AUC | Shuffled F1 | Shuffled AUC | Original shared-error AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-frame | 0.107 | **0.873** | 0.099 | 0.784 | 0.814 |
| Three-frame | 0.052 | **0.859** | 0.083 | 0.710 | 0.757 |
| Four-frame | 0.021 | **0.704** | 0.062 | 0.507 | 0.614 |

The balanced local update raises ranking at every duration and beats shuffled credit by 0.089–0.197 AUC, showing causal use of the learned trace. It does **not** convert that signal into precise signed events at the registered 0.5 threshold; on training gap exits it reaches only 0.078, 0.029, and 0.008 F1. Shuffled credit sometimes has higher thresholded F1, so the learned score improvement cannot be presented as effective event prediction.

The broad additive emitter is much worse on complete scenes. Fixed raw plus balanced history scored 0.213 F1 on unseen single patterns versus 0.971 fixed alone, and 0.234 on full occlusion versus 0.716 fixed alone. On the eight-scene occlusion suite, quiet false-alarm pixels rose from **62 fixed** to **1,540 fused** (2,550 for shuffled). This fails the non-regression and false-alarm gates by wide margins. Stop tuning this passive 17×17 trace emitter or its scalar threshold.

The evidence now supports separating **hidden visual motion state** from **visible-event emission**. A future opt-in model should learn a local spatial transition that propagates a compact latent state during blank input, while separately learning when visible sensory events are likely. Evaluate probabilistic timing/uncertainty as well as event F1, because variable-length occlusion without a revealing cue may make the exact reappearance frame intrinsically ambiguous. Use a frozen transition, state reset, shuffled credit, and quiet-frame controls. No in-place M1A architecture has changed or been approved.
