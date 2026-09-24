# Matched diverse visual experience: modest transfer, no working forecast

The two opt-in circuits received the same 96-episode budget at each training phase. Square-only training and varied square/plus/L/bar training used identical sensory, transition, and local predictive rules. The 48 held-out diamond/ring/zigzag cases changed shape and center. No object, motion, or shape labels entered either circuit. All held-out next events remained inside the existing local readout field.

| Held-out measure | Square only | Varied experience |
| --- | ---: | ---: |
| Learned transition-code direction probe | 32/48 | 30/48 |
| Local predictive-error event F1, threshold 0.5 | 0.198 | 0.223 |
| Offline learned-code event F1, threshold 0.5 | 0.296 | 0.324 |
| Offline random-code event F1, threshold 0.5 | 0.196 | 0.160 |
| Fixed sensory correlation event F1 | 0.971 | 0.971 |

Varied experience did not meet the registered 0.50 absolute or +0.15 improvement gate. Its training-set offline F1 was 0.476, versus 0.814 for square-only training, and its local-rule training F1 was 0.238 versus 0.319. The varied arm is underfit even on its own training distribution. An exploratory common threshold of 0.3 gives 0.440 versus 0.326 held-out decoder F1, so part of the threshold-0.5 comparison reflects calibration; this does not rescue slow motion. At one pixel/frame, varied offline F1 was 0.202 versus 0.290 square-only and varied local F1 was 0.055 versus 0.178. At two pixels/frame, the varied arm improved to 0.392 versus 0.300 offline and 0.312 versus 0.211 local.

This is modest evidence that shape-diverse exposure helps some transfer, especially at faster motion, but not evidence of a reusable predictive M1A state. The equal-budget design means each varied pattern received far fewer repetitions than square. A single larger-exposure diverse-arm test is justified to separate insufficient examples from a representation/readout limit. Keep the architecture, control, and held-out set fixed; if training fit and slow-motion transfer remain poor, move to a different local temporal representation rather than tuning more thresholds. No production M1A code changed.
