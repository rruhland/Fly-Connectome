# Reject the polarity-return observer teacher

**Decision:** Reject this registered teacher without changing its window or
adding exceptions. Requiring a same-pixel opposite-polarity return within eight
samples does not preserve useful visual evidence in this experiment.

The aligned contextual learner acquires **zero updates** and has effects [0, 0].
Aligned, shuffled and frozen states consequently have identical results:

| Check | Result |
|---|---:|
| Clean context top-32 | 4/16 |
| Familiar-noise context top-32 | 1/16 |
| Heavy-noise context top-32 | 2/16 |
| Constant-motion top-32 | 7/16 |
| Two-mover top-64 | 5/16 |
| Continued target strength >=.5 | 6/16 |
| Vanished false target strength >=.5 | 0/16 |

Rejecting false persistence while destroying continuing state is not success.
The result establishes a failure of this pseudo-label/observer combination; it
does not by itself identify the relative contributions of slow returns, static
cues, missed events or erroneous motion association. Do not claim that any one
of those explanations was demonstrated by this run.

Credit was causal: arrival features were buffered and updated only after eight
subsequent observed samples. No rendered frames or hidden labels were used.
Tests check delay, same-pixel polarity correspondence, and reset isolation.
The complete three-arm run took 114.61 seconds.

Reproduce: `.venv/Scripts/python scripts/run_polarity_return_teacher.py`.
[Raw results](2026-09-26-polarity-return-results.json).
