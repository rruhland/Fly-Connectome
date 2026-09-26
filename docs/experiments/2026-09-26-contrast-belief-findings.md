# Contrast-transition beliefs do not improve the state gate

**Decision:** Reject this replacement as a route to M1A.5 acceptance. Do not
sweep priors, state counts, thresholds or clipping behavior.

The original causal event-agreement observer was preserved, including its
internal contrast feedback. Only exported contrast was replaced by the mean of
three-state per-pixel transition beliefs. Thus this experiment changes sensory
inference, not the teacher, contextual objective or association architecture.

Aligned results remain 16/16 clean context, 5/16 familiar noise, 3/16 heavy noise,
16/16 constant motion, 16/16 two-mover top-64, 16/16 continuing reacquisitions and
0/16 clean vanished-target false positives. These are the same scores as the
baseline surface candidate. Its learned effects and observed credit counts are
also unchanged. Retaining probability mass is mathematically coherent, but this
run gives no evidence that exporting its mean improves usable state.

The three-level model is only a binary-contrast simulator approximation. Its
normalized beliefs are not evidence of calibrated world-state uncertainty; a
downstream scalar contrast mean also discards much of that uncertainty. Neither
claim is made here. Tests establish mass conservation, causal signed returns,
preservation under empty input and symmetry under conflicting evidence.

Reproduce: `.venv/Scripts/python scripts/run_contrast_belief.py`.
[Raw results](2026-09-26-contrast-belief-results.json) include aligned, shuffled
and frozen controls and total runtime. Close this batch rather than extend it
with further local filters. The positive dense-appearance control and failure
of event-only observation repair now motivate a broader observation-model
comparison before another prediction mechanism.
