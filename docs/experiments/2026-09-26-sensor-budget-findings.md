# Accurate appearance rescues the clean surface state under event noise

**Decision:** For this bounded contextual benchmark, prioritize sensory
reconstruction before further world-state or credit-rule changes. Neither the
event-only nor sparse-frame candidate passes. Exact per-frame visible appearance
provides a positive state-capacity control, not M1A.5 acceptance.

The surface/context architecture, observer teacher, training split and state
settings were held fixed. At image arrivals, contrast was anchored to the current
visible image minus its spatial median. No hidden mask, true background label,
future image or privileged intensity-training teacher was supplied. The image
sensor itself was noiseless in this synthetic comparison.

| Declared input | Clean context /16 | Familiar noise /16 | Heavy noise /16 | Constant /16 | Two entities /16 | Continued /16 | Vanished false /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Events only | 16 | 5 | 3 | 16 | 16 | 16 | 0 |
| Events + visible image every 8 frames | 16 | 9 | 7 | 16 | 16 | 16 | 0 |
| Events + visible image every frame | 16 | 16 | 16 | 16 | 16 | 16 | 0 |

Context uses top-32; two entities uses top-64. Every-frame aligned also scores
16/16 top-8 on all three context conditions. Its shuffled and frozen contextual
controls score 4/16 top-32 and 0/16 top-8 in each condition. Learning therefore
adds useful contextual information even when current visible appearance is
accurate. Matched continue/disappear predictions remain identical before reveal.

This intervention establishes that the unchanged state can express the useful
learned relation under event noise **when appearance is externally corrected**.
It does not identify every event-only failure cause, prove sufficiency for
general vision, or prove event-only reconstruction can recover identical
information. Noise may erase evidence that no inference can recover perfectly.

Sparse images add 256 image-pixel samples per camera step on average; every-frame
images add 2,048, on top of event input. Neither is free, neither is event-only,
and no real-camera noise or exposure effects were tested. Median anchoring also
assumes a dominant background in these scenes and is not a general illumination
solution. The prior .35-blended sparse arm is separately retained; this run uses
actual hard sensor observations at the declared arrivals.

One follow-up is registered: learn event credibility from locally observed
polarity returns with delayed causal credit, then rerun the unchanged event-only
surface state. It tests a different sensory teaching principle; no window sweep
or additional association tuning is authorized by these results.

Reproduce: `.venv/Scripts/python scripts/run_sensor_budget.py`.
[All six result arms](2026-09-26-sensor-budget-results.json) took 139.19 seconds
including setup, training, inference and scoring. Full suite after this chunk:
648 passed, 4 skipped. Causality tests verify that hidden targets/background
metadata are ignored and changed future frames cannot affect prior states.
