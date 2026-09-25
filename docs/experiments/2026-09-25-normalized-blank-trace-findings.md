# Local trace normalization trades one gap duration for another

The [registered evaluation-only control](2026-09-25-normalized-blank-trace-protocol.md) divided the 24 trace channels by their total at each nonzero retinotopic site, refit the same frozen 17×17 local linear readout on the same 64 training cases, and evaluated unchanged held-out scenes. This preserved active sites, channel ratios, target reachability, and all learned M1A weights.

| Gap-exit future-event F1 | Raw trace | Locally normalized trace |
| --- | ---: | ---: |
| Training three-frame gap | 0.768 | 0.697 |
| Unseen three-frame gap | **0.516** | **0.391** |
| Unseen early two-frame gap | 0.262 | 0.161 |
| Unseen late four-frame gap | 0.091 | **0.228** |

Normalization rescued some long-gap activity (34→91 true positives over the 48 late-gap cases) but weakened familiar and early-gap predictions. It misses all three registered joint-transfer thresholds: familiar ≥0.45, early ≥0.35, late ≥0.20 (only late passes). The result suggests that absolute trace strength carries useful timing information even as it causes late-gap decay; simply discarding it cannot provide a stable general forecast. The 17×17 field also still cannot reach 19.1% of four-frame target events.

Stop tuning the scale of this frozen trace. A bounded opt-in architecture experiment should now give learned local state a transition that can remain active and move through blank frames, with one-step online local predictive credit on visible sequences and direct gap-exit evaluation. Compare a frozen/untrained transition and state reset; measure false alarms and visible-motion regression. This is an experiment only—no production M1A architecture has changed or been approved.
