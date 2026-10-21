# Fresh confirmation: visual dependence, insufficient acquisition evidence

The unchanged theta9 returned first balls in 34/64 fresh serves, close to its
earlier 16/32. Its original initialization was stronger on this cohort: 27/64.
Both primary lift intervals cross zero and the unchanged screen failed. No new
training or final holdout was used. Blanking the camera caused a clear drop,
supporting whole visual-input dependence, not established reward acquisition.

The [prospective protocol](2026-10-21-frozen-hit-confirmation-protocol.md) retained
the exact readouts, corrected frozen vision, DCT/cap, .65 opponent and ten-second
fresh-phase serves. The primary comparator maximum includes all six controls:

| Frozen policy | First incoming returns | All returns/opportunities |
| --- | --- | --- |
| Trained theta9 | 34/64 | 48/106 |
| Original theta0 | 27/64 | 34/95 |
| Constant down | 17/64 | 19/82 |
| Constant up | 14/64 | 16/79 |
| No-op | 14/64 | 16/79 |
| Random hold1 | 13/64 | 16/78 |
| Random hold10 | 16/64 | 18/82 |
| Secondary blank theta9 | 14/64 | 16/79 |

Against the strongest primary comparator, first lift was **+10.94 pp** (paired
95% interval **−6.25 to +26.56**); all-opportunity lift was **+9.49 pp**
(**−1.65 to +20.03**). Neither meets the fixed ≥20 pp/both-lower-bounds-positive
screen. The original 32-serve outcome remains separately failed; the cohorts are
not pooled for qualification. This is one trained seed, not a training replication.

The separately registered full-minus-blank contrast was **+31.25 pp** first
(**+14.06 to +48.44**) and **+25.03 pp** overall (**+12.44 to +37.60**). Blanking
never entered the primary maximum. It is out of distribution and removes the
whole visual input: these effects establish neither object/forecast-specific
benefit nor ongoing feedback/tracking rather than initial scene conditioning.
The endpoint produced twelve rallies with at least two returns versus seven for
initialization, maximum streak three versus two. Across all cohorts the opponent
returned 445/445 balls, including 181/181 after own returns; no natural wins.

All 512 jobs completed in **854.79 s** from the original clock, with **83,814
evaluation transitions and zero training**. The initial four tasks failed before
session construction because staging omitted required weight sidecars. The
explicit zero-camera amendment retained the original clock, all serves/readouts
and failed artifacts; no deadline was extended. Source **5c9e219**, nine distinct
focused checks and exact source archives are preserved in the
[artifact](/workspace/m1b-experiments/pong-frozen-hit-confirmation-20261021-r1/summary.json).

The [independent audit](/workspace/m1b-experiments/frozen-confirmation-parent-audit-20261021/summary.json)
passed all 83,814 physics rows, 512 jobs, seed/action-seed recipes, frozen readouts,
64 paired first trajectories and both exact bootstrap analyses in **69.56 s**.
[Zero-work recovery](/workspace/m1b-experiments/pong-frozen-hit-confirmation-20261021-r1/completion-verification.json)
added zero observations/replay, preserved all record/state/chains and both policy
files, and verified all 512 completion pins in **2.65 s**.

Next is a separately registered same-class continuation, retaining theta9 as a
secondary comparison. It will test additional reward learning with fixed paired
center/reference anchors, rather than change thresholds or infer reproducibility
from this one seed. Final streams remain unused and the broader M1B checklist open.
