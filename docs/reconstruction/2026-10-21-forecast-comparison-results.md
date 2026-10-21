# One-pair state/forecast comparison

The paired development comparison found no clear forecast advantage. FULL minus
state-only was **−4.69 pp [−12.50, 1.56]** for first returns and
**+4.13 pp [−3.57, 11.60]** for all-opportunity return fraction. Both intervals
cross zero. This completes the prospective ablation; it establishes neither
equivalence nor an across-training-seed effect.

| Frozen policy | Initial first returns | Phase0 first returns | Endpoint first returns | Endpoint all returns |
|---|---:|---:|---:|---:|
| FULL (third unchanged seed) | 18/64 | 32/64 | 35/64 | 55/113 |
| State-only | 18/64 | 30/64 | 38/64 | 49/110 |

Both endpoints improve over their own initial policies: FULL first/all lifts
**26.56/24.28 pp**, paired 95% CIs **[9.38,43.75]/[12.60,35.75]**; state-only
**31.25/18.35 pp**, CIs **[15.63,46.88]/[7.49,28.75]**. These are secondary
comparisons, not substitutes for the registered strongest-control screen.

Both fail the unchanged primary requirement: at least 20 pp point lift on BOTH
metrics over own initialization plus five matched simple controls, with both
paired CI lower bounds above zero. FULL first/all lifts were **26.5625/19.9369342 pp**;
its all-return lift is strictly below 20 pp and receives no rounding tolerance.
State-only gave **31.25/15.81 pp**. Earlier two qualifying FULL seeds remain intact;
this third seed has positive learning evidence but does not become a third pass.

Both endpoint-minus-blank effects are positive, but this is whole-input OOD
sensitivity, not forecast-specific necessity or proof of optimal tracking.
There were no score wins against the .65 opponent. The ten-second horizon also
limits possible repeated returns; maximum three returns is not a capacity ceiling.

FULL retains the successful original transform exactly. State-only zeros all
raw forecast basis columns 1792:3848 BEFORE centering/whitening, leaving current
metadata and carried tracker beliefs intact. Supported ranks are 183/103
(366/206 action contrasts), so this compares complete declared representations
and their conditioning, not just removing a prediction at inference. Old unlabeled
calibration is retained for both; the prior repaired-row mismatch remains explicit.

Training budgets, optimizer settings, task schedule and the prospective shared
64-serve cohort were fixed before either run. Training roots/draw counts start
matched, but state-dependent costs and phase completion can change later streams.
No final holdout streams, backpropagation or privileged policy inputs were used.

FULL: 199,702 training cameras =186,074 used +13,628 unused; 97,513 evaluation
cameras; 45.64 minutes. State-only: 199,688 =173,040 used +26,648 unused;
47,786 newly executed evaluation cameras, plus 47,496 previously executed FULL
simple-control cameras referenced with verified provenance; 40.72 minutes.
Reuse contributes zero new state-only observations or claimed compute.

Actual zero-work recovery and all completion pins passed for both arms, with
every recorded state/chain and endpoint/RNG boundary unchanged. FULL independent
physics/optimizer/statistics audit passed 140.02 s (297,215 rows); STATE passed
119.65 s (247,474 own rows plus verified references to FULL physics). Both
optimizer, raw-physics, seed, annotation and paired-statistics audits passed.
Own verification took 13.46/13.54 s, with zero new transitions or replay.

The next allocation is the separately planned fixed-policy persistent-vision
integration/drift study. Do not spend more seeds seeking forecast superiority
from this one inconclusive pair; prioritize learning on a second visual task.

[Protocol](2026-10-21-forecast-comparison-protocol.md) · [Shared results/chart](/workspace/m1b-experiments/pong-forecast-comparison-20261021-r1/forecast-comparison.png) · [FULL audit](/workspace/m1b-experiments/third-full-parent-audit-20261021/summary.json) · [STATE audit](/workspace/m1b-experiments/state-only-parent-audit-20261021/summary.json)
