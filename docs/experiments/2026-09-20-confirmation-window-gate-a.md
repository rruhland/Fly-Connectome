# Confirmation-window Gate A: failed

The approved prerequisite did not pass: **zero of the three fixed connections
passed on both fresh development seeds**, versus the required two. The learning
rule was not implemented and no new training was started. This follows the
proposal's explicit conditional stopping rule; thresholds and candidates were
not changed after seeing results.

## What was tested

Use the unchanged 10,000-frame checkpoint on seeds 1103/1104 for 2,000 scripted
frames. Retain frame-end decisions [12, 1988), matching the preceding trace audit.
The target is the first subsequent nonzero local input within 12 frames, zero if
none; omit incomplete tail windows. No current-frame event confirms itself.

At each decision, construct the local history from already observed input:
last nonzero polarity and its age capped at 12 frames, with a distinct never-seen
state. Ten permutations (seeds 421-430) shuffle target labels only within the same
history state, separately for each development seed and postsynaptic neuron.
This preserves the chosen local-history association while disrupting additional
trace timing. It is a coarse conditional control, not a fitted readout or a
formal significance test.

The candidate pairs were fixed before this run. Passing required at least five
unique referenced future events, trace power >1e-10, correlation >=0.05, positive
target-times-trace moment, and covariance above all ten conditional shuffles, on
both seeds. The same two pairs had to pass both seeds; successes could not be
combined across different pairs.

## Results

Correlation and covariance comparisons agree in rank because the permutations
preserve each target's distribution and leave traces unchanged.

| L2 -> L1 body IDs | Seed | Distinct events | Actual correlation | Conditional-shuffle correlation range | Shuffles beaten | Result |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 50062 -> 51452 | 1103 | 41 | 0.11783 | 0.13983 to 0.17819 | 0/10 | Fail |
| 50062 -> 51452 | 1104 | 80 | 0.16091 | 0.16406 to 0.21404 | 0/10 | Fail |
| 75482 -> 71110 | 1103 | 8 | 0.06498 | 0.04855 to 0.12039 | 4/10 | Fail |
| 75482 -> 71110 | 1104 | 18 | 0.13523 | 0.12325 to 0.15042 | 3/10 | Fail |
| 42416 -> 44723 | 1103 | 20 | 0.04417 | -0.02328 to 0.02777 | 10/10 | Fail: correlation <0.05 |
| 42416 -> 44723 | 1104 | 14 | 0.07329 | 0.02227 to 0.09247 | 7/10 | Fail |

All six comparisons had sufficient target-event counts, positive target/trace
moments and nontrivial trace power. Between 98.58% and 100% of decision windows
belonged to history groups containing more than one target label, so label
permutability was not the limiting factor. This is a failed prerequisite rather
than an inconclusive result from inadequate event counts or constant strata.

The first pair's raw correlations replicate and even increase on one new seed,
yet every conditional shuffle exceeds them. The second also retains raw
correlation but fails the conditional comparison. The third beats all shuffles
on one seed, but misses the preregistered correlation threshold there and fails
the shuffle comparison on the other seed. None justifies relaxing the gate.

The JSON contains all stratum sizes, label counts, shuffle values, and zero,
current-frame persistence, last/opposite-polarity controls with and without
100 ms expiry. Those MSE values refer only to the fixed three target neurons;
they must not be presented as network-wide performance. A tiny improvement over
zero at an individual target does not override the information-control gate.

## Consequence

The prior raw correlations were an exploratory lead, not proof of additional
predictive information. This stricter test does not establish that the traces
add information beyond the chosen local event-history description. It does not
prove that all information is explained by history, that every temporal rule
would fail, or that the connectome cannot learn.

The approved confirmation-window proposal stops here. Its conditional Gate B
implementation and Gate C training are not authorized to proceed after this
failure. Final seeds remain untouched, and M1A remains unmet. No source weights,
topology, signs, dynamics, learning rules or acceptance goals were changed.

The next discussion should revisit the modeled local prediction target and its
relationship to circuit activity, rather than continue horizon/eligibility sweeps.
In particular, the existing recurrent current serves both as a forecast and as
physical drive: earlier C2 experiments showed that restoring drive rescued firing
without rescuing prediction. A new proposal should distinguish representational
adequacy, genuinely incremental sensory information and learnability on a controlled
task before another large-graph learning run. This report implements no such
architecture or goal revision.

## Verification and reproduction

```powershell
.venv/Scripts/python scripts/audit_history_gate.py checkpoints/event-v1-combined-rate-10000.pt --output docs/experiments/2026-09-20-confirmation-window-gate-a.json
.venv/Scripts/python -m pytest -q
```

The collector reconstructs full predictive current into the three target neurons
from all their measured incoming predictive edges; only the fixed candidate traces
enter the control analysis. Maximum reconstruction error: 4.66e-10. Checkpoint,
graph and configuration hashes are in the JSON and match the previous model;
source checkpoint bytes were verified unchanged. Warmup is retained normally.

Tests cover causal history, within-stratum distribution preservation, constant
history groups, the same-pair/both-seed gate, frozen collection and reconstruction.
Fresh independent review found no actionable defects. Full suite: **198 passed,
four CUDA skips**. All experiment processes have finished.
