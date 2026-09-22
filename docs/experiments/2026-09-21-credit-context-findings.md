# Credit context: polarity is separable locally; quiet credit remains mixed

Follow-up: [actual-learning replay and omission results](2026-09-21-credit-replay-findings.md) show that the tag exactly tracks last polarity; event direction separates, but quiet credit remains mixed.

The saved frozen-baseline audit contains a simple local cue that distinguishes the opposing ON/OFF update directions on both examined predictive edges. The sign of the postsynaptic sensory current **at forecast time** correctly separates all examined nonzero ON/OFF proposals at every test tempo. At held-out dwell 3, that is 90 ON and 90 OFF cases per edge. This supports investigating context-dependent plasticity, but does not establish a working learning rule or sufficient forecast capacity. M1A remains unmet.

No new neural simulations or training runs were needed. No model, weights, topology, transmitter signs, local rule or production code changed. The existing signed sensory state supplies the cue; no future label is injected as a feature.

## What was measured

Source: the saved 30-trial-per-tempo mixed-baseline context audit, seed 9070, with original area-matched kinetics. This is **not** a replay of the D/F training trajectories, and contains no facilitation state. The two edges are excitatory L2 body 20655 and inhibitory L1 body 26550 into L3 body 82450.

For each edge, compute the normalized counterfactual proposal `(future_target - issued_prediction) * saved_eligibility` at the frozen weights. The common positive learning rate is omitted. Zero proposals are excluded from sign classification and counted separately. These are proposed local updates, not applied or clipped training updates.

FramePrediction captures the eligibility and prediction at issue and uses them eight ticks later. The source collector records local features immediately after that capture. Probe inputs contain only the same edge's eligibility/arrival history and/or postsynaptic state at issue; no other incoming edge's private trace, future observation, trial, tempo or phase is supplied. This is a test of context that could be tagged to the eligibility. It is not an exhaustive audit of all state available at confirmation.

Fit on the first 15 trials of tempos 2/4/6; test the last 10 trials of those tempos and all 30 trials of unseen tempo 3. Original calibration trials are unused. Scaling and boundary selection use training samples only. Compare a single-feature threshold and a fixed 15-neighbor probe. Twenty shuffled-label controls are retained for the neighbor probe, not a formal multiple-testing significance claim.

## Why ordinary update-sign accuracy was misleading

The initial count-balanced threshold in the postsynaptic/edge feature group achieved about 90% balanced sign accuracy at held-out tempo 3 while getting **0/90 ON proposals correct on both edges**. It mostly recognized quiet corrections. Its correctly classified share of absolute update magnitude was only 33.7% on L2 and 20.7% on L1. The nearest-neighbor probe also missed every held-out ON proposal.

After inspecting that failure, this exploratory audit added two training-only boundary objectives: equal total weight to ON/OFF/quiet categories, and weight by absolute proposal magnitude. These were offline probe objectives, not changes to network learning. The event-balanced boundary selected the existing postsynaptic `sensory_state` on both edges (same threshold -1.7260, opposite directions). A subsequent zero-threshold structural comparator reproduced perfect ON/OFF direction separation. These additions were not preregistered; fresh validation is required before adopting a mechanism.

## Zero sensory-current boundary

For L2, positive forecast-time sensory current predicts a negative update, and negative current predicts a positive update. For L1 the directions reverse. This uses the existing polarity/history of local sensory current, not a learned temporal gate.

| Tempo | Edge | ON direction correct | OFF direction correct | Quiet direction correct | Correct share of absolute update magnitude |
|---|---|---:|---:|---:|---:|
| 2 | L2 | 100.0% (30) | 100.0% (30) | 74.5% | 88.7% |
| 2 | L1 | 100.0% (30) | 100.0% (30) | 74.5% | 85.7% |
| 3 | L2 | 100.0% (90) | 100.0% (90) | 81.6% | 91.7% |
| 3 | L1 | 100.0% (90) | 100.0% (90) | 81.7% | 88.3% |
| 4 | L2 | 100.0% (30) | 100.0% (30) | 70.7% | 87.8% |
| 4 | L1 | 100.0% (30) | 100.0% (30) | 70.7% | 79.8% |
| 6 | L2 | 100.0% (30) | 100.0% (30) | 51.3% | 73.0% |
| 6 | L1 | 100.0% (30) | 100.0% (30) | 51.3% | 51.9% |

Quiet credit is not cleanly separated: approximately half of its directions are wrong at the slowest tempo. Quiet direction accuracy is **not** a forecast false-alarm rate. Perfect event-update direction classification is **not** an event detector: the same sensory sign persists during quiet frames before and after events. The alternating two-position stimulus makes previous polarity informative about the next polarity, so this result is especially limited to that task structure.

The JSON also reports aggregate cancellation before/after partitioning proposals into two predicted groups. Those descriptive sums are not achieved learning performance: partitioning can preserve the wrong groups and does not specify how predictions would be expressed.

## Interpretation and next investigation

The evidence favors a narrower statement than either 'the representation is sufficient' or 'one scalar is impossible': opposing event updates are locally distinguishable in this frozen baseline. The shared rule currently collapses that distinction when it accumulates proposals into one magnitude. Opposing updates by themselves are normal in error minimization; they do not prove irreconcilable objectives. Tiny correct-sign outputs before training also do not prove sufficient capacity for strong, sharply timed forecasts.

The next useful bounded diagnostic is to record this same cue at issue and confirmation during a short, exact replay of baseline and facilitation learning, checking that the tag remains separable as weights move and that summed captured proposals match the actual accumulator. Include quiet phases and a fresh sequence with omitted/repeated transitions so previous polarity is not automatically equivalent to the next event. No extended training sweep is warranted yet.

If that survives, a context-dependent plasticity proposal is justified. Distinct eligibility channels that still sum into one magnitude can change which examples influence the compromise, but do not by themselves create independently retained predictions. A proposal for separate synaptic components must specify how the **forecast-time local cue** selects both credit and expression; routing only by the observed target at confirmation would not establish a causal forecast mechanism. These are future design questions, not approved or implemented changes.

Failure of a simple probe would also not prove that context is absent. The present result does not justify adding an interval clock, a hardcoded gate, or promoting STP to M1A.

## Verification and reproduction

Source checksums match the original audit manifest. Saved targets equal raw observations at issue +8 ticks, and predictions equal raw predictions at issue. Independently checked forecast sensory increments match the contemporaneous observation, not the future label. Analytical checks verify known-boundary selection and the distinction between sign counts and update magnitude. Relevant regression checks: **10 passed** (context audit and STP comparison suites). No production code changed.

```powershell
.venv/Scripts/python scripts/credit_context_audit.py
.venv/Scripts/python -m pytest tests/test_context_audit.py tests/test_short_term_comparison.py -q
```

- [Results, boundaries, per-category counts and source hashes](2026-09-21-credit-context-results.json)
- [Source/alignment verification](2026-09-21-credit-context-verification.json)
- [Original context audit](2026-09-21-context-audit-findings.md)
- [STP learning findings](2026-09-21-short-term-learning-findings.md)
