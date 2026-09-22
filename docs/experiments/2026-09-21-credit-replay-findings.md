# Credit replay: real polarity context, unresolved event-versus-quiet credit

The polarity cue survives actual learning, but the follow-up sharply limits what that means. At every recorded forecast, its sign equals the last observed sensory-event polarity. The scored column contains one binary pixel, so its nonzero events must alternate. Thus the earlier perfect ON/OFF separator is a local occupancy/history signal, not evidence that the network has learned timing or that the credit rule is the sole bottleneck.

Quiet and event proposals still oppose each other inside the same polarity context. There is no evidence here that a polarity-only channel split would solve prediction. M1A remains unmet; no new plasticity mechanism was implemented.

## Bounded protocol and exact checks

A passive observer recorded forecast-time eligibility, prediction, selected weights, and presynaptic/postsynaptic state, then recorded confirmation-time state and the actual local proposal eight ticks later. It did not change transmission, learning, synchronization, signs or topology. Tests require identical state, proposals and weights with/without the recorder on baseline and facilitation fixtures.

For baseline and F separately, replayed the first 24 trials of the original 200-trial seed-9060 schedule, starting from original weights and normal warmup. Predictions, targets, spikes and issued eligibility matched the saved training prefixes **bit-for-bit**. At every recorded confirmation, the two selected-edge proposals matched the actual accumulator with zero error. Each selected weight after synchronization matched the unchanged clipping/homeostasis calculation exactly. All recorded states were finite.

Then ran ordinary and omission sequences from each model's existing 200-trial weights with fresh normal warmup/state: 16 trials, seed9091, four each at dwells2/3/4/6. Learning remained active. Omission holds the non-target position through one scheduled ON/OFF pair. The zero sensory-state separator from the prior audit was used unchanged, with no refitting. These are diagnostic learning trajectories, not frozen before/after acceptance tests.

The planned repeated-polarity test was stopped at stimulus preflight: only pixel(30,39) maps to scored column30 at32x64. A second independently activated pixel in that column does not exist. Skipping a transition cannot create ON->ON at a single binary pixel. No camera, resolution, retinotopy, neuron or edge was changed to make the test run. The generic two-pixel stimulus has a synthetic test, but there is **no real-data repeated-polarity result** on this target.

## Cue stability during learning

To compare with the previous audit, this table excludes the first motion cycle, using the same recurrent-target convention. Counts are for nonzero proposals; all-frame, early/late and per-tempo counts remain in the JSON.

| Run | ON direction correct, each edge | OFF direction correct, each edge | Quiet direction correct, L2 | Quiet direction correct, L1 |
|---|---:|---:|---:|---:|
| baseline-replay | 72/72 | 72/72 | 63.5% | 63.6% |
| F-replay | 72/72 | 72/72 | 63.9% | 63.9% |
| baseline-standard | 48/48 | 48/48 | 64.4% | 64.6% |
| F-standard | 48/48 | 48/48 | 65.7% | 65.8% |
| baseline-omitted | 32/32 | 32/32 | 60.9% | 61.1% |
| F-omitted | 32/32 | 32/32 | 61.5% | 61.7% |

All 6,122 recorded issue-time tags (3,061 per model) matched the last observed polarity exactly, including zeros before the first event. Weights changed substantially during the 24-trial replays: the L1 inhibitory magnitude rose from0.045 to0.09156 in baseline and to0.06990 in F. Tag stability therefore is not an artifact of keeping the weights frozen.

However, event-update direction is structurally simple here: targets are +/-1, predictions are bounded to[-1,1], and each selected eligibility has its edge's fixed sign. Except for zero proposals, `sign(delta)` on event samples is therefore fixed by target polarity and edge sign. Predicting update direction conditional on an event is largely predicting which polarity comes next. It says nothing by itself about **whether the next frame contains an event**.

## Conflict remains inside a polarity context

During the F replay, the positive-sensory-state context contributed these actual proposals:

| Edge | ON proposal sum | Quiet proposal sum | Interpretation |
|---|---:|---:|---|
| L2 excitatory | -0.40189 | +0.14775 | Quiet partly reverses the ON direction |
| L1 inhibitory | +0.42863 | -0.29330 | Quiet partly reverses the ON direction |

The negative-sensory-state context separates the OFF proposals from these ON proposals, but does not distinguish event-due frames from surrounding quiet. These are proposal sums; clipping/homeostasis mean they are not net weight differences. Opposing error corrections are not intrinsically a defect: quiet penalties are necessary to suppress false alarms. The data do not justify removing quiet learning.

## The omission exposes the causal limit directly

The first omitted event has an identical visual/neural history to the ordinary sequence through issue tick208. Recorded issue state, eligibility, prediction and weights are identical. At confirmation tick216, the ordinary sequence supplies ON and the omission supplies quiet.

For F, the same issued prediction is-0.0847166. The ordinary ON produces L2/L1 proposals[-0.01084998,+0.01888796]; omission produces[+0.00100425,-0.00174823]. Baseline shows the same reversal. The observed sensory state differs at confirmation, as expected, because the new event has arrived in only one sequence.

This is a deliberately unannounced first omission, so **no causal forecast-time mechanism can distinguish these two futures from their shared prefix**. That pair is a control, not proof that extra state or another architecture is necessary. Confirmation-time separation alone would be tautological: the current error already contains the new observation. A useful credit mechanism must associate that error with a context available when the forecast was issued.

## Decision and next step

The hypothesis worth pursuing is narrower: separate polarity contexts may reduce interference between different event classes, but temporal selectivity and quiet corrections still have to be handled inside those contexts. This study does not demonstrate that one scalar magnitude is mathematically incapable, nor that extra magnitudes will work.

Before changing the biological model, the next bounded feasibility check should compare a shared-magnitude and context-separated **offline capacity bound** on the already saved frozen local histories, retaining quiet samples and held-out tempo. Fit only existing edge-local eligibility features and the already observed local polarity cue; treat the fit as a diagnostic, never as model weights or an artificial prediction head. If even an optimistic context-separated bound cannot retain both event anticipation and low quiet error, a polarity-only plasticity experiment is poorly justified. If it can, prepare an explicit, approval-gated local learning/expression proposal and test actual learning against that bound. No such fit or architecture change is included in this report.

This avoids treating another good sign-classification number as prediction success, and avoids adding a new temporal clock without evidence. The repeated-polarity generalization question remains untested under the current one-pixel mapping.

## Verification, cost and artifacts

The focused suite passed **36 tests** after the stimulus fallback correction. A second runner invocation successfully reused all six completed conditions after checksum validation, without rerunning neural simulation. The report independently verifies raw artifact hashes, source-training hashes, recorder identity, +8 timing, finite states, zero accumulator error and the paired identical prefixes. The 6 real-data sequences used about79.28 seconds of measured simulation/recording time in total, excluding source loading and warmup. This is a diagnostic runtime, not a production throughput benchmark.

- [Full results, actual update sums, source hashes and paired examples](2026-09-21-credit-replay-results.json)
- [Protocol and documented stimulus limitation](2026-09-21-credit-replay-protocol.md)
- [Earlier frozen-context pilot](2026-09-21-credit-context-findings.md)

```powershell
.venv/Scripts/python scripts/credit_replay.py
.venv/Scripts/python scripts/credit_replay_report.py
.venv/Scripts/python -m pytest tests/test_credit_replay.py tests/test_frame_prediction.py tests/test_short_term.py tests/test_context_audit.py tests/test_short_term_comparison.py -q
```

Raw arrays are under `runs/credit-replay-v1`. Completed conditions are reused only after runner, source, stimulus and raw artifact checksums match. Existing checkpoints and production learning code remain untouched.
