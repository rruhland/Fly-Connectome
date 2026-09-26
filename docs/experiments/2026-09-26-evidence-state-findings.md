# Weighted evidence and surface association: one bounded comparison

**Decision:** Neither candidate passes M1A.5's noisy-state gate. Preserve the
surface candidate as a clean-state improvement, not as an accepted M1A state.
Do not tune either association beam after this run.

All arms use the causal event-only observer teacher, identical training streams,
and the registered benchmark. Local context credit uses observed reacquisition;
hidden truth is scoring-only. Production architecture is unchanged.

| Aligned state | Clean context /16 | Familiar noise /16 | Heavy noise /16 | Constant /16 | Two movers /16 | Continued /16 | Vanished false /16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weighted events + association beam | 8 | 7 | 2 | 16 | 16 | 16 | 4 |
| Contrast surface + association beam | 16 | 5 | 3 | 16 | 16 | 16 | 0 |
| Required | 14 | 12 | 8 | 14 | 14 | 14 | <=2 |

Localization uses top-32 except the two-entity top-64 regression. Continued and
vanished use the previously registered strength >=.5 check after reveal; these
strengths are **not calibrated probabilities**. Both candidates give exactly
identical outputs before the reveal on matched continue/disappear histories.

The surface state's aligned context score exceeds its frozen and shuffled
controls (both 4/16 top-32, 0/16 top-8). Aligned reaches 16/16 top-8 as well.
Its continuing reacquisition improves from the previous candidate's 10/16 to
16/16, without increasing clean vanished-target false persistence. This is a
useful clean-state improvement with the same local credit principle. It does
not overcome noise: surface aligned/frozen/shuffled familiar-noise scores are
5/4/3 of 16. Sparse-frame inference with the old .35 blend reaches only 8/16.

Weighted event state gives 7/16 familiar-noise versus frozen 0/16 and shuffled
1/16, but loses half the clean context hits and produces four false persistent
targets. Its improved noisy count is not an overall win.

Important limits: each slot retains four local match/miss histories, with greedy
competition between the strongest slot hypotheses. This is not globally optimal
multi-target Bayesian inference. Birth confirmation requires repeated evidence;
the prior contrast-support confidence mechanism still determines disappearance.
Context extraction remains the specialized above/below comparator, and state
positions/velocities remain engineered rather than a discovered generic latent.
Top-k localization is weaker than occupancy precision or verified identity.

The next broad question is **sensor reconstruction versus state capacity**.
Hold the cleaner surface state fixed and provide declared sparse/all-frame
visible-image anchors. If accurate appearance still fails, discard the state;
if appearance rescues it, prioritize the event-to-surface interface before
inventing another context/plasticity mechanism. This is registered separately in
[the sensor-budget protocol](2026-09-26-sensor-budget-protocol.md).

Reproduce with `.venv/Scripts/python scripts/run_evidence_state.py`.
The [raw results](2026-09-26-evidence-state-results.json) include all six arms
and wall-clock timing. Unit checks cover delayed birth, separate movers, causal
continuation through missing input and reset. Full suite after this first chunk:
646 passed, 4 skipped.
