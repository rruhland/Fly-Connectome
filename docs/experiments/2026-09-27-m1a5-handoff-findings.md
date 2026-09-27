# M1A.5 handoff: usable interface, unresolved continual-dynamics gate

The approved production visual model is preserved. The generic central adapter is
implemented. The remaining continual-learning audit finds useful transfer, but
does **not** establish the full requested continual-learning capability. Stop here
for the user's requested reassessment before a deeper memory/credit investigation.
Do not call the continual upgrade accepted or start M1B reward learning by implication.

## Five-seed causal dynamics audit

Protocol: [registered audit](2026-09-27-continual-vision-protocol.md).
Raw curves: [results](2026-09-27-continual-vision-results.json).
Each training segment has 2,000 samples. Every arm starts with the same bundled
prior. Independent fixed probe streams are never trained on. Cached actual
observations reproduce production pending credit exactly in the parity test.
Elapsed audit time: 285.76 seconds, including observation generation and probes.

Lower NLL is better; averages below combine horizons 4/8 and then seeds.

| Change after A | Online initial NLL | Online final | Fresh final | A degradation after B |
|---|---:|---:|---:|---:|
| New appearance, shared dynamics | 3.934 | 3.704 | 3.991 | -0.194 (improvement) |
| Changed dynamics | 11.080 | 3.221 | 3.181 | +0.196 |

Shared-domain final online scores beat fresh in **5/5** seeds. Learning B also
improves A in **5/5** seeds without A retraining: genuine backward benefit on these
fixed probes. However, fresh never reaches the predeclared 3.5 criterion within
2,000 samples; online reaches it in only one seed. Consequently, the registered
20% acquisition-speed comparison is censored, not passed or disproven. Longer
training may resolve that particular measurement; compute is not a hard veto.

Changed-domain acquisition takes `[1000,1500,1000,1000,1000]` samples online versus
`[1000,1000,1000,1000,1000]` fresh. Four tie and one is slower; this is not a
forward-transfer gain. A degradation per seed is
`[0.1433,0.1945,0.1966,0.3474,0.0983]` nat. Four meet the <=0.2 retention screen;
seed 3 fails. All return-A forecast-availability coverage remains 100%; the failure
is prediction quality, not missing outputs. Final changed-B marginal calibration
coverage is approximately 91.5–95.8% across horizons/seeds; this is not joint coverage.

This is a meaningful interference signal, not catastrophic loss and not evidence
that the whole visual representation should be discarded. It also does not identify
the cause. Reservoir replacement, retrieval mixing and histories that alias different
futures remain distinct hypotheses. The audit does not justify declaring any one
of them responsible. Fully visible single-mover probes do not establish continual
binding quality, interaction modeling, or transfer across actual games.

## Separate context-memory retention screen

[Context results](2026-09-27-context-retention-results.json): the production context
memory initially solves 64/64 spatial and 64/64 appearance probes. After 8,192
actual camera samples of appearance-relation experience, both remain 64/64.
After another 8,192 spatial-relation samples, both remain 64/64. Keys change and
the memory remains at 256 examples. Whole-scene online credit uses observed
reappearances; hidden masks are evaluation-only. This deterministic known-relation
screen passes, but is not proof of indefinite FIFO retention or novel concept learning.

## Central interface

[Contract](../plans/2026-09-27-central-visual-interface.md): `VisualStateEncoder`
retains every entity and full mixture, adds actual-observation-only velocity with
age, explicit cadence, coarse static context and a permutation/ID-invariant sparse
transport vector. No game labels, action semantics, reward or learned-neuron
replacement enters it. Sparse feature utility still requires an M1B comparison
against full structured beliefs; encoding alone is not evidence of useful control.

[Timing results](2026-09-27-visual-interface-timing.json), single CPU thread,
learning enabled, 200 measured samples after 40 warmup samples per arm:

| Visible entities | Total p50 / p95 ms | Encoder p50 / p95 ms | Active features p50 |
|---|---:|---:|---:|
| 2 | 75.44 / 80.26 | 5.90 / 6.63 | 94 |
| 8 | 294.62 / 354.98 | 24.78 / 28.70 | 167 |

All entities are detected in these timing scenes. Sensor rendering/acquisition and
I/O are excluded. No other task-launched audit/test was running during this final
timing pass; background host load is not controlled. These timings include online
credit, full mixtures and calibrated interval export, unlike earlier frozen
single-entity timing. The 50 Hz target is not met. This is a profiling opportunity,
not an argument to discard useful learning. Sparse transport output does not make
the underlying image and predictive-memory computation sparse. No cause of the
runtime gap has been established by this timing measurement alone.

Verification: full suite **707 passed, 4 skipped** (106.79 seconds). Independent
review found and verified the fix for rectangular-sensor off-image reporting;
no remaining actionable findings. The approved checkpoint and visual learning
rules are unchanged. New tests cover causal cache parity, transport invariance,
full-mixture preservation, missing evidence, reset, static context and off-field mass.

## Recommendation / next decision

Keep the production observation/context system and the interface. Do not neuralize
them or launch another broad latent search. Preserve the demonstrated shared-domain
benefit. The unresolved issue is continual predictive associations under changed
dynamics, not whether the system can learn anything online.

Before claiming the remaining continual gate complete, run one causal comparison:
separate memory interference from insufficient temporal context. Compare the
planned recent+stable memory against current reservoir retrieval on the same
fixed A/B probes, while checking whether identical available histories demand
different outcomes. A memory-only fix cannot resolve genuinely aliased contexts.
This is a recommendation, not an implemented production architecture revision.
The user asked to be informed when deeper investigation is warranted; the failed
retention seed and unresolved transfer-speed measurement are that decision point.
