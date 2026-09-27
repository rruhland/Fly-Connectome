# Eight-step motion context: useful improvement, retention exception remains

Recommendation: preserve this candidate and move toward production integration,
subject to user approval and live-runtime regression checks. Stop mechanism/length
sweeps. More observed temporal context improved actual held-out prediction and
reuse of learned dynamics; the strict immediate-retention screen did not fully pass.
Production is unchanged. See the [integration proposal](../plans/2026-09-27-eight-step-vision-integration-proposal.md).

## Comparison

[Protocol](2026-09-27-long-context-protocol.md),
[raw results](2026-09-27-long-context-results.json).
Five seeds; 2,000 samples per A/B/return-A segment; evaluation every 500 samples.
Reported target times are first measured crossings, not exact learning latencies.
Run completed in 1,302.46 seconds, including camera-observation generation and
repeated held-out evaluation. This is not a production throughput measurement.

Both experimental arms have the same full four-step append-all learner and a new
specialist trained at identical eligible origins. The only specialist difference
is four versus eight observed displacements, using the same normalization function
and learning/retrieval rule. Eight-step histories are never invented from the old
checkpoint. Nine consecutive observed positions and 32 specialist outcomes enable
the specialist; otherwise the four-step learner supplies the forecast.

Mean NLL combines horizons 4/8, then seeds; lower is better. Reference controls come
from the prior memory comparison. Embedded four-step shadow scores reproduce the
append-all reference within 1e-6 over every online/fresh/return curve.

| Model | Shared-B final NLL | Changed-B final NLL | A before changed B | A after changed B | A degradation | Retention passes |
|---|---:|---:|---:|---:|---:|---:|
| Current production reservoir | 3.704 | 3.221 | 3.939 | 4.135 | +0.196 | 4/5 |
| Full four-step append-all | 3.656 | 3.202 | 3.921 | 4.046 | +0.125 | 4/5 |
| Matched four-step specialist | 3.649 | 3.149 | 3.823 | 4.026 | +0.204 | 2/5 |
| Eight-step specialist + fallback | **3.251** | **2.846** | **3.453** | **3.610** | +0.157 | **3/5** |

Eight-step final prediction scores beat the matched specialist in every seed in
both domains. Absolute retained A quality also beats it in every seed. The model
learns A better and remains more accurate after B, but that is not equivalent to
passing the separate degradation limit.

## Transfer and adaptation

Eight-step shared-B samples to target: `[0,0,0,1000,0]`.
Its fresh-at-switch controls: `[2000,1000,null,null,2000]`, where null means no
crossing at the measured checkpoints through 2,000 samples. Four seeds reach the
criterion before any B training; the fifth reaches it while fresh remains censored.
This is useful transfer across held-out appearance/layout draws with shared
dynamics, not evidence of competence in arbitrary games or natural-camera scenes.

Changed-B target is reached at the first 500-sample check in all eight-step seeds;
fresh eight-step controls do the same. That is faster adaptation than the old
reservoir (1,000–1,500), **not** a demonstrated acquisition benefit from A learning
when the dynamics themselves change. The matched four-step specialist takes
`[1000,1000,1000,1000,500]`.

Learning shared B improves A in all five eight-step seeds, by 0.182 nat on average
without A retraining. After changed B followed by 2,000 return-A training samples,
mean A NLL is 3.485 for eight-step versus 3.967 for the matched specialist.

## Context is actually used

On these short scenes, eight-step prediction supplies 60% of evaluated horizon-4
forecasts and one third of horizon-8 forecasts. All remaining forecasts still count
in overall NLL; availability is 100% throughout both arms' evaluated curves.

| Domain | Horizon | Eight-step NLL on eligible origins | Full four-step shadow, same origins |
|---|---:|---:|---:|
| Shared dynamics | 4 | 2.509 | 3.105 |
| Shared dynamics | 8 | 2.925 | 4.281 |
| Changed dynamics | 4 | 2.264 | 2.911 |
| Changed dynamics | 8 | 2.506 | 3.475 |

This is predictive evidence beyond the earlier key-collision audit. It does not
isolate every contribution of the longer normalization window from retrieval;
the tested change is longer history under the same encoding/learning procedure.
Final marginal interval coverage ranges from 87.5% to 96.5% over the two domains,
five seeds and horizons 4/8. These are marginal, not joint or conditional guarantees.

## Retention exception, explicitly not waived

Eight-step A degradation after changed B is
`[0.1185,0.2299,0.2083,0.1111,0.1151]` nat. Seeds 1 and 2 exceed the registered
0.2 limit. Compared with the reservoir, the mean and worst degradation improve,
but the number of strict passes drops from four to three. Compared with append-all,
mean relative degradation is worse even though absolute prediction is much better.
Do not present this as an all-seed continual-retention pass.

Both fallback and longer-context forecasts can lose accuracy after the shift.
At horizon 8 in the two failing seeds, fallback loss is 0.428/0.328 nat versus
0.185/0.221 on long-context origins. Those strata explain part of the residual;
they do not prove a universal irreducible uncertainty bound or justify dropping
early forecasts from the benchmark.

My recommendation is to accept the useful predictive upgrade as the next integration
candidate and track this limitation rather than reopen a broad latent search.
Default promotion requires explicit acceptance of that tradeoff, as well as passing
integrated regression checks. No new mechanism sweep was run after these results.

## Verification / scope

Full suite: **713 passed, 4 skipped**. Tests cover distinct futures with identical
last-four-step histories, exact fallback behavior, issued-distribution calibration,
gaps, and production four-step replay parity. Independent code review found no
causal or fair-comparison defects. Production source/default checkpoint is untouched.
The reusable audit runner now supports longer contiguous histories while preserving
its default four-step behavior. New code remains in experimental scripts.

All conclusions are from the declared generic synthetic camera streams. The
existing observation model and coherent-entity interface are preserved. No reward,
game label, semantic object identity, or simulator target enters learning. Append
memory has growing storage/retrieval cost; this was not used to discard the result.
