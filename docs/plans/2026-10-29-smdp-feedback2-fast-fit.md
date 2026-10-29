# Second feedback cycle with exact training-query optimization

Candidate1 improved candidate0 on 32 TRAIN games, 44 to 50 returns and 15 to 21
later returns, while remaining below baseline 40069's 82 total and 53 later.
This authorizes one more declared data-feedback cycle, not a parameter retry.
No fresh evaluation or promotion follows from these TRAIN results.

Preserve every candidate1 prepared array and start ID. Admit all 56 newly executed
TRAIN comparison traces: candidate0's 24 missing games and candidate1's complete
32 games. Do not duplicate the eight cached candidate0 games already admitted.
The 56 new traces contain 21,664 cameras. Their 32 starts add three new starts to
the previous 139, making 142 exclusions for the next collection. Candidate1 acts
frozen during collection2 on 64 new TRAIN starts, 32 per phase, with the unchanged
post-hit epsilon 0.25 option distribution and early-ending macro contract. The
collector has its own root-approved 180-second, 64,000-camera allocation.

The new append wrapper reuses the existing Appender and collection-completeness
guards via explicit process-local collector path/model configuration. It never
edits the prior source files. Every started receipt, actual camera count, model,
receipt/scene/transition hash and frozen source/asset summary must match. Append
every valid observation from all 56 admissions and all collected traces. Unknown
MC tails remain omitted from MC only; real-transition FQI eligibility is unchanged.
The same global option/start/window weights are recomputed over all data.

The learned state, estimator, seed, physical reward timing, support rule, MC
initialization, Bellman targets and final-completed-iteration selection stay the
same. A new runner changes only training query computation:

- Compute actual-option presence with leaf-option bincounts. The gate uses presence
  in 17/32 queried leaves, so per-update distinct-start sorting is unnecessary.
- Use one C `apply` traversal to retrieve leaf values and support. Accumulate
  float64 values in the same tree order as sklearn's single-thread prediction.
- Cache observed training predictions across iterations. Each candidate's single
  `apply(F)` supplies both predictions and option presence.
- Compute actual independent-start counts only for initial and final portable
  exports and diagnostics. Presence is never labeled independent evidence.

The fixed 180-second clock now reserves 12 seconds for larger-model finalization,
with a 168-second fitting cutoff. All export/parity/cleanup time remains charged.
Any increased update count is disclosed; there is no outcome-based checkpoint
selection. Initial and final endpoints are the only saved policies.

No-fit verification rehydrated the existing candidate1 parameters into query-only
C trees, preserving all finite-input thresholds and values. On 65,409 observed
training inputs and 57,295 actual successor states (229,180 option queries), the
original and optimized scores, support, and complete Bellman targets were exactly
equal. Recomputed true independent counts matched the saved export. The source
model and data stayed unchanged. Synthetic query-only tests also pass.

One-pass timings on that same saved model/data: old independent-start leaf
statistics 1.328 seconds versus new presence plus observed predictions 0.255;
old successor predict-plus-apply 1.472 versus apply-only 0.764. The combined two
costs fell from 2.80 to 1.02 seconds, before caching repeated observed predictions.
These measurements predict lower overhead, not the duration of fitting a new
forest on a larger corpus. The report is candidate1's `training-speed-parity.json`.

Before the sole refit, root reviews the new loop, fixed data plan, exact parity
and final dataset/spec. The native model format and stateful runtime remain
unchanged. Final parity includes original C predictions, optimized queries and
serialized portable queries. Stop on any exact mismatch. No new native games are
run by the learner task.
