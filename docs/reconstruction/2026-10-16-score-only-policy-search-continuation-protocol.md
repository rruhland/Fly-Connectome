# Prospective fixed score-only policy-search continuation (.25 opponent)

## Rationale and inherited evidence

The original [registered search](2026-10-16-score-only-policy-search-protocol.md)
completed seven updates from 59,903 training camera transitions. Its endpoint won
12/24 DEV rallies (four own-hit-followed wins), versus its original restricted
initial readout's 10/24 (two own-hit wins). This FAILED the 14-total/6-own-hit gate;
no acquisition or final evaluation is claimed. Exactly 56,089 observed transitions
contributed to those seven updates; 3,814 belonged to incomplete generation index7
and did not contribute. A separately quarantined restricted-class oracle witness
controlled 6/12 rallies, supporting usable capacity, not a reward-learning claim.
Only 17 of the original 56 sampled directions had selected nonzero paired reward
contrasts. A finite continuation is informative without concluding that compute
alone will solve acquisition.

This NEW allocation fixes one continuation from the last complete theta7. The expanded budget is fixed prospectively for this continuation, motivated by
the failed DEV screen and limited search exposure. Learning rule, basis, reward,
physics and optimizer hyperparameters remain unchanged; there is no midrun budget
extension or checkpoint selection. No oracle coefficients select the allocation. Frozen raw
perception and score-only reward remain; no backprop or oracle policy inputs.
Claims remain opponent speed .25. Original .45 is unchanged and receives no training.

## Exact continuation and finite budget

Reuse the EXACT original 214-coordinate transform, rank183/366 effective action
contrasts, score-free 1024-input calibration, centering through retained
frame-present bias, second-moment support/ridge .01, cap256, and mapped full linear
heads. No new calibration observations or fit. Reuse eight Gaussian directions,
function-RMS normalization, noise .25, top4 mirrored score update, step .10 and
step-RMS cap .10. Fitness is ONLY mean terminal score across four matched fresh-phase
training serves: +1 win, -1 loss, zero truncation. No hit/teacher shaping.
Hold1, .25 opponent, 20-second rally limit, exact prior reset/vision semantics.

Start parameters are original theta_at7. Discard ALL incomplete original generation7
fitness and journals from this new allocation; preserve them as unused historical
work. Start RNG from original g07-definition's rng_state_after_draw, so its unused
perturbations are not reused. First new global serve-generation index is8, using
unchanged SeedSequence([172900127,872631,global_index]) and fixed four serves shared
by all16 candidates. Each serve begins a fresh camera phase. Local files g00..g15
correspond to global indices8..23. Sixteen additional completed updates would mean
23 TOTAL updates (7+16), not24; global index7 was skipped, never updated.

Stop at the first of16 ADDITIONAL full generations, 120,000 NEW training camera
transitions, or34minutes training wall. Overall shared wall is40minutes including
setup and endpoint evaluation; no deadline extension. Four persistent workers,
per-camera commits, <=200-camera chunk boundaries, full snapshots, warm sessions,
complete actual-recovery verification and pre-dispatch camera reservations remain.
Fewer than200 remaining cameras may be unused. Failed work stops further training;
incomplete generations never update. At least TWO additional full generations are
required for an evaluable new endpoint. Freeze the LAST completed update, never the
best. Training camera count includes all partial work; maximum cumulative observed
training is179,903 (59,903+120,000). Report used and unused counts separately.

Four workers are supported by a separate exact replay ABBA benchmark (18.13% wall
reduction). The sole core-source change is bulk tensor byte conversion in graph
hashing: unchanged digest format, complete rescanning, exact reference IDs and
mutation checks, no cache shortcut. Timing reports distinguish uninstrumented rollout comparisons from instrumented
call attribution. Instrumented pairs are excluded from deployment speed estimates;
no hash microbenchmark is presented as end-to-end gain. Uninstrumented ABBA
reductions were2.53% and0.71%, so NO confident deployment gain is claimed. The
exact fastpath is retained; four-worker throughput is the supported runtime gain. Old source archives and
new source bridge retain earlier baseline/action/physics projection proofs.

## Endpoint, controls and scope

No intermediate DEV, checkpoint selection, new oracle labels or final seed use.
At the fixed endpoint run ONLY one independent frozen evaluation on the existing
24 DEV rallies, seed904271833, 20-second limits, full perception. Evaluation receives
at most24,000 NEW camera transitions and the remaining overall wall budget; partial
results stay partial. Reuse source-proven initial/down, up, random, no-op controls
and the ORIGINAL theta0 restricted initial control (10/24, two own-hit wins).
Original theta7's 12/24/four own-hit result is descriptive earlier training evidence,
not a new initial control that raises the acceptance gate.

Total-win threshold=max(14, maximum matched-control total wins +4).
Own-hit-win threshold=max(6, maximum matched-control own-hit wins +4).
The two maxima are independent. Both are currently14/6. Timeouts are nonwins.
Report every rally, natural versus own-hit-followed wins, losses/timeouts, score
per completed rally, physical returns/time, hits, camera/reset/replay work, and
training/evaluation wall costs. A passing DEV screen permits only separately
registered replication; it is not genuine acquisition evidence by itself.

All source/config/protocol/runtime/transform/calibration/lineage/report hashes,
optimizer RNG and per-camera journals are retained. Coefficients from quarantined
oracle diagnostics remain unavailable to the learner. Reserved final streams
308713321,1203078761,1983461927 remain untouched. Any later independent replication
must reproduce the WHOLE staged learning procedure from zero; selective continuation
based on an intermediate DEV result is not an independent replication.
