# First-return → full-rally reward curriculum

Prospective, one-seed development study under the user's continuing authorization.
No backpropagation, teacher labels, privileged policy inputs, new visual features,
refit, or final holdout streams. A hit-learning screen is not complete M1B.

The audited same-class continuation failed to improve theta9: first returns
35/64 versus 37/64, all returns 48/105 versus 48/107. Its original-initial primary
screen also failed. The new allocation starts afresh, preserving those negatives.
Existing full-rally data spent 63,473/145,561 cameras strictly after the first hit
(43.61%; first-hit camera retained). Cached disjoint three-serve gradient halves
were weakly aligned. Thus phase zero spends potential stopping savings on twelve
shared serves, not more directions. This changes objective emphasis, exposure
and sample allocation; it is not an isolated causal or algorithm comparison.

## Fixed condition and budgets

Frozen repaired vision, model a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173,
empty long banks/four-step fallback; unchanged generic DCT transform (214 stored
coordinates, supported rank 183/366 action contrasts), radial cap 256, opponent
speed .65, hold1. Eight mirrored directions/top four, noise .25, update .10 with
function-RMS step cap .10. From-scratch score-free random initialization RMS .10,
not zero or an earlier trained/oracle readout.

| Phase | Fitness | Shared serves / population | Cameras | Updates | Wall |
| --- | --- | --- | --- | --- | --- |
| phase0-first-return | mean binary first own hit | 12 / 192 jobs | 120,000 | ≤16 | ≤24 min |
| phase1-full-rally | mean total own hits | 6 / 96 jobs | 80,000 | ≤16 | ≤16 min |

At least two complete updates per phase for valid endpoint eligibility. Maximum
200,000 new training cameras, 40 minutes shared training, 60 minutes overall;
20 evaluation minutes, with 180 seconds reserved inside that for annotation and
statistics. Four single-threaded workers use the previously validated imports-only
GC freeze initializer once per spawned process; collection remains enabled.
No extension, outcome-triggered transition, intermediate DEV, anchor allocation,
or best-checkpoint selection. Stage inputs are copied before any clock starts.
Actual root/phase starts and frozen timestamps persist across restart. The
evaluation clock starts at the durable phase-one frozen timestamp, and includes
rollouts, annotation and statistics; exceeding its 20-minute cap makes the
allocation partial even if the overall 60-minute bound is met.

## Reward and task boundaries

All training policies receive images only. Fitness uses logged scalar own hits;
raw terminal score, opportunities, labels and forecasts' diagnostic errors do not
enter fitness. First-return jobs explicitly opt into `first_hit_cutoff=True`,
validated at the worker boundary as single-episode `training_fitness` only.
They execute and encode the complete first-hit camera transition, then finish
that training task before subsequent cameras. Raw reward, `terminated` and
`truncated` remain unchanged. Explicit `task_stopped`, `task_stop_reason`,
`task_stops` and task definition distinguish a successful task stop from a game
win, loss or time-limit truncation. A first-hit task stop is not a scored win.
Natural termination or ten-second time limit ends an unsuccessful zero-hit task.
Multi-episode cutoff is rejected to prevent silently resuming a stopped world.

Phase one carries the last complete phase-zero theta and the explicitly recorded
next RNG/global index. Any consumed incomplete draw, including an unexecuted
pinned definition, is skipped; its partial observations never update or transfer.
Definitions pin all candidates and shared serve seeds before rollout. A complete
pending population is reconciled before budget checks on recovery, including a
result-write-before-state crash at exhausted budget. Only state-committed results
count as used fitness. Failed task costs remain failures after resume. Frozen
boundaries verify theta, RNG, global index, counters and failure flags.

## Seeds and immutable inputs

Training root 1518615419; optimizer 1097410397; development root 1746610937/tag
29107; random-action root 1297478891. Root collision checks precede expansion.
Training serves use `[1518615419,872631,global_generation]`, rejecting metadata-
snapshotted prior training, DEV, calibration and reserved IDs plus new DEV IDs.
The twelve or six resulting seeds are identical across all sixteen candidates.
Every serve starts a fresh environment/camera phase. DEV seeds draw sequentially
from `default_rng(SeedSequence([1746610937,29107]))`, rejecting prior/reserved IDs
and duplicates until 64 accepted. Random controls use action seeds from
`SeedSequence([1297478891,1746610937,serve_index]).generate_state(1)[0]`.
Reserved final roots 308713321, 1203078761, 1983461927 remain unexpanded.

Exact T, fit, cap and score-free 1024-row calibration are reused, never refitted.
Parent's exact calibration replay reproduced every archived raw row, then found
27/1024 rows changed under the continuity repair: raw relative Frobenius 3.10%,
projected 16.17%, covariance 7.87%, aggregate energy .983×. Across 144 saved search
directions RMS ratios were .9661–1.0088 (median .9913). This quantifies a modest
normalization shift; it does not prove policy robustness. See external
`/workspace/m1b-experiments/calibration-continuity-check-20261021`.
Manifest pins source/runtime/model, all helpers/protocol bytes, input file hashes,
seed metadata and exact configuration. Per-camera journals/snapshots, RNG,
optimizer state, completion integrity pins and durable recovery are retained.

## Endpoint-only fresh evaluation

Only after both phases freeze, evaluate 64 shared new ten-second full-rally serves
for nine cohorts (576 jobs, maximum 288,000 evaluation cameras): final endpoint;
its own from-scratch initial; constant down/up/no-op; random hold1/hold10;
phase-zero endpoint; final endpoint with blank camera. All learned policies use
frozen greedy hold1, no updates, no first-hit cutoff. Simple policies may skip
perception using the previously verified identical action/physics/reset path.
Phase-zero readout is secondary, never a replacement if the final endpoint fails.

Primary controls are own initial plus five simple policies. Unchanged screen:
≥20 percentage-point lift on BOTH first-incoming return success and all-incoming
return fraction, with both paired 95% CI lower bounds >0 and zero invalid draws.
Ten thousand paired serve-cluster bootstrap draws use seed 193870291; the strongest
control is recomputed independently for each metric inside every draw. All nine
cohorts must match initial incoming geometry. Report raw opportunities/returns,
first successes, ≥2-return rallies, hits/physical second, durations, game scores,
timeouts and opponent reliability. Later opportunities are policy-dependent;
this is one training seed and 64 paired serves, not independent pooled trials.

Secondary comparisons separately report phase-zero minus own initial first/all,
final minus phase zero first/all, and final minus blank first/all, without changing
the primary screen. Positive final-minus-phase-zero all-return CI supports extra
full-rally-phase learning; it is not required if the whole procedure qualifies
without it. Blank is OOD whole-input dependence, not tracking, object-specific or
forecast benefit. No opportunistic promotion of the phase-zero checkpoint.

Every partial rollout/population, task failure, recovery replay, calibration reuse,
annotation/audit computation and resource-stop is reported. Opportunity labels
are generated after evaluation only; parent independently audits training physics,
cutoff truth, optimizer/phase lineage and paired statistics. A negative result
triggers review of the next bounded genuine-reward experiment, not an automatic
stop or unregistered budget extension. Replication and broader B1/B2/B3 work remain.
