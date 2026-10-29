# One closed-loop acquisition and unchanged SMDP learner refit

The initial dense SMDP learner failed its actual eight-game comparison: baseline
23 returns versus candidate0 12, first returns 7 unchanged, later returns 16 to 5.
This cycle changes the learner's experience to include the states it actually
induces. It preserves the representation, physical reward timing, estimator,
support rule, sample-weight formula and runtime option contract.

Candidate0 remains frozen and preserved:
`ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295`.
The existing pinned learner/core are reused without edits. The adapter is the new
`run_smdp_closed_loop_cycle.py`; collection belongs to the evaluation worker.

Before collection, parent reviews fixed collection and learner plans. Collect 64
fixed TRAIN starts, 32 per initial phase, excluding the previous 75 training starts
when available. All eight candidate0 evaluation starts are already among those
75. No fresh starts or reserved fresh stream 63233 enter training.

The collector follows baseline 40069 until the first actual own hit. At each later
option decision it selects candidate0's greedy option with probability 0.8125 and
each other option with probability 0.0625: 0.75 greedy plus 0.25 uniform over all
four options. Baseline option 3 executes one camera; command options 0/1/2 execute
up to 16 constant-command cameras, stopping early on an actual hit or natural end.
The bound is 180 seconds, 64,000 transition cameras, at most 64 initial cameras.
Save every success, failure and censored partial with N+1 compact scenes, exact
actions, elapsed times/ticks, baseline scores, option decisions/probabilities and
termination/censor flags. Probabilities are acquisition audit data, not labels or
policy inputs.

Also admit ALL eight existing candidate0 TRAIN evaluation trajectories, 3,150
already-executed cameras and 12 own hits. This includes every loss and retained or
gaining case. This explicit evaluation-to-TRAIN admission costs no new cameras.
Those archives have N pre-action scenes: retain verified natural terminal finals
without a successor, and omit censored finals lacking a successor. Do not label
unknown tails as failures or reuse the old forest's full-160-window zero targets.

The adapter appends the admitted and collected traces to every old valid SMDP
tuple. It uses the SAME actual-window extraction as candidate0: only observed
constant-command windows or actual baseline one-camera steps, with early own-hit
and natural-end boundaries. It does not relabel changing actions. Empty/overflow
states remain unsupported. All real outcomes and unknown MC tails are recorded;
unknown MC tails are omitted from initialization only and remain eligible for FQI
when they have valid actual successors.

Recompute the same global weights on the combined dataset: equal option mass,
then equal distinct-start mass within option, then uniform windows within that
start-option group. Remap new start IDs without treating repeated occurrences of
old/admitted starts as independent starts. Report each cohort's cameras/windows,
known/unknown MC labels, per-option rows, distinct-start counts and resulting data
mass. Overlapping windows are correlated observations.

One 180-second refit allocation uses the unchanged observed BEHAVIOR-tail MC
initialization plus up to 32 actual-transition FQI updates. All options use the
same physical target, expected `2**(-time_to_next_observed_hit/3.2)`, including
discount through the hit interval. Keep initial MC and final completed endpoints
only. The final iteration is determined by time, never outcomes. Export and exact
query checks remain charged. Preserve candidate0, all receipts, original clocks
and both data cohorts. No parameter sweep or learned-graph distillation follows.

The new data offers deliberate option coverage in candidate-induced states. It
does not guarantee adequate conditional support or cure partial observability.
Actual candidate1 comparisons must include both candidate0 and the baseline;
recovering partway toward baseline is not promotion or the user's milestone.

## Completed append and offline refit

The completed collection summary verified frozen sources/assets, all 64 started
receipts and camera counts, candidate0 model identity, receipt hashes and each
scene/transition hash before ingestion. The original plan remains preserved as
`cycle-plan.before-collection-guard.json`; guarded plan SHA:
`0d285bd11098a8b13374494689cfae21d7a21151c05025e88ff938ff53fcd3a5`.
Collection spec SHA:
`4428646edb4c832c6bce9b6754915e5651db655b26286f00a0d8e9a8c6d08ccf`.
The guard changes were committed as `b7d5c18`; the original adapter is `e4007b3`.
Five focused append/guard tests passed, alongside the seven previously passed
unchanged learner tests.

All eight candidate0 TRAIN evaluations contributed 3,150 already-recorded cameras
and 2,836 SMDP windows. All 64 newly collected trajectories contributed 22,876
cameras, 86 actual hits and 18,035 windows. Every old prepared array remains a
byte-identical prefix of the combined dataset. There are 139 distinct starts and
65,726 actual option windows, of which 65,409 have supported source scenes.

| Option | Eligible rows | Distinct starts | Known MC labels | Unknown MC tails |
| --- | ---: | ---: | ---: | ---: |
| Held 0 | 4,160 | 133 | 3,789 | 371 |
| Held 1 | 3,712 | 130 | 3,695 | 17 |
| Held 2 | 2,558 | 129 | 2,510 | 48 |
| Baseline 1 camera | 54,979 | 139 | 51,702 | 3,277 |

The unchanged weight formula gives each option 25% total mass. The final combined
data spec SHA is
`53629b2ed3e527224895bbbf75f7048cc23b19ea6a14d366573088187b9b1208`.
The prepared dataset occupies 1.26 MB.

The one refit completed in 166.942 seconds, with all export/parity work charged.
Initial MC fitting took 7.594 seconds using 61,696 known labels, omitting 3,713
unknown tails only from MC initialization. Sixteen FQI updates completed; another
complete update did not fit the remaining allocation. This time-based final
iteration differs from candidate0's 26 updates under the same envelope; no native
or training outcome selected it, and neither endpoint is claimed converged.

The exported final model matched sklearn exactly on all 65,409 training inputs.
Serialized values/support also matched all four options on 512 saved states,
2,048 queries. The final forest has 798,878 nodes. New artifacts occupy 22.67 MB,
below the 50 MB bound. No new native comparisons ran in the learner task.

Artifacts: `/workspace/m1b-experiments/pong-smdp-closed-loop-refit-20261029-r1`.
Final `final-forest.npz` SHA:
`e58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd`.
Initial `initial-mc.npz` SHA:
`29216b0319d24efa79780db563e0030b4e54b70d4acce32924ef2bc44e8d7cb9`.
Candidate0 remains unchanged. The final hash and stable API were sent to the
evaluation worker for parent review before candidate1 comparison.
