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
