# Staged score-only Pong step-size comparison

Both raw and conditioned delayed-toy policies remained 10/20 after the separately
fixed 200-episode exposure diagnostic. Retain that failure. Return to Pong and
change one parameter: alpha .0001 -> **.001**, with unscaled raw visual features,
lambda=.995, gamma_per_second=.95, epsilon=.2, zero initial weights, frozen prior,
score-only reward and unchanged environment/camera/reset semantics. Typical raw
feature norm squared is about 77.5, making alpha times squared norm about .0775;
record rare maximum norms, weight magnitudes, Q spread and numerical failures.
Do not assume this typical value bounds all inputs.

Train the first original development seed **172900127** for exactly **6000 new
transitions**. Save weights at 2000/6000. Evaluate both checkpoints, freshly and
greedily, on the original **12-rally development serve seed 904271833**, 2400-tick
20-second rally limit. Truncations are nonwins. Reuse original initial/down,
uniform-random and no-op baseline evidence with matching physics, action policy,
seed and runtime. Preserve both checkpoint outcomes regardless of the decision.

A checkpoint is promising only with **>=3/12 wins and zero truncations**. If neither
passes, stop this arm without replication or extension. If both pass, choose most
wins, tie to the earlier checkpoint. Freeze that choice in a selection artifact.
Before replication, evaluate constant-up on the same serve sequence; require at
least three more wins than constant-up as well. If that check fails, stop and label
the directional-bias alternative rather than spending on replication.

If promising, replicate the selected exact training budget (2000 or 6000) on the
other original training seeds **736187003 and 1597334677**, with identical alpha
and settings. Evaluate only the selected endpoint for each replication, on the
same development serves. Run all allocated replications, even if one fails.
In parallel, evaluate the first candidate with an explicitly blank camera input
on the same physics/serve sequence. Only the policy observation is blanked; score,
camera acquisition/reset and elapsed time stay unchanged. This checks dependence
on visual input and is not a forecast-benefit claim. No final seeds are used.

Maximum allocation: **18000 learning transitions**, plus at most **72 development
evaluation rallies** (two first checkpoints, constant-up, blank-camera candidate,
two replications). Conservative evaluation bound: 86400 transitions. Aggregate
wall cap: **30 minutes**. The common expected stop case uses only 6000 learning
transitions plus two 12-rally evaluations. Each durable chunk contains <=200 new
transitions; up to five chunks reuse one interpreter, each saved/stopped separately,
within a shared160-second work budget and180-second subprocess kill bound.

Preserve full source/runtime/seed manifests, all per-transition journals, fitted
weights, update/Q/feature diagnostics, complete and partial allocations and process
costs. Do not tune another knob, select replacement seeds or extend the budget on
apparent success/failure. Evaluate the proposed generic multiresolution credit-
sharing hypothesis separately if this cheap isolated step-size screen stays flat.
Final holdout seeds remain untouched until settings and acceptance criteria freeze.
