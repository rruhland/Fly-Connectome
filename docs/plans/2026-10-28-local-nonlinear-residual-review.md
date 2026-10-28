# Draft review: localized nonlinear policy residual

Status: ready for user review; not approved or implemented. The opening-retention
trial completed at403 versus409 returns, with all143 known successful openings
preserved and one additional opening recovered. Later returns regressed by seven.
This is interference evidence, not a proof that the current policy cannot learn.

## Problem and hypothesis

The existing linear head can learn complete reward-earned later-return sequences,
but changes often disrupt many other rallies. Preserving all earlier successful
actions did not produce a verified fit within three bounded attempts. Soft score
preservation made fitting fast, but its first whole-bank candidate regressed
409→391 returns. Those results do not prove that the linear policy is incapable.

A small nonlinear residual may express a correction in a limited region of visual
experience without changing decisions across the wider state distribution. The
current inputs already combine generic visual positions, directional velocities,
forecasts and image context; a nonlinear readout can combine these quantities.
It cannot reconstruct information lost by the encoder or distinguish identical
encoded inputs requiring different behavior.

## Concrete bounded proposal

Keep the existing frozen vision, scaler, actuator and learned linear head.
Add64 fixed radial-basis features of the same legal visual input. Take a fixed-seed,
episode-balanced sample of2,048 current native TRAIN observations, select64
farthest-point centers using the existing diagonal feature scaling, and set one
bandwidth from the median positive distance to each center's eight nearest centers.
No reward labels, new donor trajectories or physics labels select the centers.
Calibrate activity on TRAIN only. Record nonlinear variation beyond a fitted linear map; do not repeat the
previous nearly linear64-tanh branch without checking its redundancy.

Initialize all new output weights to zero: the complete carried policy is exactly
unchanged. First train only the3×64 residual output weights using the existing
own-reward donor trajectories and no backpropagation. The centers and bandwidth
remain fixed during this comparison. No score/ball-coordinate/opponent-state
inputs enter the policy. Existing raw vision features remain its sole sensor data.

Use the saved shortest new later-return donor and known opening-prefix retention
contract for a single bounded comparison. Verify initialization action parity,
record fit time and inference overhead, and test actual fixed TRAIN sentinel,
full160 if useful, then new games. A fitted witness is insufficient for acceptance.
Compare against the preserved3ac7f87f head; keep every old checkpoint immutable.
Measure nonlinear novelty on separate TRAIN episodes from those used to fit the
linear explanatory map; do not use an in-sample fit of2,048 observations with4,389
linear inputs as evidence of redundancy. Record activation coverage, saturation
and effective rank before spending native cameras. These are TRAIN diagnostics,
not new-game policy qualification.

Use a fixed minimum empirical residual-score-change objective with the documented
normalized1%ridge, the whole reward-earned donor and known opening prefixes hard.
Require exact original-row and deployed-action certificates. If no verified fit
exists within the budget, perform no native run and no bandwidth retry.

No architecture/bandwidth sweep follows automatically. Proposed execution budget:
10minutes including fitting and native checks; stop this branch on clear global
regression or unavailable useful nonlinear variation.

## Approval boundary

This adds a nonlinear policy branch, so it is a major architecture experiment
reserved for user review. The earlier approval concerned a specific64-feature
residual test already completed; it is not treated as blanket approval for this
new construction. The parent and independent Sol review now recommend this bounded test. The prior
velocity diagnostic already showed a possible smoothing benefit in one case, but
other misses had usable visual information; velocity cleanup remains a separate,
smaller alternative and is not evidence that a nonlinear branch is necessary.
