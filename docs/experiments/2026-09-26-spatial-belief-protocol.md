# Bounded spatial-belief experiment

Registered before training/evaluation. Production remains unchanged. This tests
the original future-spatial-belief objective; it does not erase the failed point
forecast gate or establish event-only acceptance.

Keep the statistical observation state and its declared every-sample grayscale
image plus event sensor. Store up to 2,048 causal observed history/outcome pairs
per horizon using deterministic reservoir sampling. Normalize four displacement
samples by their mean speed and mean heading (last nonzero heading if the mean
cancels). Retrieve 32 nearby histories; preserve their separate outcomes as a
Gaussian mixture with a one-pixel observation floor. No backprop, motion-family
label, truth position, or future evidence at query time. Credit arrives only
when the same visual identity is actually observed at the forecast deadline.

Train on the existing 192-stream random-angle/speed protocol. Freeze, then score
fresh L/T/line shapes, phases 31/37, angles 12/48/132 degrees and scales .8/1.3.
Use 216 scenes and horizons 1/4/8. Training outcomes are observed positions;
renderer truth is scoring only. Keep scene boundaries, missing observations,
and identity checks explicit. Report missing coverage separately.

Compare conditional outcome retrieval, shuffled key/outcome associations,
unconditional outcome retrieval, and persistence/last/mean/two-lag/acceleration
forecasters with residual distributions calibrated on the same training data.
All distributions use the same Gaussian floor. Score log density (NLL), point
mean error, and marginal probability integral transforms / 90% interval coverage.
No held-out parameter selection. One configuration only.

Promising distribution criterion: at horizons 4 and 8, improve mean NLL by at
least .2 nat over the strongest calibrated fixed control and shuffled retrieval;
no dynamics family more than .2 nat worse than its strongest fixed control;
90% marginal interval coverage between .85 and .95 pooled, and observation
coverage at least .95. Report every family. Passing this is evidence for an
explicit distributional acceptance revision, not automatic milestone completion.
Also test a two-outcome identical-history fixture to verify retained modes and
causality. Do not tune weak results; reassess architectural information limits.
