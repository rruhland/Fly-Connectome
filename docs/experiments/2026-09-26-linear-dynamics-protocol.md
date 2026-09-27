# Direct local linear association control

One distinct learning principle, with the same observed data and forecast gates:
replace competitive nearest-history prototypes with a four-lag linear readout.
Accumulate local lag/lag and lag/observed-displacement correlations, solve the
four-dimensional regularized normal equations, and share temporal coefficients
across the two image axes. This gives rotation equivariance without direction
labels or named trajectory families. No backpropagation; no optimizer sweep.

Use a fixed 0.001 diagonal prior, the same horizons, 24 training scenes and 36
transfer scenes. This is an engineered local association readout, not a claim of
synapse-local biological implementation. Compare directly with the preserved
competitive learner and fixed controls. If weak, reject rather than tune.
