# Context-dependent local sensory selectivity

The joint 256-example memory fails appearance transfer despite success in
separate specialists. Compare one structurally different retrieval rule: learn
feature selectivity within the 32 nearest observed sensory contexts, rather than
forcing one global covariance metric onto every relation. Neighborhood admission
uses variance-normalized sensory distance. The local pre/post covariance then
selects the predictive features for four-neighbor association. If local outcome
covariance is zero, retain sensory distance rather than inventing a prediction.

The neighborhood is selected from observed patches/directions only. No relation
label, object label, hidden mask or task-specific routing. Keep 256 stored examples
and the same joint training/held-out scenes. Test an XOR-style unit example where
global linear covariance is inadequate. Do not sweep neighborhood sizes; accept
or reject against both existing relation gates.
