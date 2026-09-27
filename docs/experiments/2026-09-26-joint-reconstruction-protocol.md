# Joint observation reconstruction: bounded comparison

Registered before implementation/evaluation. Production is unchanged.

Two routes, one configuration each:

1. **Event-only generative patches:** learn 32 nonnegative space-time atoms over
   four event frames and 5x5 neighborhoods with competitive local residual
   updates. Infer sparse coefficients from the joint recent evidence and
   reconstruct the current event slice, including missing evidence. Integrate
   reconstructed signed changes into the visible surface. This is a sensory
   prior, not the final world-state or a hand-built motion-feature bank.
2. **Sparse-frame local reconstruction:** reconstruct current signed contrast
   from local observed event history and previous reconstruction. Update a
   shared local linear readout by normalized error times presynaptic features
   only when a declared visible image arrives every eight frames. Anchor to that
   same image. No backpropagation or hidden labels; no intermediate frame
   teacher. The hybrid contract is explicitly distinct from event-only sensing.

Training: 64 generic streams, deterministic mixture of motion strides 1,4,8.
Use only observed events for the first route and declared sparse frames for the
second. Noise is the previously registered .1 dropout/.001 false-event rate.
Inference must be causal. Compare learned and frozen reconstruction with the
same aligned/shuffled-context controls where useful.

Keep the surface/context architecture and all gate thresholds fixed. Report
clean context, fresh familiar/heavy noise (seed base 50000), constant motion,
two movers, and matched continue/disappear scores. A passing candidate must meet
the existing joint gate; isolated improvement does not count. Hybrid evaluation
also includes noise on sampled images (independent .01 pixel flips), separately
reported. One implementation-bug repair pass is allowed; no hyperparameter
sweeps or post-result changes to dictionary size, temporal span, learning rate,
sensor cadence, thresholds or readout features.

If both fail, close these particular implementations and reassess at the level
of observation-model assumptions or the benchmark's information demands. If one
passes, require fresh scene families and the outstanding generic-state and
forecast gates before proposing production integration.
