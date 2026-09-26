# One contrast-belief inference experiment

The polarity-return teacher destroys clean state and acquires no contextual
updates; reject it without a window sweep. Retain the original causal local
agreement observer weights and its complete internal dynamics.

Test a different inference principle, not another teacher: event reliability
should express uncertainty about a state transition, rather than simply scale
the signed intensity increment. Maintain per-pixel probabilities for negative,
baseline and positive contrast. An ON event shifts mass toward positive and an
OFF event toward negative, mixed with the unchanged state according to the
original observer's credibility. Simultaneous conflicting events leave their
joint mass unchanged. Export mean signed contrast to the frozen surface state.

This is an engineered three-level sensory model appropriate to the current
binary simulator, not a general radiometric model or learned world-state. It
adds no motion/object/occluder rules and uses no frames in the primary arm. Its
probabilities are model beliefs, not demonstrated empirical calibration.

Use the same learned event stream as the baseline: keep the original observer
internals intact and replace only the exported contrast planes. The same local
context learner is trained from observed reacquisition. Run aligned, shuffled,
and frozen controls once against every fixed gate. No clipping/threshold/prior
or state-count sweep after the result. If it fails, close this sensory batch and
record the architectural bottleneck and next broad alternatives, rather than
adding more local filters.
