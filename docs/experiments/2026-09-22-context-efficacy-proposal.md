# Proposed context-dependent efficacy on existing visual synapses

## Decision requested

Approve a bounded experimental extension of the measured L3 motif: each of
its **existing 12 predictive incoming edges** has two nonnegative magnitude
components, selected by the postsynaptic neuron's local sensory-state sign
when a forecast is issued and when recurrent current is integrated. Edge
identity, presynaptic neuron, postsynaptic neuron, fixed transmitter sign,
delay, synaptic waveform and graph topology stay the same. There is no new
neuron, connection, error network, backpropagation, teacher tempo/phase label,
or change to the behavior-learning rule. Production model code and checkpoints
remain untouched. Treat this as a test of state-dependent efficacy at the
same anatomical synapse, not proof of a particular molecular mechanism.

The need is empirical: the approved one-magnitude live rule improves MSE but
fails OFF at dwell2/6 and gives wrong-sign ON in steady continuous dwell6.
There, the same L2 magnitude supplies -.609 signed ON and +.057 signed OFF;
opposite training updates compete on that scalar. ON/OFF issue contexts are
locally distinguishable. A frozen-history split model transferred better,
but this proposal requires a new **live coupled** validation.

## Local expression and physical current

Use context0 when postsynaptic sensory state is <=0, context1 when >0.
The sign is read after the current frame's local sensory injection and before
the newly issued forecast. Context selection is local; it is not the target
event polarity or an externally supplied motion label. Each edge's fixed
transmitter sign multiplies either magnitude, and magnitudes remain in [0,10].

The two magnitude components must affect the physical predictive current,
not merely an offline readout. Because synaptic current is linear in edge
impulses, keep four exact current sums for this one target: excitatory and
inhibitory traces under each context's magnitudes. Decay both context sums
on every neural tick with the already established 20 ms excitatory and5 ms
inhibitory kernels, add the same anatomical arrivals to both using their
respective magnitudes, then integrate the *selected* sum into the neuron's
membrane and expose it as the local recurrent prediction. This factorization
retains the information needed for each context's total physical current;
the existing per-edge eligibility still carries edge-specific credit. Other
neurons use the original area-matched network unchanged.

When both context magnitudes are identical, the new experimental network
must reproduce the previous network's spikes, signed currents, forecasts,
eligibility and sensory state on the same stimuli within the exact floating
point tolerance established by a parity test. No training should begin until
that comparison passes. The local timing reference/window stays unchanged.

## Causal local update

At a frame issue, snapshot the sensory context, timing gate, encoded physical
recurrent current, and existing incoming edge eligibilities. At the
confirmation eight neural ticks later, update **only the context component
that was selected at issue**:

```text
issued_prediction = gate * clip(selected_recurrent_current / threshold, -1, 1)
issued_eligibility = gate * existing_causal_eligibility
delta_magnitude[issue_context, edge]
  = eta * prior_local_event_gain * (observed_event - issued_prediction)
    * issued_eligibility[edge]
```

The bounded prior quiet/event count gain and timing-reference update follow
the approved shared-magnitude rule. A closed gate still has no amplitude
credit, but an unexpected event updates the local reference and event count.
Only existing edge magnitudes and local synaptic state are plastic. No
gradient passes through a neuron or through the context selector.

## Bounded implementation and checks

Add experimental network/plasticity subclasses and a headless runner under
`scripts/`, leaving the production network and plasticity unchanged.
The initial two components both equal each edge's original magnitude;
non-target weights start from the saved mixed-training source and stay frozen.
Run the same 42-trial one-pass training stream, nine-trial validation stream,
held-out dwells2/3/4/6 and standard/omitted continuous challenges as the
approved shared-rule test. Calibrate a small rate grid on training/validation
only; keep tests untouched until selection. Compare with the already saved
shared-rule results using identical stimuli and initial weights.

Required tests: equal-component neural parity; state-sign/context selection
at issue; incoming edge order/sign; context-specific physical current;
closed-gate surprise; only issued-context credit; exact update reconstruction;
no non-target changes; 0..10 bounds; finite neural state; no trial-boundary
reset; and measured throughput. Report ON/OFF and quiet separately at every
tempo, especially steady dwell6 and omission recovery. A result is useful
only if the live model clears both signs without quiet alarms exceeding5%
and improves held-out MSE over initial and persistence. This still precedes
Pong M1A and cannot itself establish full vision learning.

If the equal-component parity fails, investigate the current factorization
and update order before any training. If the learned split still fails at
slow tempo, preserve the evidence and revisit temporal representation or
credit rather than adding further state without a new diagnosis.
