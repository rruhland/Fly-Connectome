# Proposed first live local timing learner

## Decision this proposal requests

Approve a **bounded experimental implementation** of the shared-magnitude,
gate-consistent rule on the existing small measured visual motif. Each existing
predictive edge retains one nonnegative magnitude and fixed transmitter sign.
No connectome edges, error network, backpropagation, labels, phase/tempo input,
or change to behavioral R-STDP are introduced. Production plasticity and the
saved benchmark checkpoints remain untouched.

The frozen-history probe indicates this is the minimum useful change: it
passes pooled held-out ON/OFF and quiet criteria. Its tempo2 OFF response
remains below .1, so this is a test of live learnability, not a declaration
that the architecture already satisfies M1A. A two-magnitude synapse is
deferred until the simpler coupled experiment establishes its need.

## Local state and timing

At each camera-frame boundary, each directly sensory-driven postsynaptic
neuron stores its current sensory state. When a local observed event arrives,
it sets an expected sensory magnitude to the absolute state stored one frame
earlier. The expected magnitude persists until another local event. The
current frame's issue opens the local comparator when abs(current sensory
state) is within a half-frame decay margin of that reference, exactly as in
the causal diagnostic. Before acquiring a reference, the comparator is closed.
The event updates this reference before a *new* forecast is issued but never
changes the forecast that was issued eight ticks earlier.

Use the existing recurrent predictive current as the amplitude source. The
comparator gate determines whether this current constitutes an issued sensory
forecast for learning and scoring; recurrent current still participates in
the neuron's membrane dynamics on closed-gate frames. This preserves the
tested neuronal dynamics at zero learning rate, while adding a local decision
about when its current is a sensory prediction. A closed gate does not sever
the anatomical synapse.

At issue, snapshot the gate, encoded recurrent current and existing causal
visual-edge eligibility. At confirmation eight ticks later, compute

```text
issued_p = gate * clip(recurrent_current / threshold, -1, 1)
issued_e = gate * existing_edge_eligibility
delta_w = eta * local_gain * (observed_event - issued_p) * issued_e
```

`local_gain` is one for quiet confirmations. For an event, it is the ratio of
prior local quiet to event counts, bounded to [1,8], both counts initialized at
one. Advance counts only *after* the corresponding confirmation update.
These are local counts at the postsynaptic neuron, not global class frequencies
or future labels. Current diagnostic uses eta1 for shared balanced; live eta
must be calibrated only on training/validation sequences because coupled
feedback may change its stable range. Keep magnitude bounds [0,10].

A closed-gate unexpected event updates the sensory timing reference and the
local event count. It gives no amplitude credit, consistent with the fact
that no forecast or predictive eligibility was issued. Repeated motion can
reopen the gate; omission recovery will be reported separately. Whether this
missed-event behavior is sufficient is part of the live test.

## Implementation scope

Add an experimental `FramePrediction` subclass and a small headless runner
under `scripts/`; do not alter the production `Plasticity` rule or generic
network. The subclass keeps a per-neuron previous sensory state, reference,
gate and two local counts. It applies the new rule only to the scored motif's
directly sensory-driven L3 neuron; all other synapses stay frozen for the
first causal comparison. This isolates the target from other concurrent
plasticity while retaining all existing recurrent neuronal dynamics.

Start from the original magnitude on the target's existing predictive edges
and the saved mixed-training weights elsewhere. Run a frozen eta0 parity
control against the prior area-matched model, followed by one-pass live
training on repeated 2/4/6-frame motion. Hold out unseen tempo3, varied
blank lengths, and the continuous tempo-switch/omission challenge. Do not
reset neural or local timing state at trial boundaries. Report each polarity,
quiet alarms, MSE, persistence comparison, firing stability, weight bounds,
and elapsed frames/seconds. Record issue-time gate, current, eligibility and
post-confirmation updates so exact causal ordering can be audited.

Unit tests must cover a closed-gate unexpected event, reference acquisition,
prediction/eligibility masking, count ordering, bounded updates, and no
changes to non-target weights or signs. A live result is a success only if
held-out forecast error decreases against the initial network and persistence,
both anticipation signs clear .1, quiet alarms stay <=5%, and firing remains
stable. The motif remains a precursor to Pong M1A, not the milestone itself.

If the coupled network fails, inspect neural trajectories and the recorded
credit chain before modifying the rule. If the shared rule meets pooled means
but fails a tempo or the continuous challenge, draft a separate proposal for
two local magnitude channels per existing edge and jointly specify how those
magnitudes affect physical recurrent current as well as the comparator. The
offline split result alone does not authorize that architectural change.
