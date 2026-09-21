# Does the existing local state contain event timing information?

User requests an information diagnostic before any further timing mechanism.
Use the trained area-matched 20 ms excitatory / 5 ms inhibitory model. Freeze
all neural weights and dynamics; fit only an explicitly offline diagnostic.
No diagnostic classifier becomes a neural readout or gate in this experiment.

## Causal observations

At the existing forecast-issuance tick (first tick of each camera frame), record
target L3 82450's current local information after processing that tick:

- Signed prediction current and its excitatory/inhibitory components.
- Sensory, feedforward and behavioral currents; current local input increments.
- Membrane voltage, adaptation, refractory state, own current spike.
- Existing local post-spike trace and firing-rate state.
- Each existing incoming predictive edge's eligibility, arrival trace and
  arrivals at the current tick. No remote source voltage, future source spikes,
  global activity or nonincident edge state.

Use the existing FramePrediction machinery with zero prediction/reward learning
rates and zero homeostasis to observe traces without changing the model. This
shadow observer is initialized as in prior evaluation (fresh state after neural
warmup); it reconstructs local learning state that the normal online rule would
maintain, not new temporal features. Verify recorded target, forecast and full
network spike arrays exactly match the previous frozen evaluation.

The label is whether an event occurs at +8 ticks. Future target, trial ID and
motion phase are used only for labels, splitting and reporting, never features.
Keep the original first-cycle cue exclusion and all other quiet frames. Distinguish
each adjacent quiet phase and intertrial blanks in the reports.

## Fixed offline probes

Use a fixed 15-nearest-neighbor binary event detector, standardized using training
statistics only; discard constant training features. Four predeclared subsets:
prediction currents alone; all cell-level state; synaptic state plus currents;
all local state. Features at distinct incoming synapses are distributed local
information; success there alone would not prove a postsynaptic cell can access
that vector without a mechanism. The cell-only comparison addresses this gap.

Split existing seed9023 evaluation by whole trials: 1-25 fit, 26-35 calibrate,
36-50 held-out test. Calibration selects the threshold maximizing the weaker
ON/OFF event-detection recall subject to at most5% quiet false positives, with
fixed tie breakers. Report held-out and fresh-seed results without retuning.
Fresh confirmation: 50 frozen trials with randomized blank seed9024 and the same
motion task. This is not a test of novel direction, speed or trajectory geometry.

Report ON and OFF recall separately, quiet false-positive rate, precision,
event-vs-quiet AUC, and false-positive rates in each near-event quiet phase.
Above90% recall in both polarities at <=5% quiet false positives on independent
confirmation would be useful positive evidence of accessible timing information;
the values and phase breakdown matter more than a binary declaration.

Controls: shuffled training labels with otherwise identical fitting/calibration;
1% training-standard-deviation perturbation of test features (a numerical
sensitivity check, not a calibrated biological noise model). No parameter sweep,
backpropagation, neural weight changes or new architecture. A failed diagnostic
does not prove that no possible classifier could separate these finite states.

```
.venv/Scripts/python scripts/local_information.py
```

If low-dimensional existing cell state separates the classes robustly, the next
step is designing a local expression mechanism. If probes overlap substantially,
inspect what information is missing before adding a temporal representation.
Even a successful offline probe neither passes M1A nor proves local learning
can discover the gate.
