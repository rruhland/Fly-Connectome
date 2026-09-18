# Event-aligned local prediction: proposed M1A correction

Status: approved by the user on 2026-09-18; implementation in progress. An audit
of the approved signed-current rule found a target mismatch and limited active
feedback. Existing experiments/checkpoints remain authoritative records of that rule.

## Evidence

The network trains against `clip((feedforward_current + sensory_state)/threshold)`
each neural tick. The event metric instead tests that forecast against an unfiltered
unit camera event at each frame boundary. With gain 30, sensory decay 20 ms and
dt 1/960 s, one ON event leaves current -19.7772 at the next frame (8 ticks later).
The local target is still -1, while the event target is 0. Saturation lasts about
65 neural ticks. Even a perfect filtered-current predictor fails the event task.
Rescaling the already clipped output cannot undo this loss of information.

Frozen development seeds 1101/1102 after 500 training steps show event-conditioned
MSE about 1 (the zero predictor's value). Only 2/892 L1 neurons receive spikes from
an inhibitory predictive source, despite 884 having such measured connections.
The main inactive inhibitory L1 partner is C2; C3 supplies substantial L2 feedback.
Thus the sign-preserving circuit currently lacks useful active pathways for many
predictions. Merely making a signed output possible did not make learning effective.

The present eligibility also mixes arrivals from the current tick into an error
for the previous forecast, and retains 90.5% of an arrival after 100 ms although
its 5-ms synaptic current retains only 2.06e-9. These are properties of the approved
rule, not newly discovered coding regressions. A longer unchanged continuation to
2,000 training steps is being tested as a control.

## Recommended bounded experimental revision

Keep the same conceptual local equation `delta_magnitude = eta * delta * eligibility`.
Explicitly version two visual-only choices, with old settings retained for old
checkpoints and separate ablations for target-only, trace-only and combined changes.

1. **Predict newly arriving local input rather than its decaying tail.**
   At sensory-facing neurons, the target is the current signed camera-derived
   injection divided by the fixed sensory gain (therefore -1, 0 or +1). At other
   visual neurons, it is the newly arrived feedforward current divided by baseline
   threshold, clipped to [-1,1]. These increments are already available at the
   postsynaptic neuron; they are not simulator coordinates or a new error network.
   Forward membrane/current filtering stays unchanged. Predictive current still
   provides `p_j = clip(recurrent_current_j / threshold_j, -1, 1)`.

   At tick t, compare `target_j(t)` with `p_j(t-1)`. Score this exact forecast against
   the same signed camera event at frame boundaries. Intermediate sensory ticks
   have target zero. Retain old current-proxy metrics separately, with explicit
   names, rather than silently replacing the previous experiment's metric.

2. **Assign visual error to the trace that produced the forecast.**
   Score using the visual eligibility stored with the previous prediction before
   adding current arrivals. Then decay the visual trace with the postsynaptic
   synaptic-current time constant and add signed arrivals for the next forecast:

   ```
   delta_j(t) = target_j(t) - p_j(t-1)
   delta_magnitude_ij(t) = eta_prediction * delta_j(t) * e_ij(t-1)
   e_ij(t) = exp(-dt/tau_current_j) * e_ij(t-1) + sign_ij * arrival_ij(t)
   p_j(t) = clip(recurrent_current_j(t)/threshold_j, -1, 1)
   ```

   This is a local causal tag on an existing synapse, not backpropagation or a
   learned delay. Behavioral pair traces, eligibility decay, reward curriculum and
   R-STDP stay unchanged. Intrinsic current remains outside target and prediction.

3. **Check the existing feedback circuit's operating point.**
   Trace C2/C3 and the other measured lamina feedback partners before selecting
   any fixed class currents. If necessary, preregister a small bounded calibration
   using frozen ON/OFF, motion and no-event probes, then verify input dependence
   by temporary evaluation-only silencing. No direct sensor input beyond L1-L3,
   invented edges, sign changes, automatic firing floor or Pong-score tuning.
   The current-based LIF architecture remains unchanged. Values are model
   approximations, not claimed physiological measurements.

## Verification and honest acceptance

- Hand-check unexpected/confirmed/expired predictions of both signs, including
  first-ever arrivals that cannot receive credit for an earlier forecast.
- Test an isolated event: target is zero after injection even while sensory
  current persists; confirm the forward trajectory is unchanged by target choice.
- Demonstrate delayed-association learning on a deterministic small fixture before
  spending on the measured graph. Fixtures verify software only.
- Test exact resume, old checkpoints, sign/topology preservation, reward isolation,
  duplicated/permuted batches and CPU/CUDA where hardware is available.
- Preregister development settings and compare target-only/trace-only/combined
  versions on development seeds. Use untouched final seeds for acceptance.
- Require held-out improvement over the same initial network, persistence and the
  zero-event predictor, plus improved event-conditioned prediction and appropriate
  temporally shuffled controls. Do not count a reduction in tonic false positives
  alone as event anticipation. Record sparse activity and no-event behavior.
- Keep the selected settings fixed across anatomical thresholds. Continue M1B from
  compatible learned visual edges only after honest M1A evidence.

The user approved the changed learning target, visual credit timing and bounded
experiments. It does not guarantee success on the measured graph. If the bounded
revision fails, report the result rather than introducing an artificial predictor.

## Biological context for feedback, not parameter values

C2/C3 are known GABAergic centrifugal neurons with medulla/lamina connectivity;
their existence and transmitter identity do not imply that the model's current
values are biologically measured. See
[Kolodziejczyk et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2373871/) and
[Tuthill et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3806040/).
The actual partners and signs used here come from the pinned MaleCNS graph.
