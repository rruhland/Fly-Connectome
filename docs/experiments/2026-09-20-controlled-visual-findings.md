# Controlled moving-dot experiment

The unchanged model did not learn useful anticipation on this run. More
importantly, the selected crop is not a clean test of general learning capability:
one target cannot represent one required event polarity, and some potentially
useful prediction sources never fire. This feasibility check should have happened
before training. Do not interpret the failure as proof that local learning cannot
work, or as justification for another full-Pong run.

## What ran

The [pre-run protocol](2026-09-20-controlled-visual-protocol.md) fixes the stimulus,
selection, scoring and thresholds. The induced measured graph contains 112 neurons
and 509 edges, spanning nine neighboring retinal columns and their two-hop return
paths. No retained weights, signs, delays, neuron parameters, retinal coordinates
or learning rules were changed. Cropping omits 3,251 incoming and 6,744 outgoing
boundary edges; these missing drives are a material limitation.

Train 200 traversals (6,919 frames), evaluate on 50 separate traversals (1,734
frames), predicting the next camera event 8.33 ms ahead. The three target bodies
are L1 38366, L2 39087 and L3 114687. The actual event camera and neural spikes
drive learning. Future labels are used only for offline evaluation.

The first execution took 112.11 seconds. Review added raw-state stability checks
and removed unavailable frozen eligibility arrays. One identical verification
replay took 67.52 seconds. Combined execution was 179.63 seconds, within the
600-second budget. There was no parameter sweep. Both runs produced bit-identical
forecasts, targets, spikes, arrival counts and final learned weights. During the
verified run all checked raw states were finite and weights stayed within bounds.
Mean evaluation firing was 3.92 Hz initially and 3.86 Hz after training.

## Prediction outcome

| Predictor | All-frame MSE | ON-event MSE | OFF-event MSE |
|---|---:|---:|---:|
| Frozen initial network | 0.05782128 | 1.00112247 | 0.99913758 |
| Trained network | 0.05772970 | 1.00001264 | 1.00056744 |
| Zero | 0.05770341 | 1.00000000 | 1.00000000 |
| Last-event persistence | 0.17311022 | 1.00000000 | 4.00000000 |
| Causal stimulus control | 0 | 0 | 0 |

There are 5,199 scored neuron-frame samples, including 150 ON and 150 OFF events.
The stimulus control uses only the currently visible pixel and fixed movement
law; its exact predictions verify task predictability without feeding a teacher
or simulator state to the neurons.

Trained error improves only about 0.16% over frozen and remains worse than zero.
Quiet-frame MSE drops from 0.00011713 to 0.00001015; event MSE slightly worsens.
The secondary one-neural-tick score is also effectively zero-predictor performance
(0.05769400 versus zero 0.05770341), not meaningful anticipation. The fixed
acceptance gate fails. M1A remains unmet and M1B remains blocked.

## Directly observable limitations

**L3 cannot express the positive forecast in this crop.** Its five retained
predictive sources are four Dm12 neurons and C2, all inhibitory. With fixed signs,
nonnegative weight magnitudes and the current signed-current forecast, their sum
cannot become positive. A positive OFF target therefore cannot be predicted at
this cell even if these sources begin firing. In this run they supplied zero
arrivals, so L3's prediction and eligibility stayed zero despite L3 emitting 586
spikes during training. This is not total sensory-network silence.

**L1 lacks an active source for negative forecasts in this run.** Its only
inhibitory predictive source, C2 332251, supplied zero arrivals. The active Mi1 and
Tm3 inputs are excitatory. They cannot produce the negative ON forecast through
their own positive currents. Their initial magnitudes 0.095 and 0.060 shrink to
0.0012432 and 0.0003105. This is an activity limitation observed here, not proof
that C2 can never fire in another circuit context.

**Updates mostly suppress the predictions that do occur.** Across the incoming
target edges, summed local proposals are -0.12377 at ON ticks, +0.11799 at OFF
ticks, and -0.45127 at quiet ticks. These are proposals before weight clipping and
homeostasis, not net changes in weight. For example, C3-to-L2's quiet contribution
is -0.22440, versus +0.00374 at ON ticks. Tm2-to-L2 has a positive observed-target
term (+0.00890), but its prediction penalty totals -0.05104. Quiet suppression is
an outcome of the recorded rule and activity; it alone does not establish that
removing the penalty would yield useful prediction.

Seventy of 112 neurons spike during training, producing 24,701 spikes. Complete
small-run spikes, predictions, actual observations and training eligibility are
saved locally. The [edge diagnosis](2026-09-20-controlled-visual-diagnosis.json)
lists all retained incoming target edges, signs, source activity, weights and
ON/OFF/quiet update contributions.

## Decision and next boundary

Stop this attempt without changing the model or launching another training run.
The next controlled test needs a measured motif with a **sign-compatible
prediction path and an observed precursor response before the target**. Check
those conditions with a short frozen preflight before spending any training
budget. Select the stimulus and target from that anatomical motif explicitly;
do not select a target retrospectively because its training score improved.

If the preflight cannot find such a path in the bounded motif, that is a circuit
or target-representation problem to discuss before modifying learning. If the
preflight succeeds but training fails, the same trace will provide a valid basis
for testing one local-rule change. Neither outcome requires another full-Pong
search. No synthetic connections, backpropagation, prediction head, sign changes
or new biological dynamics have been introduced.

## Reproduction and verification

Run `.venv/Scripts/python scripts/controlled_visual.py --output runs/controlled-visual-v1-verified`
from the repository root. The command expects the existing initial checkpoint;
its checksum and all retained parent-edge indices are in the
[results](2026-09-20-controlled-visual-results.json). The original checkpoint was
verified unchanged. The results also contain artifact checksums for the full
local NPZ traces, raw-state maxima and per-target metrics. Frozen artifacts omit
learning eligibility because no plasticity process runs during evaluation.

Verification: 205 tests pass, four CUDA tests skip on this CPU-only host. Focused
review confirms causal alignment, fixed crop parameters, frozen evaluation and
the corrected artifact/stability checks. No source model changes or checkpoint
mutations were made.

Diagnosis uses the recorded preceding-tick eligibility. For each incoming edge,
the observed-target term is the sum of `eta * target * eligibility`, and the
prediction penalty is the sum of `-eta * previous_prediction * eligibility`.
Their sum matches the local predictive proposal, before clipping/homeostasis.
No inference about the full network's missing boundary drives follows from these
cropped-circuit measurements.
