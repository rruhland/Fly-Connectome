# OFF sign audit and bounded predictive kinetics experiment

User authorization: inspect OFF target -> precursor spikes/current -> recurrent
contribution -> local weight update, then try distinct pathway time courses if
the audit does not reveal a corrective sign error.

## Sign audit

Select the first three recurring OFF cases in the fixed precursor schedule,
before inspecting their outcomes. Replay one trial using fresh warmed copies of
both initial and frame-horizon-trained weights. Enable the actual frame-horizon
rule to observe local update proposals; never modify the source artifacts.
Capture every tick from 24 ticks before each event to eight ticks after it.
Record unsigned source spikes, transmitter signs, delayed arrivals, per-edge
signed currents, camera target, retained issue-time eligibility/prediction, and
actual weight proposals. Reconstruct currents by decaying each edge's historical
contributions and adding sign times weight AT ARRIVAL, not current weight times
a spike trace. Check agreement with the actual raw recurrent current within
1e-6; bound any omitted warmup residue below that tolerance in audited windows.

Confirm OFF maps to +1 under the existing contrast encoding. Positive error
should increase an excitatory magnitude and decrease an inhibitory magnitude;
the latter is a positive change to signed predictive drive. This audit is a
small online replay, not a frozen-performance measurement. Report all six cases.

## Slow-excitation-v1

Fixed hypothesis: existing excitatory and inhibitory precursor signals decay too
similarly to separate the two phases. Change only predictive-current decay to
20 ms for positive edges and 5 ms for negative edges. Preserve their arrival
amplitudes, fixed signs, measured connections/delays and initial magnitudes.
These constants are diagnostic choices, not fitted or established physiological
values. Since arrival amplitude stays fixed, longer decay also increases total
excitatory impulse area (about fourfold); this experiment does not isolate
temporal shape from integrated drive.

Use two local signed current stores per neuron. All arrivals remain sign times
current weight, and both currents enter the same existing membrane equation.
Local visual eligibility must follow the corresponding edge's decay, so the
retained frame-horizon credit remains matched to its physical predictive current.
Keep eight ticks/frame, frame-horizon-v1 supervision, quiet targets, learning
rate, weight synchronization/bounds, membrane/rest/adaptation/sensory dynamics,
feedforward/behavioral currents and R-STDP unchanged. No backpropagation or new
connections/readout. This is an opt-in reference-runner experiment; native
engines and production checkpoints do not support its extra current state.

First test impulse-current and eligibility formulas, sign preservation and
unchanged default common decay. Frozen precursor reconstruction must include
warmup spikes: the first attempted run stopped before training when its longer
tail exposed the old diagnostic's omitted warmup history. Fix the diagnostic,
not the 1e-6 tolerance, then repeat the same protocol.

Run the same crop, seeds, 200 training trials and 50 frozen evaluation trials as
frame-horizon-v1. Use unchanged fixed ON/OFF/quiet gates, and compare both each
model's own initial frozen baseline and the prior learned model. No parameter
sweep or longer training in this experiment. If it fails, document which errors
changed rather than silently continuing architecture search.

```
.venv/Scripts/python scripts/temporal_visual.py --visual-schedule frame-horizon-v1 --predictive-kinetics slow-excitation-v1 --output runs/temporal-slow-excitation-v1-warmup
```
