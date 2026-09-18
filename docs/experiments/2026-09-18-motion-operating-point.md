# T4/T5 silence: input-current diagnosis and bounded correction

The user requested diagnosis and repair on 2026-09-18. This work retains the
approved current-based adaptive LIF model, measured topology, signs, local updates,
and exclusive L1-L3 event injection. It uses the existing fixed class-parameter
mechanism; no artificial motion detector or forced firing floor is introduced.

## Root cause evidence

`scripts/trace_motion_inputs.py` replays a frozen ON edge, OFF edge and matched
no-event condition. It separately reconstructs excitation and inhibition from
actual synaptic arrivals, and checks their sum against the network's currents.
With the original resting profile, all 3,438 T4 and 3,356 T5 neurons remain silent.
Their maximum recorded voltages are 0.114 and 0.123 against threshold 1. The maximum
excitatory current is below 0.48 in both populations. T5 receives no active
inhibition in this probe, so inhibitory cancellation cannot explain its silence.

The expected measured pathways exist and deliver spikes: Mi1/Tm3 into T4 and
Tm1/Tm2 into T5. The profile supplies intrinsic current to first-order medulla but
leaves motion neurons at zero. With this synaptic gain and these transient inputs,
the voltage cannot approach threshold. This is a model operating-point mismatch,
not an absent-edge or spike-delivery failure. Mi4, Tm4 and Tm9 are also silent;
correcting T4/T5 output alone does not establish complete motion computation.

## Biological scope

Electrophysiology supports excitation and inhibition acting around a nontrivial
T4 operating point: Mi1/Tm3 provide cholinergic input, and Mi9-mediated inhibition
affects resting potential and input resistance. The experiments concern graded
voltages and conductances, not our normalized spike threshold.
[Groschner et al., 2022](https://www.nature.com/articles/s41586-022-04428-3).

Measured T5 anatomy includes Tm1, Tm2, Tm4 and Tm9 excitatory inputs.
[Shinomiya et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6338461/).
These sources motivate checking synaptic integration and operating points; they
do not provide the numerical resting currents used below or validate the LIF
approximation. Conductance dynamics remain outside this correction.

## Preregistered correction experiment

`configs/motion-resting-calibration-v1.json` fixes currents 0, 0.85, 0.90, 0.95 for
all eight annotated T4/T5 classes. Everything else is held fixed. Each value is
below threshold, so it cannot generate spikes in an isolated neuron. Selection
uses the smallest value giving stimulus-dependent T4/T5 responses and passing the
existing numerical stability bound. No Pong score, reward or parameter fitting is
involved. The same class table must apply across T1/T3/T5 thresholds.

Independent validation repeats multiple directions/speeds and local flashes,
compares no-event controls, and silences the non-motion neurons with actual inputs
to T4/T5. The latter tests that restored spikes depend on the existing circuit.
Direction and polarity tuning remain exploratory, not asserted from any nonzero
spike count. Reports and selected profile are recorded after the experiment.

## Results

The smallest qualifying tested candidate was 0.95, recorded in
`configs/resting-v2-motion-provisional.json` with the signed-current encoding.
The grid report is `2026-09-18-motion-resting-calibration-v1.json`. At 0.85 both
populations remain silent; at 0.90 T5 responds to OFF but not the ON stimulus
period. The preregistered both-polarities per-neuron modulation screen selects 0.95.
This is not an estimate of a biological or mathematical minimum current.

The independent 11-condition validation (`2026-09-18-motion-v2-probe-suite.json`)
uses two edge directions, two speeds, both polarities, local flashes and a matched
no-event control. For a fast edge, T4 counts are 356 ON / 538 OFF / 335 no-event;
T5 counts are 21 ON / 1,831 OFF / 21 no-event during the 400-tick stimulus.
T5's equal ON and control totals conceal redistribution (20 neurons change counts
in calibration); they are not increased population firing. T4's stronger OFF
response does not establish physiological ON preference. Peak activity is unchanged
at 1,785 spikes per tick across the 47,413-neuron roster, below the fixed ceiling.
Baseline rates over the latter warm-up half are 0.469 Hz T4 and 0.0332 Hz T5.

Silencing all non-T4/T5 neurons with measured inputs to these populations eliminates
all T4/T5 spikes in baseline, stimulus and recovery, across all 11 conditions
(`2026-09-18-motion-v2-afferents-silenced.json`). Silenced body IDs are recorded.
The motion neurons themselves remain unsilenced, with identical intrinsic currents.
This demonstrates dependence on circuit input rather than a forced firing floor.
A small LPi response also emerges; L4 and LC10 remain silent in these probes.

The zero-response failure is corrected within the existing point-neuron model.
Useful learned prediction, direction tuning and end-to-end behavioral performance
remain separate empirical questions. No additional neuron classes are adjusted
merely to inflate activity counts.
