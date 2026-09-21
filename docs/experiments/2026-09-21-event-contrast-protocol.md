# Local memory and event forecast separation

User authorized continuing the rise/decay findings' proposal to separate retained
local memory from a brief event forecast. Implement one bounded candidate:
`forecast[t] = clip((I_predictive[t]-I_predictive[t-8])/threshold, -1, 1)`.
Use the successful area-matched 20 ms excitatory / 5 ms inhibitory physical
dynamics, not the rejected rise/decay model. The eight-tick interval is one camera
frame; no fitted time constant, threshold or gain is added.

Physical memory currents still drive the original membrane equations. A private
eight-tick current history at each neuron produces a separate forecast signal;
it is not fed back into the membrane. The history advances during warmup too.
The forecast represents local change, so its sign can oppose the sustained
current's sign without changing any physical synapse sign. This is an intentional
forecast-encoding revision, not a reinterpretation of measured transmitter signs.

Keep frame-horizon supervision: issue at ticks 0,8,... and confirm against local
observation at issue+8, including quiet camera frames. The eligibility captured
at issue is `e[t]-e[t-8]`, using the same physical-memory traces. Apply the same
local equation `eta*(observation_at_confirmation-forecast_at_issue)*eligibility`.
Only the captured forecast eligibility is differenced; live memory traces decay
normally on every neural tick. Preserve keys from either frame when differencing.
Retain the original convention of cold learning eligibility after neural warmup.

No new connections, backpropagation, learned readout weights, external error
network, input cues or changes to anatomy/signs. Keep initial magnitudes, membrane,
adaptation, sensory input, eight ticks/frame, learning rate, synchronization,
bounds, homeostasis and behavioral R-STDP unchanged. Extra local histories are
experimental reference-runner state, not supported by native engines/checkpoints.

This differencing makes constant memory produce zero forecast, but does not
guarantee all transients become sharply timed or have the correct polarity.
That is the hypothesis to test, not a success claim. Do not retune after seeing
the result within this bounded comparison.

Tests: exact physical-state equality to the memory-only model at frozen weights;
exact eight-tick current difference; matching retained eligibility difference;
confirmation uses issue-time state, with no intermediate visual updates. Frozen
preflight reconstructs the difference from independent warmup-inclusive physical
traces at unchanged 1e-6 tolerance. At fixed weights, verify recorded neural
spikes equal the previous area-matched frozen control.

Same induced crop, seeds, 200 training trials and 50 frozen evaluation trials.
Compare the unchanged fixed ON/OFF/MSE/quiet gates and the same quiet-phase groups.
If this candidate fails, report the failure without threshold relaxation or
another architecture sweep. M1A must still pass before M1B.

```
.venv/Scripts/python scripts/temporal_visual.py --visual-schedule frame-horizon-v1 --predictive-kinetics event-contrast-v1 --output runs/temporal-event-contrast-v1
```
