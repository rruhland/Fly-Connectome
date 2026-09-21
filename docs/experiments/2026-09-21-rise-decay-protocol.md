# Area-matched excitatory rise and decay

User approved the proposed sign-preserving rise-and-decay synaptic response to
test temporal localization while retaining approximately 20 ms predictive memory.
Use one fixed candidate, not a parameter sweep: 5 ms rise component, 20 ms decay.

For a spike arriving at tick zero, excitatory current per unit magnitude is
`K[k] = A*(r20**k-r5**k)` for k >= 0, with `rT=exp(-dt/T)` and
`A = (1/(1-r5))/(1/(1-r20)-1/(1-r5))`. The discrete area matches the original
5 ms exponential and the previous area-matched-20-ms control. K[0]=0; K[k]>=0;
the peak is about 9 ms after arrival. This is a candidate waveform, not a measured
physiological fit. It shifts the peak but retains the 20 ms tail, so sharper
forecasts are a hypothesis, not a mathematical guarantee.

Keep inhibition's original 5 ms waveform and unit impulse unchanged. Preserve
measured topology, fixed signs/delays, original initial magnitudes, neuronal
membrane/sensory/adaptation dynamics, eight neural ticks/frame, frame-horizon
supervision, local update equation, rate, bounds, synchronization and behavioral
R-STDP. No backpropagation, added error network, learned readout or connections.

Two nonnegative excitatory current components implement one sign-preserving
response. Their difference is excitatory, not an extra inhibitory pathway.
Each arrival stores its weight at arrival; updates do not reweight old currents.
Maintain matching private local slow/fast eligibility traces and retain their
difference at forecast issuance. Intermediate neural ticks update traces without
visual targets. Confirm +8 ticks later as before. Behavioral and inhibitory
eligibility remain unchanged. The reference-runner-only experimental state is
not supported by production checkpointing or native engines.

Tests first verify actual current/eligibility formulas, retained forecast trace,
nonnegative excitation, delayed peak and discrete area. Reject incompatible
one-tick supervision for this waveform. Frozen preflight reconstructs current
independently from delayed spike history (including warmup) and both kernel
components, keeping tolerance 1e-6.

Same 250-neuron/1664-edge crop, seeds, 200 training trials and 50 frozen test
trajectories. Compare to area-matched-excitation-v1 and unscaled20ms results.
Preserve all original ON/OFF, quiet-frame and relative-improvement gates.
After evaluation, categorize the same quiet frames by motion phase and elapsed
blank time to distinguish genuine localization from shifting or weakening
forecasts. Stop this bounded candidate after reporting its result.

```
.venv/Scripts/python scripts/temporal_visual.py --visual-schedule frame-horizon-v1 --predictive-kinetics rise-decay-excitation-v1 --output runs/temporal-rise-decay-excitation-v1
```
