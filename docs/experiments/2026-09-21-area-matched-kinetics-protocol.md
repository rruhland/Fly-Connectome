# Area-matched excitatory kinetics control

User requests 20 ms excitatory decay with impulse amplitude reduced to match
the original 5 ms synapse's total area, to distinguish temporal memory from
increased excitatory drive. Preserve the working OFF pathway while diagnosing
quiet-frame selectivity afterward.

For discrete decay `r = exp(-dt/tau)`, an impulse of amplitude A has total area
`dt*A/(1-r)`. Set the excitatory amplitude factor to
`(1-exp(-dt/.020))/(1-exp(-dt/.005))`, approximately 0.270 at dt=1/960.
This is the exact infinite discrete-area match, not the continuous-time 0.25
approximation. Apply the factor to each excitatory predictive arrival and its
local eligibility increment. Keep 20 ms eligibility/current decay matched.
Inhibitory predictive impulses and 5 ms decay remain unchanged.

The area match is per unit magnitude. Magnitudes are still trainable, so a
trained network can compensate by increasing excitatory magnitudes. Record this
explicitly; survival after retraining alone is not proof of a pure shape effect.

Two fixed comparisons, no parameter sweep:

1. Train `area-matched-excitation-v1` from the identical original magnitudes,
   using the same 200 trajectories, seed, frame-horizon rule, learning rate and
   all eight neural ticks. Evaluate the identical 50 frozen trajectories and
   paired frozen precursor probes with unchanged ON/OFF/quiet criteria.
2. Freeze the previously learned unscaled-20-ms magnitudes and evaluate them
   under original class-dependent and area-matched 20 ms excitation.
   The reported L3 target originally has 5 ms decay; other cell classes can differ.
   Include a uniform-fast frozen control (all predictive decays 5 ms) to remove
   that class-dependent difference in the equal-area comparison. Compare to the saved
   unscaled-20-ms evaluation. No retraining or rescaling of these magnitudes.
   This isolates kinetics/amplitude from learned-weight compensation, although
   the coupled neural dynamics can still respond to the changed currents.

Both modes preserve measured topology, signs, delays, other dynamics, behavioral
learning, bounds and no-backprop. New waveform factor is fixed and recorded in
the manifest. Production defaults remain unchanged; this is a reference-runner
experiment, not a native/checkpoint extension.

Tests must integrate a single impulse to verify area, verify physical current
and signed eligibility at each tick, and retain the previous current/eligibility
tests. Frozen reconstruction includes warmup and the new impulse amplitude.
Stop if reconstruction, finite-state, weight bounds or input identity checks fail.

```
.venv/Scripts/python scripts/temporal_visual.py --visual-schedule frame-horizon-v1 --predictive-kinetics area-matched-excitation-v1 --output runs/temporal-area-matched-excitation-v1
.venv/Scripts/python scripts/area_matched_frozen_control.py
```

Then characterize which quiet frames produce false alarms using frozen saved
predictions, without weakening OFF, changing gates, or starting an unbounded
architecture search. A specific temporal-selectivity change should follow the
evidence rather than be bundled into this area control.
