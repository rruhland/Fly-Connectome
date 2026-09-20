# Local predictive signals and training exposure

The 2,000-step result did not pass zero-event prediction. Before changing the
model again, two frozen diagnostics separate available local signals from training
exposure. Neither fits a predictor or updates weights.

`scripts/audit_local_signals.py` reconstructs signed per-edge causal traces for
measured predictive edges onto L1-L3. It checks their weighted sum against actual
predictive current at every tick, including warm-up. Statistics compare the old
trace with the next local event. The first game frame is excluded because its
metric forecast is initialized to zero. Development seeds 1101/1102, 500 frames:

| Checkpoint | Population | Active traces | Positive covariance | Positive mean local update |
| --- | --- | ---: | ---: | ---: |
| Initial | L1 | 3,123 | 121 | 60 |
| Initial | L2 | 4,742 | 323 | 41 |
| Initial | L3 | 755 | 32 | 2 |
| 2,000 steps | L1 | 2,704 | 89 | 25 |
| 2,000 steps | L2 | 4,497 | 404 | 86 |
| 2,000 steps | L3 | 375 | 13 | 0 |

Maximum current reconstruction errors were 5.96e-8 and 4.47e-8. Most measured
local updates favor reducing predictions. A few positive correlations exist,
but top correlations selected among many edges are descriptive and may involve
very few events; they are not held-out learning evidence or significance tests.
These are physical-current traces including warm-up arrivals. Actual training
initializes plasticity traces empty after warm-up, so the earliest potential
update estimates retain a decaying warm-up difference; they are not an exact
replay of training updates. Independent review confirmed timing and moment math
with this qualification.

`scripts/audit_pong_exposure.py` replays the exact scripted M1A camera/Pong stream
without neural computation. A fixture compares its events and final environment
state with actual Trainer steps. For training seed 1:
Counts mean nonzero projected signed injections, including initial-frame events;
opposing pixel polarities that cancel in a retinotopic bin count as zero.

| Simulated steps | Seconds | L1 never stimulated | L1 with fewer than 10 events | Median events per L1 neuron |
| --- | ---: | ---: | ---: | ---: |
| 2,000 | 16.67 | 563 / 892 | 865 / 892 | 0 |
| 10,000 | 83.33 | 228 / 892 | 725 / 892 | 4 |

L2/L3 have nearly identical coverage. Blank receptive fields can legitimately
remain unvisited, so this alone does not prove undertraining causes the failure.
It does show that the pilot provides little direct experience to most local
synapses and cannot establish an architectural learning limit.

The refined mapped-field report (`2026-09-19-pong-exposure-seed1-mapped.json`)
separates columns receiving no render pixels from unvisited mapped columns.
Of 761 mapped L1 neurons, 432 had no events after 2,000 steps and 734 had fewer
than ten; at 10,000 steps these counts fall to 97 and 594. Thus the sparse-exposure
finding remains after excluding 131 unmapped L1 neurons.

Frozen errors grouped by the first 2,000 steps of training exposure show:

| L1-L3 exposure group | Neurons | Initial event MSE | Trained event MSE | Zero MSE |
| --- | ---: | ---: | ---: | ---: |
| No mapped render pixels | 393 | 0.000156564 | 0.000034235 | 0 |
| Mapped, no training events | 1,296 | 0.000639495 | 0.000436299 | 0.000388889 |
| 1-9 training events | 907 | 0.001050409 | 0.000874170 | 0.000824697 |
| At least 10 training events | 81 | 0.015639946 | 0.015576485 | 0.015555556 |

The more exposed group has slightly improved event-conditioned error (1.000625
to 0.999969), but still loses to zero overall. No group supplies an acceptance
claim. Absolute errors differ because event frequencies differ between groups;
these are descriptive comparisons, not causal effects of exposure. Grouped
sample-weighted errors reproduce the full sensory score. Reproduce using
`scripts/audit_exposure_generalization.py` with `--exposure-steps 2000` and the
initial or 2,000-step checkpoint; saved reports record exact checkpoint identity.

The next fixed-parameter experiment is recorded in
`configs/event-learning-exposure-continuation-v1.json`: resume the exact 2,000-step
checkpoint for 8,000 additional steps, save every 100, then evaluate 500 frames on
development seeds 1101/1102. Topology, signs, local rule, physics and acceptance
criteria remain unchanged. Final seeds remain unused. The longer exposure replay
does not imply that 10,000 steps will suffice or that learning will succeed.
