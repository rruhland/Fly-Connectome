# Context efficacy does not yet transfer to open-loop Pong M1A

The approved bounded bridge is implemented as an opt-in, resumable full-graph
experiment. It retains all 47,413 measured visual neurons and 1,377,103 edges.
Only the 17,557 predictive edges entering 2,660 directly observed L1/L2/L3
cells receive two local magnitude components. All other weights are frozen;
topology, signs, delays, no-backprop, event-camera-only neural input, and the
scripted open-loop M1A paddle are preserved. Production training defaults and
checkpoint schema are unchanged.

The neural implementation exactly matched the area-matched reference over 50
Pong frames/400 ticks, including 53,653 selected-edge arrivals. A separate
full-graph check matched selected causal eligibility, forecasts and state to
the generic reference. Keeping eligibility only for actually plastic edges
reduced peak keys from 302,118 to 7,642, with exact selected-edge parity. A
frame-9 checkpoint resumed to the same frame-19 neural, plasticity, camera,
Pong, forecast, event and score state as an uninterrupted run. One original
checkpoint edge exceeded the existing maximum magnitude of 10; both
initial/evaluation arms apply that bound once before warmup. No model parameter
was chosen from Pong outcomes.

## Capped development result

With the controlled motif's preselected rate 1.0, the learner trained for 500
continuous camera frames (4,000 neural ticks) on development seed 1101.
Checkpoints were saved every 100 frames. Frozen initial and trained networks
then saw bit-for-bit identical camera target sequences over 500 frames on
development seed 1102. Final M1A seeds 1201-1204 were untouched.

| Seed 1102 metric | Initial | Trained |
|---|---:|---:|
| All-target next-frame MSE | .00092073 | .00092154 |
| ON-event MSE (612 events) | .999997 | 1.000002 |
| OFF-event MSE (610 events) | .999979 | .996819 |
| ON anticipation | .0000016 | **-.0000010** |
| OFF anticipation | .0000111 | .002919 |
| Quiet false alarms | .000075% | .000377% |

Whole-field MSE is dominated by 1,326,118 quiet target samples; beating
persistence there is not evidence of useful event prediction. The trained
network changes 260 context magnitudes, but ON does not improve and OFF remains
far below the controlled-task .1 anticipation threshold. Exact per-frame
update reconstruction error is zero at float32 precision, non-target weights
remain unchanged, all state is finite, mean cell firing is 1.14 Hz, and the
maximum cell rate is 30.24 Hz. The 500-frame training pass took 40.37 seconds
(12.38 camera frames/s); frozen evaluations took 42.62 and 40.84 seconds.
This is a compute limit, not waiting for real-time events: the 50-frame
benchmark spent only .037 seconds rendering/projecting camera input versus
1.896 seconds in neural execution and 1.374 seconds in plasticity.

## Narrow causal diagnosis

A frozen replay of the exact seed-1102 camera targets measured the local
information available at the prior forecast issue:

| Event due | Count | Gate open | Raw eligible precursor | Both gate and precursor |
|---|---:|---:|---:|---:|
| ON | 612 | 27 | 361 initial / 355 trained | 12 |
| OFF | 610 | 47 | 365 initial / 361 trained | 24 initial / 23 trained |

Thus the rule can issue amplitude credit on only about 2% of ON and 4% of OFF
events in these Pong frames. The timing window learned on repeated two-position
motion rarely opens at an irregular Pong event. Yet merely removing the window
at evaluation does not reveal a strong hidden predictor: with trained weights,
only 2 ON and 6 OFF raw event forecasts reach magnitude .1. Ungated raw
anticipation is -.0030 ON and +.0086 OFF; 3,242 of 1,326,118 quiet samples
would become raw false alarms at the .1 threshold. The current failure combines
scarce gate-aligned credit with weak or conflicting physical prediction; it is
not an eight-tick horizon mismatch or a lack of weight changes.

The approved bridge therefore stops after its capped pilot and diagnosis.
Simply repeating the same update for more frames is not justified by this
coverage. M1A is not passed and M1B remains blocked. The next rule experiment
must be reviewed separately: the controlled-task gate supplied useful quiet
selectivity, but its local timing reference does not express irregular Pong
events. A bounded gate-credit ablation, scored on ON/OFF and event-adjacent
quiet frames rather than quiet-dominated whole-field error, is a useful first
test before adding another temporal state or changing indirect visual cells.

Evidence: [full neural parity](2026-09-22-full-context-neural-parity.json),
[selected eligibility parity](2026-09-22-full-context-sparse-rule-parity.json),
[throughput](2026-09-22-full-context-benchmark-50.json),
[exact resume](2026-09-22-full-context-resume-check.json),
[Pong pilot](2026-09-22-full-context-m1a-pilot-results.json), and
[initial](2026-09-22-full-context-gate-audit-initial.json) / [trained](2026-09-22-full-context-gate-audit-trained.json)
issue-time audits. Full checkpoints and traces remain under
`runs/full-context-m1a-v1/`.
