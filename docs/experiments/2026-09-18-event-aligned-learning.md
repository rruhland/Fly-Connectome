# Event-aligned learning experiments

The approved target and causal-trace revision is implemented in commit `1fb7003`.
It preserves measured topology, fixed signs, forward dynamics, local updates and
behavioral R-STDP. Full verification at that commit: 95 tests passed, four CUDA
tests skipped because this installation has no CUDA backend. Independent code
review found no actionable defect. Synthetic delayed-association learning succeeds
for both signs; this establishes software behavior only.

## Feedback operating point

`configs/feedback-calibration-v1.json` fixes the candidate grid and selection rule.
The smallest qualifying C2/C3 current was 1.0, with L4/Lawf1 fixed at 0.95 and
all other class parameters unchanged. Dm12 already responded and was unchanged.
Selection used frozen stimulus responsiveness and numerical stability, not Pong.
These fixed currents are model approximations, not measured physiological values.

Independent validation used eleven conditions: ON/OFF edges in two directions
at two speeds, local ON/OFF flashes, and no-event control. Each had 500 warm-up,
400 stimulus and 200 recovery ticks. Peak activity was 1,785 of 47,413 neurons
in a tick (3.77%), within the preregistered 20% bound; voltages remained finite.

| Population | Intact stimulus spikes, summed across conditions | Intact recovery spikes | External afferents silenced: stimulus / recovery |
| --- | ---: | ---: | ---: |
| C2 | 990 | 416 | 0 / 0 |
| C3 | 1,834 | 78 | 0 / 0 |
| L4 | 970 | 792 | 0 / 0 |
| Lawf1 | 17 | 7 | 0 / 0 |

The afferent intervention silenced 6,276 measured presynaptic neurons outside
these four target classes; it did not silence the targets themselves or modify
weights. Their activity vanished, demonstrating input dependence in this model.
This broad intervention also affects other pathways and does not identify a
unique causal input class. A separate intervention silenced C2/C3 themselves
(1,766 neurons), changing lamina counts and downstream responses. Intact T4/T5
remained active (3,800 / 4,164 stimulus spikes respectively), but activity alone
does not establish correct physiological tuning or useful prediction.

Reports: `2026-09-18-feedback-v3-probe-suite.json`,
`2026-09-18-feedback-v3-silenced.json`, and
`2026-09-18-feedback-v3-afferents-silenced.json`. Each records source checkpoint
and graph hashes, all conditions, class parameters, silenced body IDs and traces.

## Development experiment protocol

`configs/event-learning-ablation-v1.json` defines six conditions on the same
initial graph and feedback-v3 dynamics, trained on seed 1 for 500 steps. Frozen
development evaluation uses seeds 1101/1102 for 200 steps each. Conditions separate
the target, trace timing and learning-rate effects. Final seeds 1201-1204 are
reserved and remain untouched until a candidate passes development checks.

The common initial event MSE is 0.001282239; zero prediction is 0.001045013 and
persistence is 0.002199290. All six comparisons completed:

| Condition | Visual learning rate | Event MSE | Event-conditioned MSE |
| --- | ---: | ---: | ---: |
| Legacy target and trace | 0.0001 | 0.001256556 | 1.006285 |
| Input-increment target only | 0.0001 | 0.001220246 | 1.000555 |
| Causal trace only | 0.0001 | 0.001281045 | 1.000542 |
| Combined target and trace | 0.0001 | 0.001282381 | 1.000512 |
| Causal trace, higher rate | 0.01 | 0.001198415 | 1.003415 |
| Combined, higher rate | 0.01 | 0.001139934 | 0.999859 |

Values pool L1-L3 by sample count. No condition qualifies: zero prediction wins
overall, and the tiny event-conditioned improvement in the combined higher-rate
condition does not establish useful anticipation. Each complete initial/trained
report is saved as `2026-09-18-event-v1-CONDITION-audit.json`. The first three
processes (target-only, trace-rate, combined-rate) started before the shuffled
diagnostic was added, so their original grid reports do not include that control.
No missing control is treated as a pass.

A separate 500-step development audit of the combined higher-rate model gives
pooled event MSE 0.001028283, with time-shuffled predictions 0.001028285.
Event-conditioned MSE is 1.000559 (shuffled 1.000556). The source checkpoint remains
the 500-training-step artifact. More evaluation samples remove even the weak
event-conditioned advantage seen in the 200-step screen. L1 now has an active negative
predictive source at 757/892 neurons, compared with 2/892 before feedback calibration.
This counts connected sources that spiked, not surviving weight or delivered current;
restored source activity has not demonstrated useful timing. Shifted-time diagnostic
event-conditioned scores remain approximately 1 across offsets -8 through +8
frames; these results do not establish a consistent late response either.

A separately recorded continuation (`configs/event-learning-continuation-v1.json`)
tests undertraining by continuing the combined higher-rate checkpoint to 2,000
steps with parameters fixed. It preserves the 500-step checkpoint and saves every
100 steps. This is a development experiment, not a final acceptance run.

Offline lag diagnostics distinguish late responses from forecasts: positive lag
compares a target with a prediction recorded later. Future offsets never enter
training or acceptance scores. A temporal permutation control tests whether
alignment carries information beyond the prediction distribution. These diagnostic
additions do not change any model checkpoint.

An independent read-only review recomputed the pooled metrics and intervention
totals and confirmed the lag indexing. One shuffled sequence is a descriptive
control, not a statistical equivalence test. Neither these short runs nor the
2,000-step continuation can establish a general representation limit; undertraining
and inadequate dynamics remain distinct hypotheses.
