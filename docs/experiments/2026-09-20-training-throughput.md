# Active training throughput, 2026-09-20

The simulator advances synthetic time without waiting for real-time camera events.
One environment step is a Pong frame (1/120 second), with eight neural ticks.
10,000 frames represent 83.33 simulated seconds and 80,000 neural ticks, not
10,000 individual sensory or neuronal events. The final 8,000-frame continuation
recorded 4,291,820 neuronal spikes over 47,413 neurons and 1,377,103 measured edges.

The continuation completed at step 10,000, checkpoint SHA-256
`6358107d489ee859f1644105eb274c29b226e6cdc71ae7e58d65a8bf6c925e58`.
Its 53,076.9-second wall duration includes three long progress gaps totaling about
49,029 seconds. Host suspension is a possible explanation but is not established.
The wall-total rate must not be treated as active compute throughput.

## Profile and first optimization

`python scripts/profile_training.py checkpoints/event-v1-combined-rate-10000.pt --output runs/training-profile-before.json`
loads read-only, warms five training frames, times twenty without checkpoint/UI
cost, then separately profiles three frames. CPU PyTorch 2.5.0, one thread.
The timed run took 13.566 seconds (13.25 process CPU seconds): learning observe
12.469 s (91.9%), network step 0.551 s (4.1%, including 0.141 s arrival delivery),
weight synchronization 0.087 s (0.6%), other orchestration 0.457 s (3.4%).
It ended with 207,923 active eligibility keys. Tensor indexing, searchsorted and
sorting dominate the separate operator trace. This is primarily sparse learning
bookkeeping cost, not insufficient speed computing membrane spikes or waiting
for the external world. This short sample is not a universal hardware limit.

Each neural tick previously merged keys with unique and then performed three
binary-search passes to recover positions. The optimized merge obtains the same
positions from unique's inverse map, removing the redundant searches. Sorted
keys, duplicate arrival counts, trace arithmetic, pruning, update ordering and
checkpoint format are preserved. No topology, signs, neuron model or learning
rule changed. No denormal flushing or approximation was enabled.

`python scripts/benchmark_sparse_merge.py checkpoints/event-v1-combined-rate-10000.pt --output runs/sparse-merge-benchmark.json`
compares deep copies of the same warmed state, twenty frames each, in reference /
optimized / optimized / reference order. Reference took 8.399 and 8.412 s;
optimized took 6.648 and 6.611 s: about 21.1% less time, or 26.8% more throughput
(2.38 -> 3.02 frames/s). Every recorded neural-tick spike and the final tensor
states matched exactly across all four runs. Source checkpoint checksum unchanged.
This balanced comparison, not the earlier cold/load-sensitive profile timing,
is the evidence for the speedup. Short-run equality is not a CUDA acceptance claim.

At this measured optimized rate, 10,000 frames would take about 55 minutes before
loading/checkpoint costs, still about 40 times slower than real time. Finishing
10,000 in five minutes would require 33.3 frames/s, about eleven times this rate.
The current CPU implementation therefore does not meet real-time online throughput.
A faster execution engine must preserve the same local rule and be profiled and
validated before claiming it solves that gap. Compute efficiency and experience
required to learn are separate questions; faster execution alone proves nothing
about learned anticipation.

Verification: 104 tests passed, four CUDA-only tests skipped. Read-only code review
found no actionable issues. Raw measurements accompany this report.

## Completed 10,000-frame learning evaluation

The longer run used unchanged combined-rate settings and training seed 1. Frozen
500-frame development evaluation on seeds 1101/1102 gives pooled event MSE
0.00095385753 (versus 0.00098373752 at 2,000 frames and 0.00116170038 initially).
Zero remains better at 0.00093836384; persistence is 0.00195293239. The trained
model is about 1.65% worse than zero. Event-conditioned MSE is 1.0000173 versus
zero's 1. The single shuffled control gives MSE 0.00095404976 and event-conditioned
1.0002291, a small descriptive alignment effect, not useful-anticipation evidence.

All four exposure groups still lose to zero in total MSE. The group labels are
fixed from the FIRST 2,000 training frames to match earlier reports; 'unseen'
does not mean unseen throughout the later 8,000-frame continuation. T4/T5 emitted
7,178/1,930 spikes over the paired evaluations; the whole network emitted 453,413
spikes, approximately 1.148 Hz per neuron. This checkpoint is neither globally
silent nor evidence of successful M1A learning. Reserved final seeds remain unused.

The additional experience mainly reduced false predictions during quiet periods.
It did not establish effective event anticipation, so do not advance M1B on the
claim that M1A passed, and do not repeat another long unchanged run as the default
next action. See `2026-09-20-exposure-generalization-10000.json` and the matching
integrity report for the complete measurements and graph/weight invariants.

Next work: profile a compiled sparse-eligibility implementation against this
reference (including merge, gather and update costs), retaining exact local rules
and a CPU comparison test; independently investigate why informative local signals
remain too weak or mistimed. Any new scientific model/target revision requires a
concrete evidence-backed proposal. Faster computation is an engineering change,
not permission to alter learning, topology, signs, or sensory inputs.
