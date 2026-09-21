# Controlled visual learning: fixed first experiment

Approved in conversation after the failed full-Pong history gate. This protocol
is recorded before running the real-data training experiment. No model changes.

## Circuit and stimulus

Use `event-v1-combined-rate-initial.pt`, not the trained Pong checkpoint.
Take sensory neurons mapped to row 16, pixels x=28 through 36 in the original
64x32 retina. Include all intermediate neurons on measured two-edge directed
paths between those sensory neurons. Keep the induced measured graph, original
weights, delays, pathway assignments, signs and class parameters. Retain the
full original retinal coordinate system; never refit/zoom the crop. Record cut
boundary connections. A missing external drive remains a limitation of cropping.

Render one bright pixel moving right by one pixel per camera frame across that
row. Each traversal has a uniformly sampled 12–36-frame blank interval and an
explicit blank frame after the dot leaves. Camera references and neural state
are continuous across traversals. The stimulus delivers real ON/OFF events;
no position, clock, future event or teacher trace enters the network.

Central-column L1/L2/L3 neurons are the fixed targets. Primary forecasts are
issued immediately after the first neural tick of a camera frame and scored
against the next frame's first-tick sensory increment: eight neural ticks,
8.33 ms ahead. The current rule still learns its original one-tick objective.
One-tick scores are secondary diagnostics, not alternative acceptance gates.

## Budget and controls

One run: 200 training traversals (seed 9001), 50 separate evaluation traversals
(seed 9002), at most 600 seconds for training and both evaluations. No sweeps or
automatic retries with other parameters. All existing learning/homeostasis
settings remain unchanged. Use the CPU reference with one Torch thread.

Evaluate frozen initial and trained weights from freshly initialized neural
state with identical normal warmup and identical evaluation images; no plasticity
during evaluation. Include zero, last camera-event persistence, and a causal
stimulus-only predictor using the observed dot's current pixel and fixed speed.
The latter is an offline task sanity check, never a neural input or trained head.

## Fixed success criteria

At the eight-tick lead, require pooled target MSE at least 20% below frozen,
zero and persistence, AND ON and OFF event-conditional MSE each at least 10%
below their zero and frozen values. Require mean correctly signed anticipation
at least 0.1 for each polarity and quiet-frame false-alarm fraction (|p|>=0.1)
no greater than 5%. All activity and weights must remain finite and bounded.
These are engineering capability thresholds, not statistical significance or
M1A acceptance. Report each target separately and the causal sanity control.

On failure, inspect this run's spikes, incoming traces, and actual local update
contributions at ON, OFF and quiet ticks. Do not initiate another search. Any
proposed model change must explain an observed failure here and come back for
approval. Passing this task would permit returning to M1A, not advancing to M1B.

## Verification

Test measured cropping, retained delays/weights/pathways, visible ON/OFF input,
positive forecast lead, exclusion of incomplete tails, future-input isolation,
real-spike learning, frozen weights and scoring. Preserve the source checkpoint.
Save machine-readable results, topology provenance and complete small-run traces.
