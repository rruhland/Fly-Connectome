# Continuous frozen-network timing challenge

Keep the source mixed-training neural weights, area-matched synaptic kinetics,
saved split event-balanced coefficients, and causal reference/window unchanged.
No optimizer, new learning rule, neural reset at a tempo boundary, or parameter
search. The selector remains an offline diagnostic.

Use the preceding sensory stress schedule: initial wait of 20 frames, followed
by 24 alternating events per interval block 2,3,6,4 frames, then 30 quiet frames.
Render the dot at the other column initially, toggle its position at each event,
and hold it after the last event. For the matched omission condition, omit
scheduled events 60 and 61 (zero based) by holding the position. The first event
is at frame22. Run each condition from the same frozen weights and ordinary
warmup. No resets inside a condition. Eight neural ticks per camera frame.

Capture all 12 issue-time incoming predictive eligibilities and local sensory
state after the issue tick, as in the original audit. Apply the exact saved
split design and coefficients, clip amplitude, then apply the unchanged causal
window. Align each issued forecast to the observation one frame later.

Before the challenge, reproduce the existing collector's selected features,
targets, forecasts and complete spikes on two trials (blank lengths12,16 at
dwell3). Check source identity, incoming edge order, fixed signs, unchanged
weights, causal omission-prefix identity and sensory reconstruction.

Report ungated and gated ON/OFF amplitudes, quiet alarms and MSE for the whole
stream, and each tempo block separately. Partition each block into its first
four scheduled events (adaptation) and the remaining events (steady); report
omission/recovery ordinals12-15 of block2 separately, excluding them from the
steady partition. Retain event-by-event records and initial/trailing quiet
metrics. Metadata partitions are reporting-only, never selector inputs.

This assesses the transfer of an offline readout and timing mechanism on new
neural histories; it cannot establish local learning or an M1A pass.
