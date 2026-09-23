# Frozen motion-stage audit before changing M1A plasticity

The failed L1/L2/L3 next-event objective may be aimed at the wrong circuit
stage. This read-only diagnostic asks whether motion direction is expressed in
the current measured optic-lobe graph before further learning-rule design.
It is exploratory, not a new Milestone 1 acceptance gate.

Use the pinned T5 full-graph initial checkpoint, the existing area-matched
context neural dynamics, fixed event-camera projection, 8 neural ticks per
camera frame, and no plasticity. Warm the unchanged circuit for 500 ticks,
then restore that exact state before each probe. Do not feed direction labels,
coordinates, or simulator state into neurons. The source checkpoint is not
modified.

Present a binary radius-1.5 dot against dark (ON) or bright (OFF) background.
After two blank camera frames, move it one pixel per frame through the same
13 positions in either order, then return to blank for three frames. Test
horizontal traversals centered at x=18 and x=46, y=16. The opposing
directions and locations share the same occupancy and event counts. Run a
matched blank control from the same neural state. Report input event counts,
transit-period spike counts and membrane voltage, both globally and for the
retinotopic columns traversed by the dot. Inspect L1/L2/L3, named medulla
partners, T4a-d, T5a-d, and LPi without fitting a decoder. For each class
compare right-minus-left response separately at both locations and report
whether its sign recurs. Store per-neuron traces locally for later audit.

The useful branches are:

1. Reproducible T4/T5 subtype direction contrasts at both locations would
   support moving the predictive objective downstream, using local incoming
   feedforward activity as its observation. A proposal must specify causal
   target/eligibility alignment before any change.
2. Direction information in medulla but not T4/T5 spikes would motivate a
   focused neural operating-point and dendritic-integration audit, not a
   larger learning sweep.
3. No direction contrast anywhere would trigger a sensor/retinotopy and
   retained-anatomy audit before changing plasticity.

These are decision guides, not claims that one small probe proves general
motion understanding. In particular, an aggregate subtype count can cancel
local tuning, so retain per-neuron and local-column measurements. No trained
readout is installed in the model.

If the small dot produces almost no T4/T5 modulation, run one predeclared
follow-up with a 3-by-13-pixel vertical bar along the same path. This raises
spatial support without changing its speed, positions, neural model, or
analysis. It distinguishes an underdriven small-object probe from a missing
motion computation; do not tune a size or gain sweep from these outcomes.
