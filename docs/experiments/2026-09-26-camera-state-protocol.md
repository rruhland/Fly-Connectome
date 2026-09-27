# Generic camera-state audit and declared candidate sensor

The generic patch-association model passes positional and appearance-coded
relations across four physical rotations with current visible images. Its
event-only familiar/heavy-noise gate remains unpassed. Keep both facts explicit.

Evaluate a **separate proposed production candidate** whose declared sensor is
events plus current visible camera images. This is an experimental architecture
revision, not promotion or a waiver of event-only criteria. Final review must
state this input requirement and its compute/bandwidth cost. Sparse/event-only
versions remain unaccepted. No renderer-only teaching signal is used: all image
observations are actual declared current sensor inputs.

One bounded audit covers two distinct patterns crossing horizontally, vertically
and diagonally on a changed 64x64 camera grid, with clean observations, .10
standard-deviation independent image noise, and a five-sample camera outage.
Noisy events retain the familiar corruption. Oracle identities/centers are used
only to score early-to-late continuity. Measure identity retention, final center
error, spurious moving hypotheses, and persistence during outages. Require at
least 90% identity continuity, <=2-pixel mean late center error and <=1 extra
moving hypothesis per scene in every condition. No threshold sweep.

A missing camera sample is represented by sensor availability, not a blank
observed image or an occlusion flag. A model cannot treat an unobserved region
as contradictory evidence. Tests enforce that availability changes only whether
new absence evidence can lower confidence, not predicted motion or learning
targets. This interface correction is separate from unknown event dropouts.

This audit does not establish calibration, arbitrary photographic transfer, or
forecast accuracy. Those remain gates before any production proposal.
