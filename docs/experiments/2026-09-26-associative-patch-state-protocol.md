# Generic local patch associations instead of an above/below cue rule

Opt-in architectural experiment. No production change. Sensor reconstruction and
the benchmark-specific context comparator are separate blockers; this experiment
addresses the latter without changing the frozen noisy-state gate.

Replace two hand-selected cue classes and their two effect magnitudes with an
online associative memory. A tracked region retains a coarse local contrast
patch and its observed displacement direction. At observed reacquisition, local
pre/post covariance learns which patch components distinguish the subsequent
displacement correction. A bounded experience memory retrieves corrections with
this learned metric. No object names, cue labels, background/occluder masks,
hidden positions, simulator direction/speed, or future frames enter the learner.
The feature receptive field and generic tracklets are engineered primitives;
the predictive sensory selectivity and associations are learned from experience.
No backpropagation.

Use one fixed 21x21 receptive field pooled to 7x7, a 256-example memory, and
four-neighbor retrieval. Train on the same observed 32 context histories.
Compare aligned, shuffled-outcome and frozen association; measure all previous
gates with fresh noise seed 60000. First use noiseless current visible frames as
an explicitly additional sensor/capacity control, then the original event-only
observer. Do not call a frame-based pass event-only acceptance.

Rotation is decisive: present rotated observed histories and outputs, train a
single memory on all four orientations, then hold out shape/position/speed as
before. This is not an oracle coordinate transform within the model: images and
events are physically rotated in the benchmark. The model receives arrays only.
Keep one configuration; reject rather than add cue-specific channels or tune
retrieval bandwidth to failures. A promising result requires >=14/16 context
top-32 per orientation and material improvement over both learning controls,
plus clean regression/reacquisition gates. Generic dynamics and real camera
transfer remain additional required tests before promotion.
