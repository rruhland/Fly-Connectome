# Pong-to-promoted-vision performance design

## Intent and scope

Measure and improve the actual CPU path from Pong physics through its 64x64
tensor render, camera events, promoted `load_default()` vision, and
`VisualStateEncoder`. Cover online learning and frozen evaluation. The user chose
a fixed, reproducible motor drive; this work does not add a reward controller.
Preserve Pong rules, rendered pixels, event polarity, vision outputs, online
learning, and checkpoint content.

## Cadence and boundary

Keep Pong's existing 120 Hz physics and the declared 50 Hz visual sample period.
At visual sample `n`, render the current game once, form dense OFF/ON events from
the previous rendered frame (initially black), call vision and the encoder, then
advance Pong by `floor((n+1)*12/5)-floor(n*12/5)` physics ticks with zero drive.
This exact 2/2/3/2/3 schedule averages 50 camera samples/s without changing
the model's physical horizon calibration. A Pong score resets the game body as
it already does; camera reference and vision histories continue across a serve.
The loop consumes the complete returned visual state and sparse transport, while
Pong's privileged state supplies only physics and scoring diagnostics.

## Measurement and preservation

A runnable script will time 40 warmup plus 200 visual samples in each mode on
one CPU thread. It will report total latency, render/event/vision/encoder/physics
breakdown, p50/p95, simulated physics ticks, game outcomes, and bank size. The
same script runs against the original source path and this worktree. A digest
check compares every rendered frame, event tensor, complete vision output,
encoded output, Pong state/outcomes, and the learned checkpoint across revisions.
Focused tests check the event conversion against `EventCamera`, exact physics
cadence, and render equivalence for any rendering optimization. Full pytest must
pass after code changes.

## Decision rule

Profile the full loop first. Keep only optimizations that improve measured
latency and retain exact outputs. In particular, do not lower render resolution,
change camera cadence, shorten forecast search, prune memories, or alter physics
to make an FPS target look better. Distinguish visual-sample throughput from
physics-tick and actual display throughput. This script includes tensor
rasterization, not browser presentation or monitor scanout.
