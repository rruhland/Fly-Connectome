# Pong-to-promoted-vision performance implementation plan

**Goal:** Benchmark and optimize the complete fixed-drive Pong render-to-vision loop while retaining identical game and learning results.

**Architecture:** A reproducible benchmark script drives the existing Pong environment at 120 Hz and the promoted vision service at its declared 50 Hz. The same orchestration runs against original and optimized source for exact parity and paired latency measurements.

**Spec:** [Pong-to-promoted-vision performance design](../specs/2026-09-27-pong-vision-performance-design.md)

## Constraints

- Use one Pong seed, zero drive, 64x64 render, OFF/ON events, and the complete vision state plus sparse encoder.
- Keep 120 Hz physics and 50 Hz visual sampling; no privileged game values enter vision.
- Preserve score reset, rendered pixels, event reference, outputs, and learning/checkpoints.

## Task 1: Executable full-loop reference

- [x] Add a reproducible script for 40 warmup and 200 measured visual samples in online and frozen learned-checkpoint modes, with phase timings, 20 ms deadline counts, source provenance, and bank/outcome summaries.
- [x] Test dense event conversion against `EventCamera` and verify the 2/2/3/2/3 physics schedule.
- [x] Run the script against explicit original and optimized source paths; record paired baseline and current latency, including a 1,000-sample later online window.

## Task 2: Measured exact optimizations

- [x] Profile render/event/vision/encoder/physics portions of the full loop.
- [x] Probe the renderer grid cache against exact pixels and measured cost; no material safe candidate remained for a production edit.
- [x] Reject the grid cache's sub-0.2% full-loop saving; preserve the current renderer.

## Task 3: Final evidence

- [x] Commit a parity runner and compare full-loop online and learned-checkpoint evaluation trajectories plus saved checkpoints across original and optimized revisions.
- [x] Run the full suite and paired training/evaluation benchmarks with no competing test process.
- [x] Record achieved visual sample rates, game/render costs, long-memory limits, and unachieved targets; commit the changes.
