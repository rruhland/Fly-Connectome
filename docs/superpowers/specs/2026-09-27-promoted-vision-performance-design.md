# Promoted M1A.5 vision performance design

## Intent and scope

Speed up the promoted `load_default()` visual service with online learning and its
`VisualStateEncoder` output. Preserve the current sensor contract, observations,
identity tracking, all forecast modes and intervals, online credit, learned bank
contents, checkpoint format, and deterministic replay. The older measured-graph
Pong trainer is outside this work, per the user's scope decision.

Measure full `model.step(..., learn=True)` plus `encoder.encode(...)` on the
existing 64x64, single-thread, two- and eight-entity benchmark. It excludes
camera acquisition and Pong physics, which are not connected to this service.
Targets are 50 samples/s (20 ms) and, if attainable without changing results,
120 samples/s (8.33 ms). Report achieved rates rather than assuming either.

## Evidence and approach

The clean worktree passes 721 tests with four CUDA skips. The same host's
baseline benchmark gives median 21.62 ms for two entities and 85.64 ms for
eight. A 100-sample eight-entity function profile spends 5.08 of 9.31 seconds
in scalar marginal-interval searches; the next large areas are state assembly,
interface encoding, repeated mixture retrieval, and support-field generation.

1. Keep the 32 bisection iterations, Gaussian CDF, rank calibration and mixture
   semantics. Batch independent interval searches by component count with
   `torch.vmap`, so each search executes the same operations and reduction order
   for every forecast. A representative 24-mixture probe was bitwise equal to
   the scalar path and ran about 14 times faster in isolation.
2. Reuse the exact mixture already produced for the point forecast and pending
   credit when exporting modes and intervals. Compute per-horizon calibration
   summaries once after settling pending endpoints; no learning occurs while
   forecasts are issued in that sample.
3. Reprofile the end-to-end path. Apply further changes only to measured
   remaining costs with deterministic output and learning parity. Native code,
   approximate retrieval, reduced mixture count, shorter quantile search, and
   weaker sensor validation all add fidelity or deployment risk and require
   separate evidence before use.

## Verification

Compare each forecast tensor, current entities and support field, sparse
transport output, learned banks, calibration ranks, and saved/reloaded model
across deterministic online streams including missing frames. Verify the batched
CDF on singleton and multi-component mixtures, calibration limits, and different
batch sizes. Run the full suite and remeasure both benchmark arms on the same
host and process settings. Do not claim Pong or camera frame rate from these
measurements.
