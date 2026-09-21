# Optional next execution redesign: persistent compiled neural frame

Status: bounded approved pass completed. Both overhead/memory experiments were
exact but timing was inconclusive; neither is adopted. Optimization is stopped
per the user's scope limit and M1A investigation has resumed. The full runner
below remains deferred. This is a larger software
architecture change beyond the completed bounded neuron and eligibility
experiments. It changes no biological model assumptions. The alternative is to
pause throughput work at the tested native implementation and resume M1A learning
investigation, as the user permits when larger compromises become the next path.

## Measured boundary

The exact native backend now runs about 49 frames/s on this host. A 10,000-frame
run projects to about 3.4 minutes excluding startup/save. Target remains 120fps.
Long wake gaps exist (median neuron mean 30.23 ticks), but the closed-form neuron
prototype changes spike timing and is rejected. Staged sparse SIMD is exact but
slower (32fps), also rejected. Fixed retinal index caching is exact and retained.

The 500-frame profile takes 11.48s: learning 6.08s, network 2.42s, synchronization
0.35s. Inside those phases, native sparse work is 2.83s and native neuron work
1.72s. Even eliminating all Python overhead in that sample would not by itself
reach the 4.17s target. A compiled runner is therefore an uncertain experiment,
not a promised solution or evidence of a hardware limit.

## Concrete scope

1. Keep Pong, camera projection, public frame boundaries and checkpoints in Python.
   Add an isolated B=1 CPU entry point for the existing eight neural ticks per
   frame. Preserve every arrival, output spike and learning update at its original
   integer tick; do not batch away online causal interactions.
2. Keep persistent scratch arrays and sorted sparse eligibility buffers inside
   that runner. Reuse capacity and fuse compatible state passes instead of
   allocating tensors and traversing Python/ctypes for each operation. Ownership
   is explicit: publish canonical tensors at frame boundaries and materialize
   before snapshots, checkpoint saves or backend switches.
3. Preserve the exact arithmetic order of neuron, visual and behavioral updates.
   In particular, reproduce PyTorch's existing rate `lerp_` behavior or retain
   that operation through an explicit fallback. Do not silently substitute a
   differently rounded formula. No altered dtype, fast-math, denormal flushing,
   epsilon pruning or closed-form neuron propagation.
4. Preserve metrics mode, motor-rate state, weight synchronization, homeostasis,
   reset/silencing, delays and save/resume behavior. Keep the existing backends
   as independent oracles. No scientific parameter tuning during benchmarks.

## Verification and stop conditions

Start with differential tiny-graph tests for all eight tick outputs and local
state, including nonzero rewards, clipping/pruning, delayed inhibition, autonomous
firing, sensory filters and interrupted save/resume. Then run balanced 1000-frame
whole-training trials and a longer development trajectory if exactness and speed
are promising. Verify every spike and complete canonical tensor state. Final
scientific evaluation seeds remain unused.

Reject or retain as experimental if exactness fails or complete runtime is not
improved. The 120fps criterion stays unchanged. No new biological assumptions
are authorized by this proposal. Any required change to equations or numerical
acceptance must be reported rather than hidden in the implementation.

## Reserved for later milestones

Reducing neural tick frequency, changing resting/autonomous activity, coarsening
plasticity timing, approximate state truncation or changing weight normalization
may alter model behavior. These are explicitly outside current implementation.
The user has left future reconsideration open; none is adopted or recommended
as scientifically validated here. No-backprop, anatomical topology and local
learning remain current nonnegotiable requirements.

## Bounded execution ruling

Ruling: implement only two small prerequisites now (persistent native scratch,
then native learning-arrival filtering), rather than a full eight-tick runner.
The user explicitly prioritized stopping after a couple of experiments and
returning to M1A soon. Cost if wrong: potential larger runtime gains remain
unmeasured; no biological or numerical compromise is introduced. These changes
leave public tensor ownership, per-tick PyTorch rate lerp and checkpoints intact.
Compare reference/scratch/filter/filter/scratch/reference over 1000 frames;
retain only useful exact wins, record the stop, and resume M1A investigation.
