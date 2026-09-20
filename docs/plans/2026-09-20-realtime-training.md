# Near-real-time CPU training plan (2026-09-20)

Priority: pause further scientific milestone training until throughput approaches
120 Pong frames/s with B=1 online updates on the measured M1A graph. Preserve the
same time resolution, topology, fixed signs, event stream, neuron equations,
plasticity and synchronization. Weight normalization is NOT authorized; it is a
model change and does not itself remove work per edge.

1. Fuse the sparse local eligibility/error/update operation into a compiled CPU
   kernel. Keep tensor reference and checkpoint format. Verify mixed pathways,
   signed targets, both trace versions, pruning, empty arrivals and exact resumed
   trajectories before benchmarking. Batches/CUDA retain the tensor reference.
2. Profile remaining costs, optimize only measured execution hotspots, and run
   relevant regression tests after each chunk. Any numerical differences must be
   quantified; do not silently change numerical semantics for speed.
3. Provide an explicit usable backend/build path, checkpoint compatibility and
   sustained throughput evidence including online weight updates. Target 120fps;
   report real measured limits honestly. Ask before any model architecture change.

Existing branch codex/milestone-1 is the isolated feature branch; start c38c582.
Progress and decisions append below; production is valid after every chunk.

Chunk 1: compiled sparse observe (both eligibility versions, mixed pathways),
zero-effect homeostasis shortcut, and neuron update implemented. All114 tests
pass, fourCUDA skips. Native/ref twenty-frame spikes and final tensor hashes
match exactly on the 10,000-frame checkpoint. First sparse-only comparison
3->11fps; homeostasis profile ~18fps; later neural-inclusive comparison ~12fps
vs~1.8fps reference under a slower host interval. Do not compare separate wall
samples as controlled speedups. Raw balanced evidence retained. Model unchanged.
Next: diagnose subnormal cost, then remaining phases; no backend default change.

Chunk 2: bounded parallel sparse updates, compact private anatomy caches, native
arrival extraction, observation preparation and synchronization implemented. An
explicit events-only logging mode preserves all neural/weight updates, spike
counts and event scores; dense MSE logging pauses and evaluation retains full
metrics. UI labels that pause. ABI5 rejects stale libraries. Review found unsafe
observation pointers; regression tests now reject noncanonical dtype/shape/stride
before native writes. Native/reference trajectory hashes match over100 frames.

Latest200-frame profile:38.9997fps (5.128s), learning2.660s including sparse1.274s,
network1.026s including native neurons0.809s, sync0.487s. Target120fps NOT reached.
Four native workers beat1/2/6 in the measured sweep. No stable benefit from
-march=native or denormal flushing; neither adopted. An exact software subnormal
helper also showed no gain and was removed. Current dynamics/signs/topology and
all update schedules remain unchanged. Next: fuse remaining learning bookkeeping,
verify exactness before benchmarking, then sustained throughput. Scientific
training remains paused; M1A learning acceptance remains unmet.
