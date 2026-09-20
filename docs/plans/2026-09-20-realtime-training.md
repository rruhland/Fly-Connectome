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
