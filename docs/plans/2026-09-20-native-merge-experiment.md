# Native sparse-merge experiment, 2026-09-20

Scope: compile only the measured CPU eligibility merge hotspot. Keep the PyTorch
implementation canonical. Existing sorted active keys permit a linear merge after
sorting only incoming arrival keys, avoiding re-sorting the full active set. This
is an execution experiment, not a learning-rule or topology revision.

1. Write hand-derived empty/duplicate/signed-trace tests against the native wrapper.
2. Compile a standalone C ABI library with the installed g++, no fast-math or
   fused arithmetic; pass contiguous CPU tensors with explicit float32/int64 types.
3. Compare every merge output exactly with the tensor reference, then compare
   complete resumed training spikes and tensor states in balanced timing runs.
4. Retain only if beneficial; document compiler and commands. Adoption into the
   package/backend selection is a subsequent separately verified chunk. CUDA,
   packaging, and broader native kernels are outside this experiment.

Status: completed, equality checks and balanced measured-graph benchmark passed.
Native merge is ~23% faster than the optimized tensor merge; the full run is still
~32 times slower than real time. Keep it experimental. Results and continuation
instructions are in docs/experiments/2026-09-20-training-throughput.md.
