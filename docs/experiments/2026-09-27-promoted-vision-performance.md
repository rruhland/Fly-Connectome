# Promoted M1A.5 vision performance pass

## Scope and result

This pass optimizes the approved `load_default()` visual service with
`step(..., learn=True)` and `VisualStateEncoder.encode`. It preserves the
grayscale-plus-event sensor, all structured outputs, online credit, append-all
forecast memory, and checkpoint version 3. The older measured-graph Pong trainer
is outside the user-selected scope. The production visual service is not yet
connected to Pong physics or motor control, so these are visual-service sample
rates, not game frame rates.

On alternating same-host CPU runs, the optimized median is **14.52 ms/sample
(68.9 samples/s)** for two visible entities and **39.86 ms/sample (25.1
samples/s)** for eight. The two-entity median exceeds the 50 Hz target; the
eight-entity case does not. Neither case demonstrates sustained 120 Hz.

## Measurement

Each process used the existing `scripts/benchmark_visual_interface.py` workload:
64x64 every-sample frames plus events, one PyTorch CPU thread, online learning,
full mixture and interval output, and sparse transport encoding. Each arm used
40 warmup and 200 measured samples. Original and optimized source paths were
verified separately, and process order was optimized, original, optimized,
original. Camera acquisition, rendering, and Pong simulation were excluded.
Host background load was not controlled; paired runs limit its effect on the
comparison but do not establish a hardware-independent latency guarantee.

| Run | Two entities p50 / p95 ms | Eight entities p50 / p95 ms |
|---|---:|---:|
| Optimized 1 | 14.82 / 18.21 | 40.85 / 47.86 |
| Original 1 | 34.14 / 39.62 | 137.40 / 155.79 |
| Optimized 2 | 14.22 / 18.13 | 38.87 / 47.64 |
| Original 2 | 35.01 / 68.14 | 139.24 / 162.70 |

The mean of the paired p50 measurements improves **2.38x** for two entities
and **3.47x** for eight. The eight-entity encoder portion measured 12.00/12.10
ms in the original runs and 9.53/8.52 ms in the optimized runs. An earlier
unpaired clean-worktree baseline was 21.62/85.64 ms; absolute CPU timings
varied substantially across the session, so the alternating comparison is the
primary result. A separate shorter 1/2/4-thread sweep gave eight-entity p50s
of 45.14/44.80/43.87 ms; this did not justify changing the single-thread
reference setting.

## Changes and parity

- The 32-iteration marginal-interval search now runs independent mixtures in
  batches grouped by component count. It uses the original scalar PyTorch
  operations through nested `vmap`. Issued mixture centers/weights and
  per-horizon calibration summaries are reused within each sample.
- Pending online outcomes are appended once per horizon to the short and long
  context banks. Every original example, insertion order, PIT calibration rank,
  and `seen` count is preserved. Other dynamics retain sequential updates.
- Context-key variance is cached until its key tensor changes or is modified
  in place; inference-mode tensors recompute variance because they have no
  version counter. The sparse transport encoder keeps its deep-copy behavior
  and orders mixture components through a stable lexicographic sort.

The original checkout and optimized worktree produced **identical SHA-256
digests for every full state and encoder output** over 82 online samples,
including three unavailable camera frames. Their saved learned checkpoints
matched, as did 22 samples after loading into a fresh scene and 30 frozen
evaluation samples with outages. The two trajectory-result JSON files had the
same SHA-256:
`221ca92d0260dc3c7c67b00e7891775f656cd55cac525cd27409e3b70804233e`.
The learned-checkpoint content digest was
`8262882a08163bae8141e8f3014e5d47d322f547bf4fcbb71410f723007a5c69`.
Focused tests cover singleton and 32-component intervals, canonical component
order, tensor aliasing, inference-mode keys, variance invalidation,
sequential-versus-batched bank updates, online and frozen outputs, and checkpoint state. The full suite passed:
**731 passed, 4 CUDA skips**.

## Remaining cost

The warmed eight-entity profile is now dominated by sequential identity
association and context-state processing, followed by transport assembly and
mixture retrieval. Those branches decide identity, matching, and future credit.
The forecast banks still keep every new example, so exact retrieval and storage
cost continue to grow with experience; the 240-sample benchmark does not prove
long-run 50 Hz operation. A separate synthetic bank-size probe with 2,000 versus
20,000 random examples per bank measured eight-entity p50s of 43.28 versus
68.75 ms. It isolates capacity pressure, not a representative trained-state
trajectory. A native tracker or a batched retrieval design would
need its own exact trajectory and checkpoint parity evidence before promotion.
Approximate neighbors, fewer retained examples or mixture components, and fewer
interval-search iterations would change the approved learning or outputs and
were not used. PyTorch's first `vmap` call also has one-time import/setup cost;
the throughput measurements include 40 warmup samples and do not describe that
first-use latency.
