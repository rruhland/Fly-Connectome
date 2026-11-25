# Exact rollout prefix reuse

The ordinary elastic matcher first runs a batch of stock and remembered
sequences with diagnostic traces. It then appends additional proposals and
runs a short screen on the enlarged batch. The first batch's unchanged
prefix repeats exactly the same native simulations. This change keeps
those outputs for the immediately following short screen. It does not
change proposals, random numbers, ranking, simulator settings, model
calibration or issued joint commands.

Across four counterbalanced whole-game pairs on two carried assessment
releases, prefix reuse reduced total process CPU by **6.6%**. Every pair
sped up by 2.3–11.4%. The issued controls and referee events were identical
on all runs.

| Across the four pairs | Stock | Reuse |
| --- | ---: | ---: |
| Whole play loop CPU | 39.73 s | 37.10 s |
| Native near-side CPU | 16.00 s | 14.05 s |
| Native candidate rows | 3,060 | 2,572 |

## What was worth fixing

On a real carried-stock run of game 11884, near-side native trials used
6.19 of 14.42 process CPU seconds. Stereo rendering took 4.58 s, actual
table exchange 2.18 s and optical fitting 0.67 s. Fragment retrieval was
0.047 s across 20 searches, about 0.3% of the whole game. Separate older
optical native profiles measured warm per-call model allocation near 1 ms,
versus about 115–130 ms for those 25-candidate batches. These did not suggest
prioritizing retrieval or a persistent native workspace.
[Earlier native profile](../tt40/rollout-bounds.md)

The stock benchmark checks, bit for bit, whether the outputs of the
initial traced rows agree with the corresponding rows in the repeated
untraced batch, including losses, legality, native outcomes and contact
feedback. The reuse benchmark checks every issued increment, referee event,
opponent placement, search note, motor audit and learned-memory digest
against the exact archived stock tape. It also checks one start per run
and every independent random opponent target. CPU timings include the
whole physical play loop, not just the easy selected calls. Batch counters
also show the actual native candidate reduction.
[Paired timing and parity](real-loop.json) · [Archived commands](../tt43_motor_modes/held-stock.json.gz)

The four timing pairs used implementation commit `dcf136c`; the
subsequent stock-source guard and patch binding ran the same optimization
on one extra short pair. That pair was again exact and omitted 69 native
rows, though its single whole-loop CPU comparison was slower (5.06 to
5.50 s) under the shared workload. Treat the four-pair timing as evidence,
not a minimum speedup on every game. [Guarded binding check](guarded-binding.json)

## Opt in

Import `ReuseMPC` from `scripts.experimental.tt45_rollout_reuse` in a
new runner and instantiate it where that runner would otherwise create
the stock `MPC`. To use the unchanged `tt42_learning_course.run` entrypoint
with an explicit opt-in, wrap that one invocation in
`patch.object(course.mpc, "MPC", ReuseMPC)`. No pinned runner, native
source file, output schema or ABI needs to change.

The prefix is used only between the first traced call and second
untraced call in the reviewed stock search. Its input rows must match
as float64 bits and remain in the same order. Uncertainty trials and
long rollouts are untouched. The guarded branch does not use this reuse.
The optimized binding also falls back when the reviewed stock methods
change. No model, buffer address, native pointer, or cross-search result
is retained. Logical trial counts are preserved. The new
`reused_native_candidates` diagnostic reports the omitted physical work.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt45_efficiency
python -m scripts.experimental.tt45_rollout_efficiency "$out/real-loop-rerun.json" --indices 2 6 --repeats 2
python -m scripts.experimental.tt45_rollout_efficiency "$out/guarded-binding.json" --indices 6 --repeats 1
python -m pytest -q tests/test_tt45_rollout_reuse.py
```
