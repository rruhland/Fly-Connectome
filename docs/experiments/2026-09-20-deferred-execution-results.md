# Deferred local updates and neuron execution

User-approved experiment on the measured M1A graph: 47,413 neurons,
1,377,103 edges, B=1, eight neural ticks per Pong frame. No topology, signs,
resting currents, timestep, or local-learning equation changed. Scientific
learning acceptance remains unmet; these runs measure execution, not efficacy.

## Balanced 1000-frame results

Each trial starts from the same checkpoint and five warmup frames, with Torch
one thread, native four threads, and event-only logging. Training remains online.
All source checkpoints remained unchanged. Results include model-state comparison
and every neural tick's spikes, rather than only total firing counts.

| Experiment | Reference fps | Experimental fps | Decision |
| --- | --- | --- | --- |
| Exact deferred tick loop | 47.02, 41.84 | 47.36, 32.85 | No reliable gain; retain only as experimental reference/fallback |
| Geometric local-error sum | 46.88, 47.78 | 52.18, 51.48 | About 9.5% throughput gain; remains separately identified |
| Exact fixed-point neuron sleeping | 47.72, 47.43 | 44.49, 45.95 | Slower; no production adoption |

The geometric result projects to about 3.2 minutes for 10,000 frames excluding
load/save. **120 frames/s remains unmet.** These are CPU measurements under
changing host conditions; compare the paired trials rather than unrelated peaks.

## What the deferred engine does

It retains at most eight ticks of each neuron's actual local error and the
anatomical presynaptic arrivals. Behavioral eligibility, R-STDP, expected-current
state, rates, homeostasis and all neurons still update every tick. Private visual
eligibility stays separate across synchronization boundaries; canonical state
is materialized before save, snapshot or explicit inspection. Weight proposals
are applied on the original synchronization ticks.

The exact version visits each visual edge once per interval and replays its
ticks in registers, preserving float32 ordering and pruning. Its first version
was slower because repeated splitting/merging erased the savings. Retaining
private arrays removed that bottleneck, but sustained trials still showed no
reliable advantage over the optimized per-tick backend.

The geometric version precomputes a per-neuron finite sum of its recorded local
errors weighted by powers of the existing current decay. For an edge without
presynaptic arrivals, its initial eligibility multiplies that sum. This preserves
every actual sensory/recurrent/clipping change in the recorded error sequence;
it does not infer a quiet sensory stream or discard small events. Trace and
eligibility decay still use their original float32 tick operations to prove that
no pruning crossing occurs. Arrival or pruning-crossing intervals use exact replay.

## Numerical audit

Exact deferred execution matched every final model tensor and every spike over
both 1000-frame trials. Geometric execution changed floating-point grouping:

- Zero differing spikes or spike times in both 8000-neural-tick trials.
- 44,336 weight entries differ; maximum absolute difference 4.47035e-8,
  mean absolute difference across all edges 5.80459e-12.
- Maximum voltage/current difference 5.96046e-8.
- Event squared-error total differs by about 7e-12; hits and rewards agree.
- Both geometric repetitions produce the same model-state hash.

This bounded observation does not establish indefinite spike equivalence. The
geometric backend records `deferred-geometric-v1` in saved manifests; the exact
experiment records `deferred-ticks-v1`. Neither replaces the production default.

## Neuron investigation

Only 3.55% of neuron-ticks receive new input, but 86.28% retain nonzero state;
94.97% of spikes occur without a same-tick arrival or injection. The complete
bitwise fixed-point candidate set is 16.29%. A sleeping prototype that wakes on
all arrivals/injection preserves every state bit, but takes 5.24% more total
time in balanced trials. Its mask/detection/dense-output costs exceed its savings.

Broader lazy execution must account for decaying currents, adaptation, refractory
expiry and autonomous threshold crossings. Input-only wakeup would change the
model. No approximate neuron propagation has been implemented.

## Reproduction and records

```powershell
.venv/Scripts/python -m fly_connectome build-native --output runs/native_cpu_deferred.dll
.venv/Scripts/python scripts/benchmark_deferred_cpu.py checkpoints/event-v1-combined-rate-10000.pt --library runs/native_cpu_deferred.dll --steps 1000 --output runs/deferred-ticks-1000.json
.venv/Scripts/python scripts/benchmark_deferred_cpu.py checkpoints/event-v1-combined-rate-10000.pt --library runs/native_cpu_deferred.dll --aggregate --steps 1000 --output runs/deferred-geometric-1000.json
```

Raw records are the adjacent `2026-09-20-deferred-ticks-1000.json`,
`2026-09-20-deferred-geometric-1000.json`, and
`2026-09-20-exact-sleeping-1000.json`. The neuron feasibility report includes the
fixed-point counts, counterexample and sleeping benchmark command.
