# Lazy neuron execution feasibility

## Scope and reproduction

Read-only continuation of `checkpoints/event-v1-combined-rate-10000.pt`, 100 Pong frames / 800 neural ticks, Torch 1 thread and existing native DLL 4 threads. Learning continued in memory using the saved rules; no checkpoint was saved and no neuron engine, topology, signs, or rules were changed. Metrics mode was `events`. There was no extra warmup after restoring the checkpoint. This is a bounded activity sample, not an evaluation of learning quality or a speed benchmark.

```powershell
.venv/Scripts/python.exe scripts/profile_neuron_activity.py checkpoints/event-v1-combined-rate-10000.pt --native-library runs/native_cpu.dll --native-threads 4 --frames 100 --output docs/experiments/2026-09-20-lazy-neuron-feasibility.json
```

Checkpoint SHA256 before and after: `6358107d489ee859f1644105eb274c29b226e6cdc71ae7e58d65a8bf6c925e58`.
DLL SHA256: `40629c97be983987060e4e3571758675cc92383b129819de7559015f7917a12b`.
Instrumented run: 6.737 seconds. Instrumentation scans and clones dense states every tick, so this timing must not be compared to uninstrumented throughput.

## Measurements

The graph contains 47,413 neurons and 1,377,103 edges. Fractions below use 37,930,400 neuron-ticks as denominator; an arrival means at least one anatomically delayed edge arrival even if its signed weight is zero or cancels another arrival. State tests use exact `!= 0`, without an epsilon or pruning.

| Condition | Neuron-ticks | Fraction |
| --- | ---: | ---: |
| Spiking | 42,549 | 0.1122% |
| Receiving arrivals | 1,345,693 | 3.5478% |
| Receiving injection | 219 | 0.000577% |
| Receiving either | 1,345,911 | 3.5484% |
| Any nonzero pre-step dynamic state | 32,725,600 | 86.2780% |
| Nonzero rest current | 13,516,000 | 35.6337% |
| No arrivals/injection, but state or rest drive | 31,379,689 | 82.7297% |
| Exactly zero state, zero rest, no arrivals/injection | 5,204,800 | 13.7220% |
| No-input bitwise fixed state, including nonzero | 6,178,544 | 16.2892% |
| Nonzero state/rest subset of bitwise fixed state | 973,744 | 2.5672% |
| Spiking with no arrivals/injection that tick | 40,410 | 0.1065% |

Exactly 6,506 neurons qualified as quiescent on every sampled tick. The other 40,907 had at least one nonzero state on every tick. Nonzero voltage occupied 85.9414% of neuron-ticks, predictive current 75.1867%, feedforward current 49.6474%, adaptation 29.1340%, sensory state 4.2180%, and refractory state 0.2244%. Behavioral current was zero throughout this sample. The JSON includes per-cell-class counts and the complete saved neuron configuration.

**94.97% of observed spikes occurred without a same-tick arrival or sensory injection.** Thus spike sparsity or external-input sparsity does not establish neuron-update sparsity.

The extended fixed-point probe compares all six float32 state tensors by their int32 bit patterns, plus exact int64 refractory state, before and after an input-free, spike-free tick. This includes signed zeros and subnormals without tolerance. It found 7,917 neurons fixed at least once and 7,435 fixed on every sampled tick; the nonzero subset contains 1,411 and 929 neurons respectively. Among 6,165,705 fixed-state transitions to a next tick still free of arrivals/injection, all remained bitwise fixed: zero failures. Transition metrics in JSON retain the overall 800-tick denominator, although transitions are observed only for the last 799 ticks. This establishes a larger exact-sleep candidate set than zero-only detection in this sample.

## Exact skip conditions and counterexample

`dynamics.py` and `native_cpu.cpp` decay all three pathway currents, update the sensory filter, decay adaptation, decrement refractory time, integrate membrane voltage including rest current, and then apply threshold/reset on each tick. The approved checkpoint uses a 20 ms sensory filter and nonzero class rest currents (for example L1/L2/L3: 1.5; Mi1/Tm3/Tm1/Tm2/Mi9: 1.2).

A sufficient exact quiescent condition is zero voltage, all three pathway currents, sensory state, adaptation, refractory counter, and rest current, with no current arrival or injection. Such a neuron stays zero until awakened by an arrival or injection. History slots and externally consumed activity outputs must still be set correctly, and future delayed events must still be delivered. This is a conservative sufficient condition; it does not claim to enumerate every possible fixed point, cancellation, or special silenced-neuron case. The measured candidate fraction is 13.7220%, before queue/mask overhead and before dense activity/learning consumers are considered.

A more general sufficient condition is a confirmed bitwise fixed point of the complete no-input, no-spike neuron transition. With unchanged parameters/silencing and no arrival/injection, that deterministic local transition repeats identically; nonzero equilibria can sleep too. Wake on arriving edges, injection, or relevant parameter/silencing changes. Current observations may be nonzero and must still be furnished to learning/metrics unchanged; sleeping the neuron is not permission to skip its learning state or homeostasis. The extended probe raises the measured candidate fraction to 16.2892%. Fixed-point detection itself has cost and cannot be implemented by checking voltage alone.

Unsafe naive skipping counterexample: an isolated, eligible L1 neuron with all dynamic states zero and its approved rest current 1.5 receives no external event. With no spikes yet, real-arithmetic voltage is `V(k) = 1.5 * (1 - exp(-k * dt / 0.02))`. At the checkpoint's `dt = 1/960` seconds it crosses threshold 1 on tick 22. An implementation that only updates neurons on arrival/injection never emits that spike. The measured 40,410 no-input spikes demonstrate that autonomous dynamics are consequential in the actual checkpoint, though the probe does not attribute each such spike to rest versus residual currents.

For a nonzero neuron without new input, the discrete current/sensory/adaptation recurrences continue to evolve, and voltage follows a driven recurrence. Refractory expiration and autonomous threshold crossings require scheduled events; resets create additional boundaries. Positive/negative pathway mixtures and decaying adaptation preclude assuming monotone voltage or a simple single exponential threshold time. Even a below-threshold rest current does not make the current voltage or adaptation static.

Closed-form multi-tick propagation is algebraically available between events, but exponentiation and regrouped sums generally differ from repeated float32 multiplies/adds. Crossing times and eventual spikes can diverge near threshold. Achieving mathematical equivalence is distinct from achieving the current discrete float32 trajectory; an exact-trajectory implementation needs a demonstrated rounding-preserving method or replay of the omitted operations. Very small nonzero values cannot be discarded under the current model. Float32 subnormal decay can also stagnate above zero; this probe does not silently treat such values as inactive.

## Recommendation

Keep the dense native neuron update for the current approved model. The separate exact-sleeping experiment below did not improve performance, so reject production adoption of that implementation. The activity sample offers 16.3% candidate neuron work removal (13.7% zero-only), not the 96.5% suggested by external-input sparsity.

The existing `2026-09-20-native-final-profile-1000.json` reports 3.623 seconds of neural-kernel work out of 20.545 seconds total. As a rough proportional-work estimate, removing 16.2892% of that kernel would save about 0.590 seconds, or 2.87% total runtime (about 1.030x speedup), before overhead. The zero-only estimate is 0.497 seconds or 2.42%. This combines a different-length profile with this activity sample and is not a measured speedup or strict upper bound.

A full event-driven adaptive LIF implementation is a separate algorithm project: define numerical-equivalence requirements, handle refractory and autonomous spike scheduling, preserve local observations/traces/homeostasis at their required times, and validate spike/state/weight/resume trajectories. Any epsilon-to-zero state pruning, changed discretization, approximate threshold scheduler, or altered biological parameters requires explicit model-change approval. No such change is implemented here.

## Exact fixed-point sleeping microexperiment: no gain

`scripts/native_sleeping.cpp` and `scripts/benchmark_sleeping.py` implement an isolated alternative kernel with persistent per-neuron sleep flags. It wakes before decay on every arriving edge and bit-nonzero injection (including negative zero), detects a complete bitwise fixed state with no spike, and continues producing dense observations/predictions/increments/spike outputs. It preserves the native scalar operation order, float32 rounding, parameters, and delayed-event behavior. It changes no production defaults or neuron implementation.

The wrapper is explicitly scoped to exclusive ownership of dynamic state and immutable config/topology/silencing between calls; external state or parameter edits require `invalidate(network)`. Normal synaptic weight updates are supported because every arriving edge wakes its target. Sleep flags are not checkpointed. This limited experiment is not a general replacement for a mutable public neuron API.

Validation: `.venv/Scripts/python.exe -m pytest tests/test_sleeping.py -q` passed **4 tests in 3.20 seconds**. Tests compare exact tensor bits across full small-network trajectories with autonomous spikes, nonzero equilibria, delayed arrivals, inhibition, refractory periods, filtered/unfiltered sensory input, signed zero, smallest-subnormal persistence, and explicit invalidation after external state/silencing changes.

The complete checkpoint benchmark used Torch 1/native 4 threads, 5 warmup frames and 1,000 measured frames per run, in balanced dense/sleeping/sleeping/dense order. Both backends use the same `runs/native_cpu_deferred.dll` for all other native operations, with the sleeping neuron kernel in a separate `runs/native_sleeping.dll`. The benchmark does not select an experimental deferred-learning rule; it changes only the neuron kernel. Hash instrumentation is the same in both modes.

```powershell
.venv/Scripts/python.exe scripts/benchmark_sleeping.py checkpoints/event-v1-combined-rate-10000.pt --native-library runs/native_cpu_deferred.dll --sleeping-library runs/native_sleeping.dll --build --frames 1000 --repeats 2 --output docs/experiments/2026-09-20-exact-sleeping-1000.json
```

| Run | Mode | Total seconds | Frames/s | Neural-kernel seconds |
| --- | --- | ---: | ---: | ---: |
| 1 | Dense | 20.9555 | 47.72 | 3.4249 |
| 2 | Sleeping | 22.4778 | 44.49 | 4.6741 |
| 3 | Sleeping | 21.7645 | 45.95 | 3.8521 |
| 4 | Dense | 21.0834 | 47.43 | 3.3335 |

Sleeping was slower in both comparisons: aggregate elapsed time increased **5.24%**. Its bitwise checks, status accesses, branches, arrival wake pass, and retained output writes outweigh the work skipped for this checkpoint. There were 7,779 sleeping neurons at the end of both sleeping runs. These runs reject this implementation's performance benefit; they do not prove every possible exact lazy algorithm is slower.

All four runs produced identical cumulative per-tick spike SHA256 `223eb6df09bfdf69d4b129b4a056f9148431c03b543bce8c8902ef3ad3597289` (including warmup), and identical final whole-model tensor SHA256 `f2ba5c654e3264d5faff941baa8f8adaf6413ff7dc4ebefc710d7b9ad2ec3555`. Checkpoint SHA256 before/after matched the identity above. Source/DLL hashes and their before/after stability checks are recorded in `2026-09-20-exact-sleeping-1000.json`. No approximate dynamics were used, and the experiment remains separate from production.
