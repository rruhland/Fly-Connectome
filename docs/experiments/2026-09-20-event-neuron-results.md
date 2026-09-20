# Event-driven neuron experiment: not adopted

The user's approved experiment preserves the discrete equations, anatomical
edges, signs, delays, resting currents, local learning and weight timing. It
regroups no-input recurrences in float64, casting materialized state to float32.
No backpropagation, pruning, parameter changes or normalization were introduced.

The wake probe supports long sleep opportunities: median per-neuron mean gap
30.23 ticks; 96.24% of neuron-ticks have no natural wake. See
`2026-09-20-wake-intervals.md` for censoring and weighting definitions.

## Whole-training audit

`2026-09-20-event-neurons-1000.json` records four balanced 1000-frame runs from
the same 10,000-frame checkpoint, with five identical native warmup frames,
Torch one/native four threads, events-only metrics, and a 128-tick horizon.
Timing includes online learning, dense activity queries, scheduling and final
state materialization. The checkpoint was unchanged.

| Run | Frames/s | Changed spike entries | First changed tick |
| --- | ---: | ---: | ---: |
| Native reference | 36.58 | 0 | none |
| Event prototype | 46.02 | 930 | 250 |
| Event prototype | 51.19 | 930 | 250 |
| Native reference | 27.73 | 0 | none |

Reference timing varied substantially; these samples do not establish a stable
speedup. Neither candidate run reaches 120 frames/s. Both reproduce exactly the
same discrepancy counts: 853 of 8000 ticks differ, with 930 changed spike entries.
The largest final weight difference is 0.00501836, far larger than the previous
deferred-learning experiment's 4.47e-8. There are 15,226,674 active neuron updates
out of 379,304,000 neuron-ticks (4.01%). The skipped-tick counter excludes terminal
pending gaps, so use total minus active for the complete deferred fraction.

The small trajectory tests do not imply full-network numerical equivalence.
Regrouping changes rounding; threshold dynamics amplify discrepancies. The audit
does not yet isolate the first divergence enough to rule out every scheduler
defect. Regardless of cause, this implementation fails adoption criteria.
No longer scientific trajectory is warranted for adoption until that failure is
resolved. The native backend remains the reference and production default is
unchanged. Final evaluation seeds were not used.

## Reproduction and limitations

```powershell
.venv/Scripts/python scripts/benchmark_event_neurons.py checkpoints/event-v1-combined-rate-10000.pt --library runs/native_cpu_deferred.dll --event-library runs/native_event_neurons.dll --build --frames 1000 --horizon 128 --output runs/event-neurons-repeat.json
```

Raw JSON identifies the exact source and DLL hashes used. Subsequent review added
an overflow fallback and safe backend-switch materialization; the recorded run
predates those guards. Regression tests reproduce and cover both edge cases.
The experiment requires exclusive state ownership: materialize before external
state/config/silencing edits, then invalidate. Snapshots and saves materialize
automatically. A new native backend flushes and removes old materialization hooks.

Remaining model-preserving work should focus on dense loop/learning overhead.
This result does not establish that biological compromises are necessary, and
does not authorize any such compromise. Scientific learning acceptance remains
unmet; faster execution is not evidence of effective learning.
