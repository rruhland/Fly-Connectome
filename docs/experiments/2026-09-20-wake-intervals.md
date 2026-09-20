# Wake intervals on the approved checkpoint

**Typical neurons have gaps of tens of ticks, rather than repeatedly waking every 1–3 ticks.** Among the 40,022 neurons with at least two observed wake events, the median of each neuron's mean completed gap is **30.23 ticks**; the 10th/90th percentiles are **13.24 / 110.54 ticks**. Short gaps remain common when weighting by individual events, because frequent neurons contribute more events.

This measures opportunities for an event-driven implementation, not its correctness or speed. It changes no neuron dynamics, topology, signs, rest currents, plasticity rules, or numerical thresholds. The earlier fixed-point sleeping experiment remains slower; this probe asks a different question about intervening events.

## Method

Read-only resume from `checkpoints/event-v1-combined-rate-10000.pt`, using existing `runs/native_cpu.dll`, Torch 1/native 4 threads. Recorded 1,000 frames = 8,000 neural ticks on 47,413 neurons, without additional warmup. One tick is 1/960 second. Learning continued normally in memory and no checkpoint was written. Instrumented execution took 27.708 seconds; it is not a throughput benchmark.

```powershell
.venv/Scripts/python.exe scripts/profile_wake_intervals.py checkpoints/event-v1-combined-rate-10000.pt --native-library runs/native_cpu.dll --frames 1000 --output docs/experiments/2026-09-20-wake-intervals.json
```

A natural wake is the per-neuron union of any arriving edge, nonzero sensory injection, spike, or refractory expiry. Arrivals count even when effective current is zero or cancels another arrival. Refractory expiry is the transition from counter 1 to 0. Autonomous spikes are spikes without a same-tick arrival/injection; they are counted separately and are already included in the spike union. Simultaneous categories count as one wake.

A completed gap is `next_event_tick - previous_event_tick`; a gap of 1 therefore has no intervening event-free tick. The first and last window intervals are censored: their observed event-free prefixes/suffixes are reported, but never entered as finite completed gaps. Cells with no event have fully censored windows; cells with one event contribute no completed gap.

Event-weighted statistics give every completed gap equal weight. Neuron-weighted bin fractions first normalize each neuron's completed-gap histogram, then average equally over neurons with at least two events. Neuron-weighted mean gap is the mean of those per-neuron means. The quantiles of per-neuron means are explicitly distinct from quantiles of individual gaps. Per-class summaries use these same definitions and are included in JSON.

## Natural events

| Completed gap | Count | Event-weighted fraction | Equal-neuron-weighted fraction |
| --- | ---: | ---: | ---: |
| 1–3 ticks | 4,485,458 | 31.50% | 16.35% |
| 4–9 ticks | 3,038,550 | 21.34% | 15.45% |
| 10–99 ticks | 6,269,803 | 44.03% | 52.89% |
| 100+ ticks | 446,125 | 3.13% | 15.30% |

The event-weighted mean gap is **22.18 ticks**, with p10/p50/p90/p99 **1 / 8 / 63 / 143**. The equal-neuron-weighted mean is **64.64 ticks**, with per-neuron-mean p10/p50/p90/p99 **13.24 / 30.23 / 110.54 / 643.30**. The latter mean is increased by relatively quiet cells and should not replace the median as a description of a typical active neuron.

There are **7,357 never-waking neurons** and **34 neurons with just one event**. Neither group is assigned a finite mean gap. There are **365,024,008 event-free neuron-ticks (96.2352%)**, including 2,224,337 left-censored prefix ticks, 2,274,422 right-censored suffix ticks, and 58,856,000 fully censored ticks. Total wake neuron-ticks are 14,279,992 and completed gaps are 14,239,936.

Category counts overlap: arrivals 13,458,188; injection 1,908; spikes 426,502; autonomous spikes 405,354; refractory expiries 426,367. Autonomous spikes constitute about 95.04% of spikes, so a correct scheduler must account for future autonomous crossings rather than simply wait for external input.

Selected class summaries, in ticks:

| Class | Neurons | Never waking | Mean of per-neuron mean gaps | Median per-neuron mean gap |
| --- | ---: | ---: | ---: | ---: |
| L1 | 892 | 0 | 30.63 | 29.56 |
| L2 | 893 | 0 | 29.26 | 29.91 |
| L3 | 892 | 0 | 53.52 | 54.90 |
| Mi1 | 887 | 0 | 29.66 | 30.18 |
| Tm3 | 1,037 | 0 | 12.54 | 12.39 |
| T4a | 849 | 0 | 26.72 | 25.11 |
| T5a | 838 | 0 | 33.27 | 32.55 |
| LC17 | 175 | 175 | Censored | Censored |

## Artificial materialization every eight ticks

A second diagnostic schedule unions the natural wakes with a forced event for every neuron at the end of ticks 8, 16, 24, and so on. This models a design that materializes all neuron states at each frame/synchronization boundary. It adds no real simulation operations and is reported separately from biological events.

| Completed gap | Event-weighted fraction | Equal-neuron-weighted fraction |
| --- | ---: | ---: |
| 1–3 ticks | 20.46% | 16.15% |
| 4–9 ticks (maximum actually 8) | 79.54% | 83.85% |
| 10–99 ticks | 0% | 0% |
| 100+ ticks | 0% | 0% |

All 47,413 neurons now have completed intervals. Mean gap falls to **6.33 ticks event-weighted / 6.64 ticks neuron-weighted**; individual-gap p50/p90/p99 are all 8. Scheduled wake neuron-ticks rise from **14.28 million to 59.92 million**, about 4.20 times as many. Event-free neuron-ticks fall to **319,385,371 (84.2030%)**. The first forced event still leaves 292,532 left-censored prefix ticks; the final forced event leaves no right-censored suffix. No interval longer than eight ticks survives.

## Interpretation and validation

Natural wake spacing supports investigating event-driven propagation over tens of ticks for many neurons. It does not establish that all those ticks can be omitted from current execution: dense learning observations, motor-rate updates, and metrics are separate consumers, and autonomous spikes/refractory boundaries need scheduling. Synaptic synchronization changes weights for future arrivals; whether it actually requires whole-neuron materialization depends on the implementation. Forcing all cells awake every eight ticks would substantially restrict the natural opportunity.

The probe observes autonomous events from the reference trajectory; it does not implement a predictor or validate an analytic threshold solver. Approximate state pruning, cheaper biological parameters, and changes to the discrete dynamics are outside this diagnostic. Window censoring can particularly bias estimated means for quiet cells; absence of a wake in 8,000 ticks is not proof of indefinite inactivity.

Checkpoint SHA256 before/after: `6358107d489ee859f1644105eb274c29b226e6cdc71ae7e58d65a8bf6c925e58`. DLL SHA256 before/after: `40629c97be983987060e4e3571758675cc92383b129819de7559015f7917a12b`. The script identity and before/after hashes are recorded in JSON. A small synthetic gap/censor fixture passed, and arithmetic checks verified both schedules: category histogram totals, completed gaps = events minus neurons ever waking, and exact decomposition of event-free ticks into completed-interval interiors plus censored prefixes/suffixes/windows.
