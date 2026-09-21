# Longer-term information survives in a few local traces

This investigation meets the preregistered exploratory stopping criterion:
three existing L2-to-L1 predictive connections pass next-event alignment checks
on both development trajectories, at three distinct postsynaptic neurons. None
pass the corresponding one-tick checks. The traces are measured on existing
anatomical edges with their fixed signs; no readout or weights were fitted.

This is a meaningful lead for local temporal credit assignment, not effective
M1A learning. Correlations are modest, the candidate set is small, and the current
network's forecasts still fail the zero-event baseline. Stop expanding the
diagnostic search here, as the user requested, rather than treating this as
permission to change architecture or start another long training run.

## Method and stopping rule

Run frozen initial and 10,000-frame checkpoints for 2,000 scripted frames on
development seeds 1101 and 1102. This increases observation time to 16.67 seconds
per seed so more local targets recur. Use common frames [12, 1988), 1,976 samples
per seed, for each trial. Graph and configuration hashes match prior experiments.
Final evaluation seeds remain unused.

Reconstruct each physical signed eligibility trace for measured predictive edges
into L1-L3, including warmup history. At camera-frame boundaries, compare the
target with traces from one tick, 30.208333 ms, 50 ms and 100 ms earlier. A bounded
ring stores completed-tick trace states, so the earlier trace cannot include the
target event. Separately compare frame-end traces with the first future nonzero
input at that sensory neuron in the following 12 frames; use zero for no event.
As before, that is a nominal 100 ms frame window, not a single fixed forecast lead.

The scripted camera stream is replayed separately to construct diagnostic labels.
Every actual frame-boundary injection is checked against that replay. Labels never
enter neural computation. Physical traces reconstruct sensory predictive current
at every game-frame end within 5.96e-8 initially and 2.98e-8 after training.

For each population and trial, seed 1101 selects up to ten edges by correlation.
Those same edges are checked on seed 1102. In each seed require:

- At least five distinct target events actually represented in the scoring window.
- Trace power above 1e-10.
- Signed trace/target correlation at least 0.05.
- Positive mean target-times-trace product.
- Covariance greater than a fixed timing shuffle, seed 421.

The stopping criterion was at least three distinct validation posts in one trial
and population. It is a practical exploratory criterion, **not** a statistical
significance threshold. Many edges and horizons are examined; one shuffle and
reused development seeds cannot establish generalization or milestone acceptance.

Next-event labels overlap in time. Repetition eligibility counts unique referenced
event indices, not the number of windows repeating an event. Review caught a
preliminary implementation counting unreferenced tail events; preliminary runs
were stopped, a regression was added, and both measured runs restarted after the
fix. The moment estimates still weight overlapping decision windows, as intended
by the diagnostic objective; they are not independent event samples.

## Results

Counts below are selected connections passing validation, with distinct
postsynaptic counts equal to the listed counts for these results.

| Trial | Initial L1 / L2 / L3 | Trained L1 / L2 / L3 |
| --- | --- | --- |
| One tick | 0 / 0 / 0 | 0 / 0 / 0 |
| 30.208 ms | 0 / 0 / 0 | 0 / 0 / 0 |
| 50 ms | 1 / 0 / 0 | 1 / 1 / 0 |
| 100 ms | 2 / 0 / 0 | 1 / 0 / 0 |
| Next event | 2 / 0 / 0 | **3 / 0 / 0** |

The three trained next-event candidates are all existing L2-to-L1 connections:

| Source body -> target body | Correlation 1101 / 1102 | Shuffled correlation 1101 / 1102 | Distinct events 1101 / 1102 | Trained magnitude |
| --- | --- | --- | --- | ---: |
| 50062 -> 51452 | 0.12158 / 0.11218 | -0.02495 / -0.04287 | 37 / 48 | 0.00399876 |
| 75482 -> 71110 | 0.06498 / 0.10104 | 0.02244 / 0.01308 | 8 / 14 | 0.00000973437 |
| 42416 -> 44723 | 0.06351 / 0.05435 | -0.01282 / 0.01617 | 20 / 6 | 0.0000134021 |

The first two also pass the next-event criterion in the initial network. Their
initial magnitudes were 0.035 and 0.03: training reduces them to about 11.4% and
0.0324% of initial strength, while their source traces retain longer-term alignment.
The third candidate only narrowly clears the correlation threshold on validation,
with six target events, and is especially tentative.

The first connection also passes the 50 ms test in both checkpoints. Its trained
correlations are 0.12191 / 0.10523, versus shuffled 0.00891 / -0.01625, with 38 / 49
target events. An existing C3-to-L2 connection (101422 -> 87871) also passes at
50 ms after training, with correlations 0.05612 / 0.08042 and 30 / 38 events.

This does not mean the whole visual circuit has a usable predictive basis. Only
55 / 94 L1 edges, 115 / 190 L2 edges and 7 / 5 L3 edges meet the trained next-event
event-count and trace-power eligibility checks on the two seeds, before testing
correlation. Limited exposure and lost activity remain substantial constraints.

## Interpretation and decision

The previous frozen-readout tests and this result address different questions.
The summed, learned current fails to predict well. A few individual signed local
traces nevertheless carry information about later events that is reproducible
across the two development trajectories under this screen. Longer-horizon local
information therefore has not been excluded by the failed current readout.

The result supports developing a **bounded proposal for local temporal credit
assignment** before abandoning the connectome topology or introducing a new
encoder/readout. It does not prove that extending an eligibility time constant,
changing the target, or restoring these weights would work. It also does not show
that these traces add information beyond all possible local event-history
baselines. Such controls and a causal online update mechanism belong in the next
proposal, along with unchanged no-backprop, fixed-sign and measured-topology rules.

Any new objective must confirm predictions only when real local events arrive
or their windows expire; offline future labels must never become online neural
input. Effects on C2 excitability and quiet-period false alarms also need explicit
checks. No architecture, learning target, delay, class current or milestone goal
was changed in this investigation.

## Reproduction and checks

```powershell
.venv/Scripts/python scripts/audit_trace_horizons.py checkpoints/event-v1-combined-rate-initial.pt --output docs/experiments/2026-09-20-trace-horizons-initial.json
.venv/Scripts/python scripts/audit_trace_horizons.py checkpoints/event-v1-combined-rate-10000.pt --output docs/experiments/2026-09-20-trace-horizons-10000.json
```

JSON records hashes, selection settings, sample windows, every selected candidate
including validation failures, and eligible/candidate counts. No trained artifact
is created. Source checkpoint checksums are verified unchanged.

Fixtures check seed separation using opposite correlations, completed-tick ring
lookup including wraparound, unique referenced-event counting at the tail,
actual camera replay, current reconstruction and checkpoint immutability.
Full suite: 194 passed, four CUDA skips. Review's counting issue was fixed with
a failing-then-passing regression. All measured processes finished.
