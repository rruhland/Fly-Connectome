# Local temporal contrast loses ON forecast support

The bounded implementation of separate memory and event forecast fails frozen
preflight: ON trace availability 10/30, OFF 30/30, against the unchanged minimum
24/30 for each. No training or final evaluation was run. Reject this specific
event-contrast encoding; retain the prior area-matched model as the reference.

## Candidate and controls

Keep all physical area-matched 20 ms excitatory / 5 ms inhibitory currents and
neuronal dynamics unchanged. The forecast is now the signed local current
difference `I[t]-I[t-8]`, encoded using the original threshold and clipping.
Learning would use the corresponding eligibility difference captured at issue,
confirmed +8 ticks later. No learned readout, new connections or backpropagation.
See the [protocol](2026-09-21-event-contrast-protocol.md).

All 20 paired frozen precursor/blank probes have exactly the same recorded
spikes, target inputs and edge arrival counts as the previous area-matched
model. Unit tests independently verify bit-exact physical current, membrane,
adaptation and spike-history equality. Thus this is a change in forecast meaning,
not a changed neural response being mistaken for one.

Warmup-inclusive reconstruction of the contrast forecast from per-edge spike
history agrees within 6.39e-9. The failure is not an eight-tick indexing or
current-reconstruction discrepancy.

## Why preflight fails

For all ten randomized blank lengths, the first scored ON cycle has a usable
negative contrast trace. The second and third scored ON cycles do not:

| Scored cycle | ON availability | OFF availability |
|---|---:|---:|
| 1 | 10/10 | 10/10 |
| 2 | 0/10 | 10/10 |
| 3 | 0/10 | 10/10 |

Example: trial 0, cycle 2, target ON at tick 280, forecast issued at tick 272:

| Predictive source | Physical sign | Unit-magnitude forecast-contrast trace |
|---|---:|---:|
| L2 20655 | +1 | +0.101629 |
| L1 26550 | -1 | +0.276630 |

The needed target is -1, but both active contrast features are positive. All
other substantial input traces are zero. The current initial forecast is
+0.014989. Nonnegative magnitude adjustments cannot form a negative sum from
these fixed positive features. Changing recurrent weights might eventually
change the source spike trains, so this is not a proof of global impossibility;
it does fail the predeclared precursor support criterion.

The inhibitory synapse's physical current never changes transmitter sign.
Instead, its negative current becomes less negative as it decays, which produces
a positive temporal difference. The intended forecast-encoding change therefore
destroys useful ON polarity on these phases. Matching local credit does not
resolve a missing correctly signed feature at the issuance time.

## Decision

Stop before training; do not bypass the gate, change input polarity, permit
negative magnitudes, or sweep difference intervals to rescue this candidate.
Neither this failure nor the prior rise/decay result disproves the general idea
of retaining memory separately from a brief forecast. They do rule out the two
specific implementations tested here as current solutions.

A subsequent design should preserve the sign-bearing memory while controlling
when it is expressed, rather than using the signed derivative as the forecast.
For example, a nonnegative local timing gate multiplying the signed prediction
would preserve its polarity. The gate's state, causal trigger and local learning
mechanism still need a concrete design; no such gate is implemented or claimed
to work here. Do not count speculative timing-gate behavior as progress on M1A.

## Verification and artifacts

Focused tests cover physical-state equivalence, exact interval contrast,
matching issue-time eligibility and delayed confirmation. Existing forecast
capture was factored into a hook with unchanged default behavior. Full suite:
223 passed, 4 CUDA skips. Independent review found no implementation blocker,
and highlighted the intentional distinction between physical and forecast signs.

Frozen preflight completed in 4.50 s. Source checkpoint and frozen weights remain
unchanged; no weights were trained. Experimental state is reference-runner only,
without production checkpoint/native support. Production defaults remain
unchanged. M1A remains unmet; no full-Pong or M1B work was performed.

[Full preflight results](2026-09-21-event-contrast-results.json) and
[local artifact hashes](2026-09-21-event-contrast-artifacts.json).
