# Causal calibration preserves useful anticipation at unseen timing

The fixed split event-balanced amplitude reference can be localized using a
reference acquired from past local observations alone. This is a promising
offline capacity result, not successful biological learning or an M1A pass.
No neural weights, topology, signs, transmission, or plasticity were changed.

## Fixed diagnostic rule

After a local event arrives, store the absolute sensory state from one frame
earlier. Issue subsequent forecasts only when the current absolute sensory
state lies within half a frame of exponential decay on either side of that
reference. Preserve the original amplitude inside the window; zero outside.
Use no tempo, phase, trial reset, or future target. This adds a reference and a
one-frame sample buffer to the diagnostic, not to the production model.

The rule and width were declared before running; no search or amplitude refit.
The complete sensory stream includes unscored initial cycles and blank periods.
Reconstruction matches saved issue states exactly at every checked row. All
source checksums and reference coefficients are preserved in the results.

## Held-out frozen histories

| Tempo | ON amplitude | OFF amplitude | ON retained | OFF retained | Quiet alarms before -> after | MSE after |
|---|---:|---:|---:|---:|---:|---:|
| 2 | .3265 | .5588 | 100% | 80.4% | 21.61% -> 0% | .0842 |
| 3 (unseen in amplitude fitting) | .6335 | .8601 | 84.2% | 100% | 29.81% -> 2.67% | .0490 |
| 4 | .4572 | .6225 | 59.8% | 100% | 41.16% -> 2.33% | .0478 |
| 6 | .3842 | .2004 | 100% | 100% | 51.82% -> 1.64% | .0537 |

All four satisfy the anticipation/quiet subset of criteria on these frozen
histories. Tempo 4 fails the declared 80% amplitude-preservation reference.
Unseen tempo 3 adapts causally to its own past observations: this is not a
zero-shot fixed gate. Strong amplitudes still come from an offline supervised
fit, not from the actual local update rule.

## Continuous sensory-only stress

Use 24 alternating events in each interval block 2,3,6,4 frames, with no reset.
The selector initially misses four events at interval 2. At each later switch
it misses the first event, then predicts every remaining event in the block.
There are four quiet activations in 313 quiet frames: initial acquisition,
two slower-tempo transitions, and one after motion stops. This tests only the
selector: no neural amplitudes were synthesized or independent neural runs
stitched together.

Omit a pair of events in the middle of the interval-6 block without warning.
The first omitted event still produces a forecast, exactly as causality requires
for its identical prefix. The second does not. When events resume, two are
missed; the third and subsequent events are predicted. Quiet activations total
five in 315 quiet frames. The first returning event records a near-zero
reference after the long silence, which explains the extra recovery miss.
This is a concrete weakness of replacing the reference with the last sample.

## Implication and next bounded question

External tempo selection is not necessary for useful timing on these repeated
trajectories. A simple past-event reference is enough to preserve most of the
saved anticipation at unseen tempo 3 and suppress most quiet output. It does
not establish general vision prediction, robustness to irregular motion, or
that the same rule is biologically justified.

Before an architecture amendment, run a frozen-network continuous challenge
with tempo changes and omissions, reconstruct the same saved amplitude readout
from its actual new eligibility histories, and apply this unchanged causal
selector. Include startup, switches, returns after silence, and steady state
separately. That tests whether the amplitude capacity survives the situations
where the sensory-only selector adapts. Do not fit a new amplitude solution to
the challenge. If it survives, propose a local expression-and-credit mechanism
together: a gate added only to expression cannot be assumed to fix shared-weight
credit conflicts. Obtain approval before changing the neural model.

## Reproduction

`python scripts/causal_timing.py`

- [Predeclared protocol](2026-09-22-causal-timing-protocol.md)
- [Exact scores, hashes and per-event stress results](2026-09-22-causal-timing-results.json)
- Local complete replay arrays: `runs/causal-timing-v1/replay.npz`
- Tests cover impulse timing, causal prefix invariance, acquisition and omission.
