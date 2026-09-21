# Delayed excitatory peak does not localize forecasts

The approved 5 ms rise / 20 ms decay experiment improves event anticipation
but worsens quiet-frame false alarms. Reject this candidate as a temporal-
localization fix; retain the prior equal-area exponential as the comparison
reference. M1A remains unmet.

The [protocol](2026-09-21-rise-decay-protocol.md) specifies one fixed, positive
excitatory kernel, area matched to the 5 ms reference. It starts at zero and peaks
about 9 ms after arrival. Matching local eligibility follows the same two
components. Inhibition, signs, topology, delays, input timing, frame-horizon
supervision and no-backprop remain unchanged. Same 200 training trials, initial
magnitudes and 50 frozen evaluation trajectories as the previous area control.

| Frozen learned metric | Equal-area exponential | Rise-and-decay |
|---|---:|---:|
| Overall MSE | 0.108329 | 0.103932 |
| ON MSE | 0.532337 | 0.499245 |
| OFF MSE | 0.792185 | 0.691499 |
| Signed ON anticipation | 0.281664 | 0.308595 |
| Signed OFF anticipation | 0.110316 | 0.169321 |
| Quiet MSE | 0.018648 | 0.024368 |
| Quiet false alarms | 25.69% | 38.69% |
| Correct ON/OFF signs | 150/150 each | 150/150 each |

Lower overall error does not establish success: it coexists with worse quiet
error and many more false alarms. The fixed 5% quiet gate fails decisively.
Including the uncued first cycle does not reverse the conclusion: all-frame
false alarms increase from 23.19% to 34.92%.

## Temporal localization checks

Same quiet samples and 0.1 forecast threshold, with no exclusion changes:

| Quiet group | Previous false alarms | Rise-and-decay false alarms |
|---|---:|---:|
| First frame after ON | 101/150 | 92/150 |
| Second frame after ON, before OFF | 94/150 | 142/150 |
| First frame after OFF | 1/150 | 92/150 |
| Second frame after OFF, before ON | 38/150 | 63/150 |
| Initial pre-stimulus blank | 9/32 | 9/32 |
| First 1-3 blank frames after motion | 117/148 | 146/148 |
| Blank frames 4-6 | 110/147 | 120/147 |
| Blank frames 7 onward | 6/926 | 53/926 |

These groups account for all quiet frames: 476 versus 717 false alarms among
1,853 samples. Early positive OFF forecasts become more common, positive
predictions linger after OFF, and post-motion predictions persist longer.
The small improvement immediately after ON does not offset those regressions.

## Interpretation and next decision

The kernel moves its peak but does not remove its 20 ms tail. With equal impulse
area, delaying the rise redistributes excitation later. The resulting coupled
learner produces stronger but still broad forecasts. This is evidence against
this specific response as a localization fix, not against every possible
rise/decay time constant or against local predictive learning in general.

Do not promote this model, weaken OFF globally, relax the gate or launch a time-
constant sweep. The remaining architectural question is whether using the
memory current itself as the event forecast couples output duration too closely
to memory duration. A next proposal should explicitly separate a retained local
memory from a brief event prediction, with matching causal eligibility, rather
than merely shifting the same long-lived output kernel. That change has not
been implemented or assumed authorized by this bounded experiment.

## Verification and reproducibility

New tests verify the actual impulse/current/eligibility formulas, nonnegative
excitation, discrete impulse area, delayed peak, retained issue-time trace and
rejection of unmatched one-tick supervision. Full suite: 221 passed, 4 CUDA skips.
Independent review found no blocker in current decomposition, sparse-key retention,
eligibility timing or default behavior. Defaults remain unchanged.

Warmup-inclusive frozen current reconstruction error is 5.38e-9, below unchanged
1e-6 tolerance. Both frozen precursor availability checks pass 30/30. Training
targets, initial magnitudes and blank intervals match the previous run exactly.
All checked states stay finite, magnitudes bounded, and source checkpoint and
frozen evaluation weights unchanged. The main run took 107.95 s.

The additional eligibility stores are local per-edge experimental reference
state, not a production optimization or supported checkpoint/native backend.
No full-Pong runs, M1B advancement or additional training sweep were performed.

[Full results](2026-09-21-rise-decay-results.json),
[quiet-phase counts](2026-09-21-rise-decay-quiet-phases.json), and
[local artifact hashes](2026-09-21-rise-decay-artifacts.json).
