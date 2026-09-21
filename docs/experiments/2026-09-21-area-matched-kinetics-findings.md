# OFF learning survives equal-area excitation

The area-matched 20 ms waveform retains positive OFF anticipation and correct
forecast signs on all 150 ON and 150 OFF evaluation events after training.
The result supports a role for temporal shape/memory, while showing that extra
drive contributed to the previous forecast's strength. It is not a binary choice:
weights can compensate during learning, and coupled dynamics respond to currents.

## Exact area control

At dt=1/960, multiply excitatory impulses by 0.2698567. For exponential decay r,
discrete impulse area is `dt*A/(1-r)`, so this factor makes each unit-weight
20 ms impulse have the area of a 5 ms impulse. Inhibition remains unchanged.
Current arrivals and local eligibility increments receive the same factor;
both retain 20 ms excitatory decay. This is a fixed waveform factor, not weight
normalization. Topology, signs, measured delays, eight neural ticks/frame,
frame-horizon supervision, learning rate and other dynamics remain unchanged.

The [protocol](2026-09-21-area-matched-kinetics-protocol.md) fixes the comparison.
Training uses the same original magnitudes, 200 trials and 50 frozen evaluation
trajectories. The prior unscaled run is the reference.

| Learned metric | Unscaled 20 ms | Area-matched 20 ms |
|---|---:|---:|
| Overall MSE | 0.107658 | 0.108329 |
| ON MSE | 0.522855 | 0.532337 |
| OFF MSE | 0.707032 | 0.792185 |
| Signed ON anticipation | 0.290590 | 0.281664 |
| Signed OFF anticipation | 0.159964 | 0.110316 |
| Quiet false alarms | 37.08% | 25.69% |
| Correct ON/OFF signs | 150/150 each | 150/150 each |

Both polarity-specific error and anticipation gates still pass. Quiet false
alarms remain above 5%. Overall MSE improves 19.927% relative to this model's
own frozen initial state, narrowly missing the fixed 20% requirement; do not
round that into a pass. M1A remains unmet. This is one seed and one cropped circuit.

## Weight compensation and fixed-weight controls

The active L2 excitatory magnitude rises to 0.604755, compared with 0.231282 in
the unscaled model. Its effective arrival amplitude after the fixed factor is
0.163197, versus 0.231282 previously. L1 inhibitory magnitude is 0.650577 versus
0.790100. Learning partially compensates; retraining alone cannot isolate shape.

Freeze the same previously learned unscaled-model magnitudes, without any
compensation, and change only the kinetic condition:

| Frozen kinetics, identical weights | Signed OFF anticipation |
|---|---:|
| Uniform 5 ms predictive decay | -0.014790 |
| 20 ms excitation, area matched to 5 ms | +0.028026 |
| 20 ms excitation, original amplitude | +0.159964 |

The original class-dependent kinetics control gives effectively the same OFF
result as uniform 5 ms here. The latter removes class-dependent predictive decay
differences from the equal-area comparison. Neither control retrains weights.

Positive OFF sign survives equal area at fixed weights, but its strength is
much weaker. Longer dynamics help produce the correct sign and additional drive
amplifies it. These comparisons allow source spikes to respond to recurrent
currents; they establish a coupled-circuit result, not isolated filter causality.

Frozen paired precursor/blank probes remain stimulus-dependent. Learned signed
stimulus-minus-blank effects are +0.252610 for ON and +0.093959 for OFF, versus
+0.026386 and +0.006720 initially. The learned absolute OFF forecast in those
probes is +0.125122; it is not solely a positive tonic offset.

## Where temporal selectivity fails

Group saved frozen predictions without changing scored quiet frames or the
0.1 false-alarm threshold. For the area-matched model:

| Quiet-frame group | False alarms / scored frames |
|---|---:|
| First frame after ON | 101/150 (67.3%) |
| Second frame after ON, before OFF | 94/150 (62.7%) |
| First frame after OFF | 1/150 (0.7%) |
| Second frame after OFF, before ON | 38/150 (25.3%) |
| First 1-3 blank frames after motion | 117/148 (79.1%) |
| Blank frames 4-6 after motion | 110/147 (74.8%) |
| Blank frames 7 onward | 6/926 (0.65%) |

There are also 9/32 false alarms in the initial pre-stimulus blank interval.
Together these groups account for all 476 false alarms among 1,853 quiet scored
frames. Randomized blank intervals and first-cycle cue exclusions are unchanged.
Negative ON forecasts linger, positive OFF forecasts appear one frame early,
and predictions persist transiently after motion stops. Longer blank intervals
are much quieter. This identifies temporal localization as the next target,
rather than a reason to globally weaken the successful OFF response.

A next bounded proposal should shape transient prediction around the intended
target time while retaining 20 ms memory and both signed pathways. A sign-
preserving rise-and-decay response at controlled area is a candidate to evaluate,
not an implemented or proven fix. Include post-motion blank transients as well
as periodic motion. Keep the existing gates; no extra training sweep or relaxed
threshold. No further architecture change was bundled into this control.

## Verification and artifacts

The added regression integrates an actual single-spike current and verifies
equal discrete area within 1e-6 relative error, plus tickwise current/eligibility
agreement. Full suite: 219 passed, 4 CUDA skips. Independent review found no
implementation blocker. Defaults remain unchanged. Experimental modes remain
unsupported by native engines and production checkpointing.

Training targets, blank schedules and initial weights match the unscaled run
exactly. Checked states remain finite, magnitudes bounded, and frozen weights
and source checkpoint unchanged. Main run 102.65 s; paired probes 13.88 s.

[Full results](2026-09-21-area-matched-kinetics-results.json),
[frozen same-weight controls](2026-09-21-area-matched-frozen-control.json),
[paired probes](2026-09-21-area-matched-precursor-comparison.json),
[quiet-phase counts](2026-09-21-area-matched-quiet-phases.json), and
[local artifact hashes](2026-09-21-area-matched-artifacts.json).
