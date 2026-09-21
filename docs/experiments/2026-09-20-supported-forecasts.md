# M1A forecasts on anatomically supported events

The 10,000-frame checkpoint still fails the zero-event baseline. Its reduction
in total prediction error is almost entirely suppression of quiet-period false
alarms, not improved event prediction. Anatomical sign limitations explain only
part of the problem: supported OFF events remain worse than zero as well.

## Frozen comparison

Initial and 10,000-frame checkpoints were evaluated for 500 frames on development
seeds 1101/1102, with plasticity disabled. Graph and configuration hashes match.
Source checkpoint checksums are recorded in the accompanying JSON files and
verified unchanged after each run. Final seeds 1201-1204 remain unused.

`scripts/audit_supported_forecasts.py` records the preceding-tick forecast at
each camera frame. A separate camera reproduces the same rendered events; an
integration test compares the audit with ordinary evaluation using nonzero
predictive currents. Metrics accumulate stored float32 predictions in float64.
Structural support includes measured edges whose current magnitude is zero;
it does not imply that their sources fire or carry useful information.

| Target group | Samples | Initial MSE | 10,000-frame MSE | Zero MSE |
| --- | ---: | ---: | ---: | ---: |
| Supported ON | 1,163 | 0.999012244 | 0.999294016 | 1 |
| Unsupported ON | 111 | 1.002330956 | 1.000257938 | 1 |
| Supported OFF | 587 | 1.000907437 | 1.000234097 | 1 |
| Unsupported OFF | 651 | 1.002887517 | 1.001072836 | 1 |
| Quiet | 2,674,488 | 0.000222977079 | 0.000015492050 | 0 |
| All | 2,677,000 | 0.001161700387 | 0.000953857566 | 0.000938363840 |

ON means negative contrast injection; OFF means positive contrast injection.
Supported ON forecasts slightly beat zero at initialization and after training,
but training makes this conditional score worse. Supported OFF forecasts improve
toward zero while remaining worse than it. A correct-sign fraction alone is
insufficient: trained supported OFF forecasts have the correct sign on 59.6% of
events, but their amplitudes still produce negative mean target-prediction product.

The sample-weighted quiet contribution falls by 0.000207290333, accounting for
99.734% of the total reduction of 0.000207842821. All-sample prediction power
falls from 0.000222876428 to 0.000015484452. Mean target-prediction product remains
negative (-2.30059e-7 initially; -4.63690e-9 after training). The exact identity
`MSE = zero_MSE + prediction_power - 2 * mean(target * prediction)` separates
suppression from useful alignment without fitting a predictor.

Population-conditioned supported ON/OFF MSE after training is respectively
1.000029474 / 0.999966508 for L1, 0.998056916 / 1.000805422 for L2, and effectively
1 / 1 for L3. Thus aggregate ON improvement over zero is primarily an L2 effect;
these selected conditional scores are descriptive, not milestone acceptance.

## Timing and shuffle controls

A fixed frame permutation (seed 421) compares strict forecasts with misaligned
forecasts on the same 500 target frames. Trained supported ON/OFF MSE is
0.999400229 / 1.001056688 after shuffling, versus 0.999294016 / 1.000234097 without
shuffling. This small descriptive alignment is insufficient to beat zero overall;
one permutation is not a significance test.

The lag sweep uses a common set of 497 interior frames at every offset, so its
scores must not be directly compared with the all-frame shuffle scores. Offset
zero is the preceding-tick forecast. Positive offsets use post-event activity
and cannot establish anticipation. Offsets are neural ticks (1.04167 ms each).

| Offset | Initial supported OFF MSE | Trained supported OFF MSE |
| --- | ---: | ---: |
| -16 | 0.999758550 | 1.000511724 |
| -8 | 0.999559280 | 1.000895356 |
| -4 | 0.999441890 | 1.000465940 |
| -2 | 1.000437730 | 1.000345965 |
| -1 | 1.000305310 | 1.000280614 |
| 0 | 1.000916810 | 1.000227044 |
| +1 | 1.000401280 | 1.000424067 |
| +2 | 0.998548260 | 1.000321361 |
| +4 | 0.998116720 | 1.000830168 |
| +8 | 0.997273440 | 1.000461829 |
| +16 | 0.997898940 | 1.000789409 |

Initially some post-event OFF responses beat zero, especially at +8 ticks.
After training, no tested offset beats zero for supported OFF events. A simple
forecast timing shift within this range therefore does not recover the missing
performance. This does not exclude longer-timescale signals or establish which
mechanism caused their suppression.

## Reproduction

Run the following for each of `initial` and `10000`:

```powershell
.venv/Scripts/python scripts/audit_supported_forecasts.py checkpoints/event-v1-combined-rate-initial.pt --steps 500 --seeds 1101,1102 --output docs/experiments/2026-09-20-supported-forecasts-initial.json
.venv/Scripts/python scripts/audit_supported_forecasts.py checkpoints/event-v1-combined-rate-10000.pt --steps 500 --seeds 1101,1102 --output docs/experiments/2026-09-20-supported-forecasts-10000.json
```

## Local eligibility signals

The existing `scripts/audit_local_signals.py` was run on the same trained
checkpoint, development seeds and 500-frame stream. Its signed per-edge traces
reconstruct actual predictive current with maximum error 2.98e-8. It measures
3,992 ticks, excluding the first frame, and changes no weights.

| Checkpoint | Population | Active traces | Positive covariance | Positive mean local update |
| --- | --- | ---: | ---: | ---: |
| Initial | L1 | 3,123 | 121 | 60 |
| 10,000 frames | L1 | 2,363 | 71 | 121 |
| Initial | L2 | 4,742 | 323 | 41 |
| 10,000 frames | L2 | 4,142 | 387 | 103 |
| Initial | L3 | 755 | 32 | 2 |
| 10,000 frames | L3 | 74 | 1 | 2 |

Active means trace power above 1e-12; an edge can have an active source trace and
zero weight. Counts use physical traces including warm-up, not exact initial
training eligibility. Mean updates omit the learning rate, weight clipping and
homeostatic scaling; they indicate local pressure, not realized weight changes.
Top correlations in the JSON are selected descriptive examples, not independent
evidence of learning. Tiny positive updates near numerical zero are included in
the counts and must not be interpreted as substantial potentiation.

The trained L1/L2/L3 sums of target-times-trace are respectively -0.00135753,
-0.00089481 and 3.70725e-8, versus prediction-times-trace sums of 0.00426086,
1.08891536 and 0.00042268. Their differences favor net depression in all three
populations. These edge sums are not error scores and are not comparable across
populations as effect sizes. Together with lost active traces, they explain why
simply observing a lower total error cannot establish effective local learning.

## Homeostasis bound and saved weights

A read-only comparison of the two checkpoint payloads found identical metadata.
There are 80,000 training ticks (83.3333 seconds). One boolean spike per tick
bounds the exponentially filtered firing rate by `1 / dt = 960 Hz`, starting
from zero. Even ignoring refractoriness and the slow rate ramp, homeostasis alone
therefore retains at least

```text
exp(-homeostasis_rate * seconds * max(1/dt - maximum_rate, 0))
= exp(-0.00001 * 83.3333333 * (960 - 50))
= 0.468446521
```

This real-arithmetic bound ignores rounding; losses below 1% of initial strength
are far beyond plausible rounding effects. It excludes homeostasis alone, not
interactions with local plasticity or homeostasis's indirect effects on activity.
The comparison counts initially positive predictive edges ending at each named
population, excluding any initial weight above the configured maximum of 10.
For reproduction, use `state.network.pathways == 1`, `state.network.post`,
`metadata.retina.cell_types`, and the ratio of saved `state.network.magnitudes`
in the trained and initial checkpoint. Raw counts, settings and checksums are in
`2026-09-20-homeostasis-bound.json`.

| Population | Edges | Trained weight zero | Below 1% of initial | Median retained ratio |
| --- | ---: | ---: | ---: | ---: |
| L1 | 3,310 | 1,046 | 2,219 | 0.0000214884 |
| L2 | 9,764 | 1,735 | 2,281 | 0.942852974 |
| L3 | 4,483 | 2 | 51 | 1 |

L1 feedback undergoes substantial local depression beyond what homeostasis alone
could cause. L3 has a different signature: most incoming weights remain unchanged,
while active incoming traces collapse. Its missing forecasts therefore require
examining source activity, not merely the weights terminating on L3. These
observations narrow the diagnosis but do not identify a sufficient remedy.

No training extension, optimization experiment, parameter calibration, biological
change, or M1B advancement is justified by these results alone. Next, attribute
the lost L3 traces to presynaptic cell classes and inspect their incoming
feedforward/predictive activity in matched frozen initial/trained runs. Check
whether event-linked activity fails to arrive, fails to cause spikes, or loses
feedback strength. Keep topology, signs and all dynamics fixed during diagnosis;
any model revision requires a separate evidence-based proposal.
