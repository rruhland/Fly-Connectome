# L3 feedback loss localizes to C2 recurrent inputs

The upstream investigation found a specific cause of the lost L3 input activity:
C2 neurons lose recurrent excitation during training. Restoring only existing
predictive weights entering C2 recovers their firing and their outgoing L3 traces.
This is an activity rescue, not a learning rescue: sensory prediction error worsens
in the restored hybrid network. M1A remains unmet.

## Matched frozen source tracing

The initial and 10,000-frame checkpoints were evaluated on development seeds
1101/1102 for 500 scripted frames. Source and physical-trace statistics exclude
the first frame, matching the preceding local-signal audit: 3,992 neural ticks
per environment, after ordinary warmup. Physical traces retain warmup history.
Graph and configuration hashes match. No training or checkpoint writes occurred.

`scripts/audit_feedback_sources.py` selects every measured predictive edge into
L3, including zero-weight edges, and records its presynaptic source's spikes,
feedforward/recurrent/sensory current, outgoing weight sum and post-reset voltage
margin. Per-source means average environments and ticks; class means average
sources. Spike counts are totals, not rates. Trace energy samples the preceding
tick; currents and spikes describe the current tick. Their aggregates do not
establish precise event-by-event causal timing.

| L3 feedback source | Initial active edges | Trained active edges |
| --- | ---: | ---: |
| C2 | 675 | 1 |
| Lawf1 | 7 | 0 |
| All other sources | 73 | 73 |
| Total | 755 | 74 |

C2 accounts for 674 of 681 lost active traces (98.97%). There are 780 C2 source
neurons supplying 788 measured predictive edges into L3. The exact source roster
is identical across comparisons; it is not selected by firing or performance.

| C2 source statistic | Initial | Trained |
| --- | ---: | ---: |
| Spiking sources | 669 | 1 |
| Spikes | 2,350 | 4 |
| Mean feedforward current | -0.009590843 | -0.009565391 |
| Mean recurrent current | 0.002239868 | -0.000000075855 |
| Mean absolute recurrent current | 0.002241520 | 0.000000416649 |
| Total outgoing weight into L3 | 49.5250001 | 45.7402625 |

Feedforward inhibition changes by less than 0.3%; the positive recurrent current
almost disappears. Most C2-to-L3 weight mass survives. The loss is therefore not
well explained by removal of those outgoing weights or disappearance of mean
feedforward input. Mean currents alone cannot exclude changes in their temporal
structure, so a selective intervention follows below.

C2 resting current is exactly 1.0, the baseline threshold. Without positive input,
a leaky membrane starting below this equilibrium need not cross threshold;
inhibition and adaptation make crossing harder. Silent C2 sources approach within
roughly 5.36e-7 of threshold in the recorded post-reset state. This is consistent
with the configured dynamics, not evidence of a threshold-comparison bug.
The observation suggests sensitivity to lost recurrent excitation; it does not
justify changing resting currents to force firing.

## Selective restoration test

After source attribution, a separate diagnostic was preregistered in the ledger:
start from the trained checkpoint and replace only the 3,091 existing predictive
magnitudes entering C2 with their initial values, before warmup. All other
magnitudes, signs, edges, delays, dynamics, sensor mapping and seeds stay fixed.
Compatibility is checked before restoration. Both source checkpoint checksums
are checked afterward. The hybrid is never saved as a trained checkpoint.

| Statistic | Trained | C2-input restoration |
| --- | ---: | ---: |
| Spiking C2 sources | 1 | 671 |
| C2 spikes | 4 | 3,070 |
| Active C2-to-L3 traces | 1 | 677 |
| All active L3 input traces | 74 | 750 |
| C2 mean recurrent current | -0.000000075855 | 0.002236324 |
| All-frame sensory event MSE | 0.000953858 | 0.000967914 |
| Zero-event MSE | 0.000938364 | 0.000938364 |

Restoring this group of incoming weights is sufficient to recover C2 activity
in the trained network, including downstream recurrent effects. It does not
identify an individually necessary synapse or prove the restored activity carries
useful anticipation. Indeed, the hybrid increases total sensory prediction error.
Blindly restoring these weights or raising C2's resting current is not supported
as an effective learning remedy.

Source/trace statistics exclude frame one; event MSE includes all 500 frames.
The trained MSE comes from the preceding matched supported-forecast audit; its
float64 accumulation differs slightly from the standard Trainer accumulator used
for the hybrid, far below the observed error increase. Neither result is an
acceptance evaluation. Final seeds 1201-1204 remain unused.

## What this resolves and leaves open

The source investigation is meaningful: it identifies a specific upstream weight
group whose restoration reverses the L3 activity collapse. It also shows that
reversing that collapse alone is insufficient for useful learning. The earlier
homeostasis bound already excludes homeostasis alone as an explanation for large
weight losses; this intervention localizes an additional functional consequence.

The remaining issue is whether the current local forecast objective supports
retaining useful recurrent activity, rather than merely suppressing unconfirmed
current. These experiments do not establish that a different prediction horizon
would fix it. They also do not rule it out: the prior lag sweep covered only about
17 ms, not 30, 50 or 100 ms, and not a next-meaningful-event objective.

The user's longer-horizon suggestion is retained for a subsequent frozen
diagnosis before any target/dynamics revision. Longer fixed-horizon forecasts
must use only earlier activity, identical target windows and explicit quiet/event
baselines. A next-meaningful-event diagnostic must state its event definition and
bounded horizon, disclose hindsight-based sample selection, and compare against
polarity/persistence controls; event-conditioned accuracy alone must not conceal
false alarms or count as online acceptance. No such model change is implemented
by this investigation. Optimizations remain closed.

## Reproduction and verification

```powershell
.venv/Scripts/python scripts/audit_feedback_sources.py checkpoints/event-v1-combined-rate-initial.pt --output docs/experiments/2026-09-20-feedback-sources-initial.json
.venv/Scripts/python scripts/audit_feedback_sources.py checkpoints/event-v1-combined-rate-10000.pt --output docs/experiments/2026-09-20-feedback-sources-10000.json
.venv/Scripts/python scripts/audit_feedback_sources.py checkpoints/event-v1-combined-rate-10000.pt --restore-c2-from checkpoints/event-v1-combined-rate-initial.pt --output docs/experiments/2026-09-20-feedback-sources-c2-restored.json
```

Each JSON includes checkpoint and graph hashes, configuration identity and the
complete source roster. Baseline bundles were produced before adding the hybrid
metric fields; rerunning adds those metadata/metric fields without changing source
statistics. The hybrid measurement-window description was corrected after its run;
no measured values were changed.

Maximum predictive-current reconstruction errors were 9.31e-9, 3.73e-9 and
7.45e-9 respectively. Fixtures verify exact tensor-state preservation by the
instrumentation and selective restoration of only existing incoming predictive
weights. Fresh review found a measurement-window wording issue, corrected to
distinguish source statistics from event MSE. Full suite: 187 passed, 4 CUDA skips.
