# Persistent-observation forecast: generic learning, failed native transfer

**Status:** Opt-in M1A experiment. No production model or graph changed. No backpropagation, privileged scene labels, Pong-specific input, or motor learning was used.

The [proposal](2026-09-25-observation-state-and-horizon-proposal.md) called for a learned recurrent hidden state receiving a persistent local event observation and the existing learned motion code. We trained that state with the existing local recurrent update. A separate shared 5×5 local emission synapse received delayed signed event error either four camera frames later or at the next observed event. Both heads used only source activity, their own forecast, and the arriving local camera event. Frozen, repeated-event, and temporally shuffled-credit controls were evaluated at the same target times. A four-channel variant added decaying ON/OFF event recency to the persistent two-channel observation. All heads were trained on generic moving patterns and evaluated zero-shot on four native 120 Hz Pong camera streams.

| Training / held-out domain | Four-frame F1 | Next-event F1 | Next-event shuffled F1 |
| --- | ---: | ---: | ---: |
| Mixed-size generic / unseen single shapes | .011 | .312 | .027 |
| Mixed-size generic / independent objects | .013 | .335 | .052 |
| Mixed-size generic / crossing objects | .076 | .369 | .032 |
| Mixed-size generic / native Pong | .000 | .0004 | .000 |
| Sparse generic dots / unseen cardinal dots | .054 | .176 | .000 |
| Sparse generic dots / unseen diagonal dots | .000 | .122 | .000 |
| Sparse generic dots / native Pong | .023 | .005 | .000 |
| Sparse dots plus local recency / unseen cardinal dots | .000 | .276 | .000 |
| Sparse dots plus local recency / independent objects | .000 | .271 | .000 |
| Sparse dots plus local recency / crossing objects | .001 | .334 | .000 |
| Sparse dots plus local recency / native Pong | .000 | .001 | .000 |

The generic next-event result is real local learning relative to shuffled credit, but it is not a solved long-horizon prediction task. In the mixed-size generic evaluation, the next event occurred one frame after the origin in every scored case. For native Pong the mean lead was 1.60 frames. The four-frame head was generally worse than repeated-event persistence or silent. Event polarity was retained in all metrics: a wrong-sign forecast counts as both a false positive and a miss. Native next-event precision was only .27% after sparse training (51 true positives, 19,119 false positives); generic next-event F1 alone hides this failure. Quiet-target false alarms are reported for the four-frame arm; next-event targets are by definition active, so that arm does not test quiet-time calibration.

The domain gap is substantial. Training episodes had 8.61 event pixels per active frame; native Pong had 2.12. Matching that sparsity with dot-only training improved native four-frame F1 to just .023 and reduced broad generic next-event transfer. A frozen ranking audit found that the sparse-trained next-event head's top eight signed locations captured 66.9% of held-out dot target events but only 1.5% of native targets. Ignoring ON/OFF sign raised native top-eight spatial recall only to 3.5%. This is a location/ranking failure, not merely a bad 0.5 output threshold or an OFF-sign convention error. Adding local recency improved some generic next-event scores while leaving native transfer near zero and eliminating four-frame output.

**Decision:** Do not promote persistent observation, these delayed heads, or the recency variant into production. The next high-value opt-in experiment should replace the frozen correlation encoder with a locally learned event-stream representation trained on diverse generic fractional-speed, diagonal, and multi-entity streams at native camera cadence. Compare its local patch distribution and held-out event ranking with the existing encoder before attaching another forecast head. Preserve generic inputs and local/no-backprop learning, and require both a nontrivial four-frame forecast and native transfer before production promotion. If the representation still fails, revisit the recurrent credit objective instead of sweeping more thresholds or decay constants.

Reproduction: `python scripts/run_observation_horizon_forecast.py`, `python scripts/run_sparse_event_transfer.py`, `python scripts/run_event_forecast_ranking.py`, and `python scripts/run_recent_observation_transfer.py`. Raw results are in the correspondingly named JSON files in this directory.
