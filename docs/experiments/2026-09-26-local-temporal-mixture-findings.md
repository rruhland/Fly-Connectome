# Local pathway selection collapses; equal fusion shows complementary dynamics

**Decision:** Reject this local temporal selector for M1A. It failed the [registered joint-retention rule](2026-09-26-local-temporal-mixture-protocol.md), and a fixed equal mixture also failed native Pong and top-eight focus. Do not sweep selector learning rate, context radius, or pathway ratio on these held-out streams. Production M1A is unchanged.

The original slow/general transition field and the 256-episode multi-cadence field were retrained and frozen. Their full-frame held-out hit counts exactly reproduced the prior [transfer](2026-09-26-observation-model-transfer-findings.md) and [multi-cadence](2026-09-26-multicadence-experience-findings.md) runs. Both received the same frozen locally learned event observer. A spatially local selector read nearby continuous state from both and changed only its own synapses according to the difference in each pathway's next-event local prediction error on unlabeled multi-cadence training scenes. An equal mixture and time-shuffled selector-credit arm controlled for mere access to two pathways. No gradient flowed through either frozen pathway.

| Full-frame top-32 next-event recall | Slow pathway | Diverse pathway | Equal mix | Aligned local selector | Shuffled selector |
|---|---:|---:|---:|---:|---:|
| Changed position | .173 | .406 | **.447** | .419 | .173 |
| Changed speed | .089 | .418 | .400 | **.421** | .089 |
| Changed shape | .344 | .459 | **.515** | .478 | .346 |
| Two separated entities | .170 | .341 | **.375** | .342 | .171 |
| Crossing entities | .172 | .271 | **.318** | .271 | .172 |
| Pong, native cadence | **.347** | .134 | .249 | .194 | **.347** |
| Pong, four-step cadence | .314 | .276 | .330 | **.333** | .314 |
| Clean generic | .685 | .583 | **.729** | .586 | .685 |
| Corrupted generic | .635 | .503 | **.656** | .503 | .634 |

Equal fusion demonstrates genuine complementary predictive information: it improved clean and noisy generic, shape, position, separated, and crossing top-32 above either single pathway. Its native Pong top-32 was only .249, however, and native Pong top-8 fell to **.015** (versus .072 slow and .091 diverse). A broad rank gain therefore cannot be mistaken for a reliable sharply timed forecast.

The trained local selector did not learn context-dependent choice. Its ON weights were all strongly negative for the slow pathway (range -1.96 to -1.79, with negative bias), and evaluation nearly matched the diverse pathway across groups. Shuffled temporal credit pushed it toward the slow pathway. It met the 90%-of-diverse changed-motion condition but retained only .586/.503/.194 on clean generic/corrupted generic/native Pong, below the required .617/.572/.312. Equal fusion met the generic and changed-motion conditions but also failed native Pong. This test cannot establish whether the current local context lacks information or the specific selector credit cannot extract it; additional gate variants would now be narrow testing.

The architecture implication is larger than a blend coefficient. The two learned transitions predict different useful futures, but choosing one from this field's local traces did not retain both across scene/cadence change. The next broad candidate should replace the linear event-trace transition with a genuinely recurrent visual state whose locally learned temporal dynamics can preserve several regimes without two manually separated training pathways. It should keep the successful learned event observer as a frozen sensory control and be evaluated against equal fusion, native Pong at both cadences, changed motion, and noisy generic streams. If that work is not pursued, the current evidence supports keeping only the observation model as a research component, not declaring M1A complete.

Training the two experts took 73 seconds and selector credit 34 seconds; the full experiment took 114 seconds. Reproduce with `.venv/Scripts/python scripts/run_local_temporal_mixture.py`; [raw results](2026-09-26-local-temporal-mixture-results.json) include per-case scores and learned selector weights.
