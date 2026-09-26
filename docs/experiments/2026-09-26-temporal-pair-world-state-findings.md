# Local temporal transitions help some forecasts, but not transferable world-state prediction

**Status:** One-shot opt-in test under the [registered protocol](2026-09-26-temporal-pair-world-state-protocol.md). The same learned paired-unit identity and signed current-error decoder were shared across aligned, time-shuffled, and frozen transition arms. The fast field, 64 unlabeled generic training scenes, and all held-out scenes were unchanged. Production M1A and the measured graph were not changed.

Full-frame next-meaningful-event top-32 recall:

| Held-out set | Fast only | Static paired | Aligned transition | Shuffled transition | Frozen transition |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic scenes | .360 | **.433** | .415 | .350 | .345 |
| Doubled speed | **.480** | .362 | .360 | .393 | .276 |
| Changed shape | **.542** | .473 | .504 | .487 | .440 |
| Separated movers | **.312** | .293 | .286 | .280 | .211 |
| Crossing movers (2 scenes) | .354 | **.396** | .385 | .370 | .312 |

The aligned temporal rule improved over its shuffled control on 14/16 generic held-out scenes and 10/16 changed-shape scenes, and improved the changed-shape forecast over the static paired readout on 10/16 scenes. It did **not** beat the shuffled control on speed (6/16 paired scenes), the static paired readout on speed (4/16), or fast-only on any of the 16 speed scenes. Its top-8 recall on speed was .143 versus .151 fast-only, and on shape .187 versus .187 fast-only. Signed local corrections and learned temporal transitions therefore carry some useful aligned information, but do not rescue speed transfer or separated-entity prediction.

The spatial direction probe and autonomous activity metrics are identical across the three temporal arms and static pair by construction: all share the same frozen paired identity, and the transition affects only forecasting. The common offline probe remained 14/16 on speed and shape and 16/16 for separated movers, but the slow state still had 3.44 occupied components per frame for two separated movers and 30.4% of significant activity outside their offline neighborhoods. The contrast between decodable direction and fragmented autonomous state remains.

**Decision:** Reject this temporal-pair world-state implementation as the M1A forecast architecture. It misses the pre-registered speed and shape preservation gates and the aligned-versus-shuffled speed gate. Do not tune recurrence, decoder scale, or footprint on these evaluation sets. Together with the paired-identity result, the evidence says local delayed pairing can create a transferable spatial code, but sparse coarse winners and a single local temporal kernel do not yet bind coherent moving entities or predict their dynamics across speed changes. The next architectural test should address **continuity and binding of moving evidence in the state itself**—for example, a generic locally plastic field that links co-moving active patches through time and predicts their spatial continuation—before another decoder experiment. Compare it against the fast-only predictor, frozen/shuffled temporal controls, and autonomous component/coherence measures; keep object coordinates strictly offline.

Reproduction: `python scripts/run_temporal_pair_world_state.py`; [raw results](2026-09-26-temporal-pair-world-state-results.json) retain all per-scene counts, recall/precision, probes, and activity measures. Runtime was about 39 seconds.
