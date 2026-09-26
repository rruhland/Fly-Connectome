# Future sensory-error credit helps some forecasts but does not train a useful slow state

**Status:** One-shot opt-in M1A comparison under the [registered protocol](2026-09-26-sensory-error-assembly-credit-protocol.md). The frozen fast field, slow sensory motifs, topology, decay, sparse competition, and 64 unlabeled generic training streams were identical across slow arms. Only recurrent local credit differed: future fast sensory error, next slow winner error, frozen self-persistence, or time-shuffled future sensory error. Every slow forecast added its local predicted fast state to the same frozen fast prediction. No direction, object, speed, reward, Pong semantics, backpropagation, or gradient tape entered training or full-frame inference. Production M1A and the measured graph remain unchanged.

Future sensory-error credit changed the forecast in a useful direction relative to the *old slow objective*, but failed the larger test. Next-meaningful-event top-32 recall over the full signed event frame was:

| Held-out set | Fast only | Sensory-error slow | Next-winner slow | Frozen slow | Shuffled sensory credit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic shapes, 16 scenes | .360 | .390 | **.422** | .359 | .344 |
| Doubled speed, 16 scenes | **.480** | .392 | .331 | .296 | .412 |
| Changed shape, 16 scenes | **.542** | .514 | .471 | .475 | .507 |
| Separated movers, 8 scenes | **.312** | .269 | .258 | .216 | .297 |
| Crossing movers, 2 scenes | .354 | **.375** | .349 | .312 | .370 |

The sensory-error arm beat next-winner credit on changed-shape forecasts in **16/16 paired cases** and on doubled speed in **12/16**. Yet it beat shuffled sensory credit in only 8/16 shape cases and **4/16 speed cases**, and did not beat the fast-only forecast in any of the 16 speed cases. Its small crossing lead comes from just two scenes. Aligned temporal credit therefore does not explain a robust, transferable forecast improvement. The rule suppressed recurrent weight mass to 4.51 versus 8.00 for the frozen self-persistence kernel, consistent with substantial negative local credit rather than a no-op.

The fixed 3×3 **offline** spatial direction probe also did not show a learned slow-state advantage:

| Probe set | Fast only | Sensory-error slow | Next-winner slow | Frozen slow | Shuffled sensory credit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Doubled speed | 12/16 | 14/16 | 14/16 | **16/16** | 14/16 |
| Changed shape | **10/16** | 8/16 | 6/16 | 8/16 | 8/16 |
| Separated movers | 14/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| Crossing movers | 2/4 | 3/4 | 3/4 | 2/4 | 2/4 |

About 44.7% of significant sensory-error slow activity on doubled-speed cases fell outside offline object neighborhoods. Two separated movers yielded 2.78 occupied slow components per active frame, with .78 more components than movers on average. Those synthetic regions and counts were used only after inference for evaluation. These measures and the matched controls give no evidence of a single coherent, autonomous assembly per moving entity.

**Decision:** Reject future-sensory-error credit applied only to the *fixed, reconstructively learned slow motifs* as an M1A architecture. It is a real local update and improves some slow-arm forecasts, but cannot meet the registered forecast-preservation, shape-transfer, and control-beating gates. Do not tune its error projection, gain, decay, or winner count on these held-out scenes. The result narrows the architectural issue: the slow unit's *identity and competition* must learn from predictive responsibility too; changing recurrent credit after the sparse motifs have already been assigned is insufficient. A further opt-in design should co-learn slow motif specialization and competition from future explanatory power, with matched frozen/shuffled controls and autonomous entity evaluation, rather than add another readout to these fixed units.

Reproduction: `python scripts/run_sensory_error_assembly_credit.py`; [raw results](2026-09-26-sensory-error-assembly-credit-results.json) retain per-scene counts, spatial-probe summaries, component measures, controls, and elapsed time (about 117 seconds).
