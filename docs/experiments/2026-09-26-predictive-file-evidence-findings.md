# Independent local evidence improves native-cadence prediction, but does not yield reliable entity state

**Status:** Completed opt-in factorial architecture experiment under the [registered protocol](2026-09-26-predictive-file-evidence-protocol.md). The fast field and diagonal affinity were trained on the same 64 unlabeled generic streams and frozen; Pong and corrupted streams were zero-shot. The [raw results](2026-09-26-predictive-file-evidence-results.json) retain per-scene signed next-event scores and live-file/birth counts. No production change, Pong concept, oracle object label, reward, or backpropagation entered inference or training.

The two mechanisms were separated: **confirmation** keeps new files provisional until visual continuation or independent fast-field support; **conditioning** borrows fast-field future-event evidence only near live file support. The combined arm used both. A fifth arm time-shuffled the independent support while preserving the same confirmation and local forecast rules.

| Evaluation set | Fast only | Base file | Confirmation only | Conditioned forecast only | Combined | Shuffled support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic held-out top-32 | .360 | .331 | .397 | .480 | **.580** | .435 |
| Changed speed top-32 | .480 | .857 | .857 | **.943** | .873 | .857 |
| Changed shape top-32 | .542 | .843 | .843 | **.931** | .865 | .843 |
| Two separated entities top-32 | .312 | .789 | .789 | **.856** | .819 | .797 |
| Pong native cadence top-32 | .402 | .064 | .421 | .151 | **.657** | .432 |
| Pong coarser cadence top-32 | **.378** | .112 | .233 | .178 | .324 | .240 |
| Corrupted generic top-32 | .088 | .050 | .058 | .122 | **.128** | .081 |

Combined beat base and shuffled support on 15/16 generic cases and all four Pong seeds at **both** cadences. Its native-Pong top-8 recall was .345 versus .145 fast-only, .064 base file, and .319 shuffled support. This is a strong causal clue that a persistent local file can use independently learned future sensory evidence; rigid translation alone was the wrong Pong predictor. The combined arm also reached generic top-8 .416 versus .293 base. Yet at the coarser Pong cadence its top-32 .324 remained below fast-only .378, and the clean speed/shape scores were below the confirmation-free conditioned arm. The fixed 20-frame provisional lifetime and local support radius were predeclared, never tuned on transfer.

The **state** remained the obstacle. On native Pong, base carried three live files per active frame, while confirmation/combined averaged **1.89** and never more than two, despite creating three files per episode. A visual file is being suppressed; these aggregate counts do not identify which. On corrupted generic streams, confirmation reduced mean live files from **11.11** to **7.05**, still far above the intended few entities, and increased mean total births from **12.75** to **18.75** as unconfirmed fragments expired and reappeared. Shuffled support gave essentially the same noisy count, so independent prediction did not solve birth/survival. On clean controlled single-entity scenes, the confirmed arm was live on about 90% rather than 100% of scored frames; on separated scenes it averaged 1.8 rather than two files. A genuine stationary mark with a single onset may remain provisional indefinitely, an information limit for event-only confirmation.

**Decision:** Do not promote this combined arm. It substantially improves native Pong prediction but fails the protocol's coarser-cadence and noisy autonomous-state criteria, and it suppresses genuine visual hypotheses. Further sweeps of provisional lifetime, radius, confidence threshold, or rank blend would be narrow tuning of the same mechanism. The next architecture should represent **graded predictive evidence and uncertainty** over time, rather than turning one confirmation into a permanent file and recycling unsupported fragments. It should learn the event-time transition distribution and distinguish a persistent but currently quiet visual entity from a transient noise explanation. This is a change in state representation, not another gate parameter. Any production M1A revision still requires user review.

Reproduce with `.venv/Scripts/python scripts/run_predictive_file_evidence.py` (about 43 seconds on this machine).
