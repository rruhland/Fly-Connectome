# Causal corroboration before tracklet formation: bounded screen failed

**Decision:** Do not keep local temporal corroboration as the M1A.5 noise solution. It modestly improves noisy localization over the previous binary tracklet input, but misses the [predeclared acceptance screen](../plans/2026-09-26-m1a5-history-state-benchmark.md) by a wide margin. The clean local-context result remains real; this experiment says its event-to-entity state is too brittle under ordinary sensor corruption. Do not tune another event-strength threshold or coherence decay on this benchmark. Production M1A is unchanged.

The opt-in candidate weighted arriving signed events by causal nearby/recent same-polarity corroboration before fixed generic tracklet association. The world-state still learned the context-to-hidden-displacement ratio from later visible reacquisition, with no hidden masks, labels or backpropagation. The event-only observer was itself trained using only local event agreement. A shuffled cue/outcome learner and an untrained association used the same front end.

| Event-only-trained observer, top-32 hidden hits | Aligned | Shuffled | Frozen | Registered aligned gate |
|---|---:|---:|---:|---:|
| Clean contextual change (16 cases) | **16** | 0 | 0 | ≥14 |
| Familiar event noise (16 cases) | **4** | 1 | 2 | ≥12 |
| Heavy event noise (16 cases) | **2** | 2 | 2 | ≥8 |
| Constant motion (16 cases) | **16** | 16 | 16 | ≥14 |
| Two entities, top-64 (16 hidden entities) | **16** | 16 | 16 | ≥14 |

With actual sparse intensity frames at inference, familiar-noise context improves to **7/16** for the aligned event-only-trained arm, still below gate. With the privileged observer instead, familiar-noise event-only is 2/16 and sparse-frame inference 5/16. The positive context association remains specific: the aligned model learns effects about −.402/+.402; balanced shuffling drives them to −.020/+.020. The coherence weighting therefore does not erase learning, but also does not solve false proposals/identity under noise.

For matched continue/disappear pairs, predictions remain numerically identical before reveal. Two frames after expected reveal, all 16 vanished cases fall below .5 at the counterfactual target, but only **10/16** continuing cases exceed .5 at the true target; the registered continued threshold is 14/16. Confidence is not calibrated as a probability. Rotated motion and unrelated contextual dynamics are untested.

The high-value next change is a different state representation, not another threshold: confidence-weighted spatial evidence with local competing entity hypotheses, delayed birth/death, and explicit uncertainty during occlusion, coupled to the demonstrated local context credit. It must preserve weak genuine events without letting isolated false ones create durable identities. Test one bounded implementation against the same clean/noisy/reacquisition/multi-entity gates, then decide whether it is worth a production proposal. Exact counts and all control arms are in [the result JSON](2026-09-26-corroborated-context-memory-results.json); reproduce with `.venv/Scripts/python scripts/run_corroborated_context_memory.py`.
