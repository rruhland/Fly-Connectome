# Local predictive-error calibration improves both coverage and focus

**Status:** Completed opt-in two-stream readout experiment under the [registered protocol](2026-09-26-local-rank-reliability-protocol.md). The [raw results](2026-09-26-local-rank-reliability-results.json) retain per-case hit counts, learned source/polarity/rank weights, credit counts, and output activity. Fast-field and sensory-affinity learning still used only 64 unlabeled generic streams; the readout saw 2,971 next-meaningful-event pairs from those streams. Its direct local update estimated whether each predicted pixel was observed next. No backpropagation, object identity/count, game semantics, reward, or oracle region entered training or inference.

Aligned local calibration beat both the time-shuffled-credit readout and the fixed equal-rank combination on the transfer sets. The most important result is that it improved **both** broad top-32 coverage and sharply ranked top-8 events, instead of trading one for the other.

| Evaluation set | Equal fusion top-32 / top-8 | Aligned local readout | Shuffled-credit readout |
| --- | ---: | ---: | ---: |
| Generic held-out (16) | .523 / .278 | **.586 / .353** | .329 / .080 |
| Changed position (16) | .862 / .627 | **.910 / .820** | .632 / .226 |
| Changed speed (16) | .913 / .398 | **.935 / .598** | .622 / .302 |
| Changed shape (16) | .899 / .495 | **.924 / .698** | .629 / .265 |
| Two separated entities (8) | .829 / .317 | **.859 / .536** | .533 / .276 |
| Crossing entities (2) | .807 / .417 | **.812 / .583** | .557 / .229 |

The trained fast-only field scored .360 top-32 on generic held-out and the online entity files .331; the learned readout's .586 is not a single-stream improvement disguised as fusion. Aligned beat equal fusion on top-32 in 15/16 generic, 14/16 speed, 16/16 shape, and 8/8 separated cases; it beat shuffled credit in all those cases. Across generic and changed-speed events it emitted at least 64 positive candidate sites per event (means 67.7 and 67.1), so the top-32 gain did not arise from being scored with fewer than 32 candidates. The same maximum-32 ranking metric was used for every arm.

The learned weights are source-, polarity-, and rank-dependent rather than one hand-selected global blend. High-rank entity predictions often acquired empirical next-event reliability around .17–.27, while many fast-field ranks were lower; shuffled credit drove both toward near-zero coincidence rates. This explains why the readout can retain useful fast-field coverage but restore entity-like top-eight focus. It also shows genuine predictive credit matters beyond adding another stream.

**Decision:** Preserve this as a promising **opt-in** M1A direction, not a production promotion. A global rank operation is an engineering form of sparse competition; whether a local recurrent/inhibitory circuit can realize it remains untested. The experiment did not change the entity files' four identity-switch frames in clean crossings or their 13/13 wrong counts under static event noise. It also has not shown transfer to a full visual scene such as Pong without retraining. The next decisive audit should freeze all generic training, run the unchanged readout on raw Pong camera streams and a small clutter/occlusion stress set, and score full-frame future events together with autonomous file state. If it transfers, present a concrete production M1A design for user review; if not, address the failed representation conditions before promotion. Do not tune rank weights on Pong.

Reproduce with `.venv/Scripts/python scripts/run_local_rank_reliability.py` (about 44 seconds on this machine).
