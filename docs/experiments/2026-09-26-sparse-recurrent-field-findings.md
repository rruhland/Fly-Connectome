# Local recurrent learning predicts event sites but does not form a transferable motion state

**Status:** One-shot opt-in M1A comparison under the [registered protocol](2026-09-25-sparse-recurrent-field-protocol.md). Production M1A, measured connectome, and motor system are unchanged. Eight spatially shared motifs learned from signed event-camera patches with local reconstruction residuals. A 7×7 recurrent kernel learned from each postsynaptic unit's local next-event prediction error and presynaptic field activity. No labels, object regions, gradient tape, or backpropagation entered training or full-field inference. The 64 training episodes contained generic unlabeled motion only; held-out shapes, positions, speeds, and entity interactions were not used to choose parameters.

Aligned local credit **did learn a transferable next-event spatial prediction**. Top-32 recall counts target event sites among the 32 highest predicted sites, over the full 32×64 signed event field. A fixed self-persistence recurrent kernel, a time-shuffled-credit kernel with the same learned sensory motifs, and a repeated-event forecast were the controls.

| Held-out set | Aligned top-32 recall | Fixed | Shuffled | Repeat |
| --- | ---: | ---: | ---: | ---: |
| Generic held-out shapes, 16 scenes | **.360** | .282 | .194 | .000 |
| Position, 16 scenes | **.588** | .581 | .307 | .000 |
| Doubled speed, 16 scenes | **.480** | .311 | .298 | .000 |
| Changed shape, 16 scenes | **.542** | .445 | .281 | .000 |
| Separated movers, 8 scenes | **.312** | .293 | .114 | .000 |
| Crossing movers, 2 scenes | .354 | **.370** | .276 | .000 |

The aligned kernel beat fixed self-persistence in 14/16 generic held-out scenes and 16/16 doubled-speed scenes, and beat shuffled credit in 15/16 and 16/16 respectively. Top-8 recall on generic scenes was .171 aligned, .095 fixed, and .083 shuffled. Thus this is more than a single aggregate or an easy repeat-last-event effect. Its absolute quality remains limited: generic top-32 precision was .047, and the crossing set is only two scenes. These scores predict *where the next meaningful event occurs*, not its wall-clock timing or an independently maintained entity state.

The field's own latent activity failed the representation gate. A frozen nearest-centroid direction probe was fitted on 16 calibration examples and applied to post-inference field activity in offline object regions. These regions never entered the model; the probe is an oracle-localized capacity measure, not autonomous tracking.

| Probe set | Aligned correct | Fixed correct | Shuffled correct |
| --- | ---: | ---: | ---: |
| Calibration | 14/16 | 14/16 | 14/16 |
| Position | 14/16 | 14/16 | 14/16 |
| Doubled speed | **6/16** | 8/16 | 10/16 |
| Changed shape | 10/16 | 12/16 | 12/16 |
| Separated movers | 14/16 | 14/16 | 14/16 |
| Crossing movers | 3/4 | 4/4 | 4/4 |

About 30.6% of aligned field activity on doubled-speed cases fell outside the offline object neighborhoods, compared with 27.7% fixed. Even with oracle-localized scoring, speed transfer reached only 37.5%, well below the registered 80% gate. Aligned recurrence improves the *synaptic prediction map* while biasing sparse winners away from a stable, transferable motion code. This reproduces the broad architectural tension seen earlier: useful temporal information can live in local weights without becoming an actionable latent state.

**Decision:** Preserve the result as evidence that local, no-backprop predictive error can learn translation-shared event dynamics. Do **not** promote this sparse field as M1A's world-state or tune its winner count, kernel width, or gains against these held-out sets. The next architecture should explicitly separate fast local event prediction from a slower, persistent learned representation that binds multiple signed fragments over time. That calls for a distinct assembly-level state with locally learned support and competition, not another path score or a wider version of this same field. It must still demonstrate autonomous entity coherence and native-stream prediction before any production proposal. The measured T4/T5 responses can be tested as a complementary input after that representation works generically.

Reproduction: `python scripts/run_sparse_recurrent_field.py`; [raw results](2026-09-26-sparse-recurrent-field-results.json) include per-scene paired forecast counts, direction-probe summaries, activity measures, and runtime (about 34–80 seconds across two identical-output runs).
