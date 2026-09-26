# Learned local observation model restores prediction under event noise

**Decision:** Preserve this as a promising **opt-in** M1A architecture candidate, not a production promotion. It passes the [registered](2026-09-26-learned-observation-model-protocol.md) clean/corrupted prediction criteria and the shuffled-credit control. It still needs zero-shot transfer across visual domains, camera cadences, and unseen corruption conditions, plus an explicit check of the stationary-change failure before presenting a production change for user review.

The observation model learned continuous credibility for each arriving signed event from its 5×5 neighborhood and its own local fast/slow filtered-event and contrast history. It trained on 64 unlabeled generic scenes with synthetically corrupted event input. Its credit target was **signed change in separately rendered intensity images** from the same training scene, exposed by the generator; no held-out clean event, object mask/count, direction, game concept, reward, or backpropagation entered training. At held-out inference, intensity was available only every eight frames and the clean future event was withheld for scoring. The downstream distributed transition field was trained once on clean generic streams and frozen in every arm.

| Held-out stream/arm | Top-32 future recall | Top-8 | Mean false contrast mass | Mean non-target forecast mass |
|---|---:|---:|---:|---:|
| Clean, raw input | .681 | **.522** | .00 | 4.87 |
| Clean, learned observer + aligned intensity | **.685** | .519 | 1.65 | 4.30 |
| Corrupted, raw input | .231 | .104 | 56.51 | 102.69 |
| Corrupted, learned + aligned | **.635** | **.447** | **7.23** | **8.93** |
| Corrupted, learned, no intensity | .514 | .239 | 22.96 | — |
| Corrupted, learned + shuffled intensity | .521 | .363 | 14.10 | — |
| Corrupted, shuffled observation credit | .282 | .156 | 4.23 | 5.02 |

On corrupted events, the aligned observer passed 0.886 average strength at true raw-event sites and 0.056 at false sites. Its top-32 recall improved on **all four** held-out generic scenes (per-scene .772/.446/.775/.531 versus raw .280/.084/.330/.201), while clean recall remained stable. The aligned gain over shuffled observation credit shows the local paired-visual update is essential. Aligned intensity added a further .121 top-32 over the same learned observer without intensity, and the unrelated-scene intensity arm was lower. The learned arm emitted about 699 positive forecast sites per scored corrupted event, far above the 32-site evaluation budget, so the top-32 gain is not an artifact of a smaller candidate set.

This is a meaningful architecture result: the bottleneck was not only future-transition learning but **which sensory events entered recurrent state**. A learned local observation model separated true/false input well enough that the previously noise-amplifying distributed predictor became useful. The experiment is still narrow in domain: training and corrupted evaluation used the same specified dropout/false-event rates on different scene seeds, and the training teacher used full rendered intensity changes even though inference saw intensity only every eight frames. That is a paired-sensor self-supervision assumption to validate for M1A, not a generic event-only learning result. The stationary noisy case still had zero future-event hits despite low false contrast mass, so onset/offset anticipation remains open.

Next, freeze this trained system and run one broad zero-shot audit on changed position, speed, shape, multiple moving entities, raw Pong at both camera cadences, and corruption rates not seen during training. Include raw-field and shuffled-credit controls, full-frame future metrics, and state/credibility summaries. If it transfers meaningfully, prepare a concrete production M1A revision for user review; if not, identify the failed transfer condition and reconsider the learned observation objective without threshold sweeps.

Reproduce with `.venv/Scripts/python scripts/run_local_observation_model.py` (about 26 seconds here); exact per-case measurements are in [the result JSON](2026-09-26-learned-observation-model-results.json).
