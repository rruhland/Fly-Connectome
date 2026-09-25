# Online local credit degrades the available trace ranking across longer gaps

The [registered threshold-free audit](2026-09-25-online-trace-ranking-protocol.md) repeated the exact one-pass 288-episode online continuous-trace training, saved its experimental model to `runs/2026-09-25-online-blank-trace-model.pt`, and measured event-versus-reachable-quiet output scores before applying any decision threshold. The saved checkpoint reloads successfully. No model parameter, training target, threshold, or production code changed.

| Gap duration | Training event/quiet AUC | Held-out event/quiet AUC | Held-out event mean | Held-out quiet mean |
| --- | ---: | ---: | ---: | ---: |
| Early two-frame | 0.792 | **0.814** | 0.042 | 0.019 |
| Familiar three-frame | 0.740 | 0.757 | 0.023 | 0.017 |
| Late four-frame | 0.603 | **0.614** | **0.009** | **0.014** |

All three groups still score zero gap-exit F1 at the fixed 0.5 threshold. The online output carries some useful ranking at short gaps, but its ranking degrades with duration and reverses in mean by four frames. A scalar gain or lower threshold cannot repair the four-frame ordering. The prior frozen offline readout fitted to exit plus quiet samples achieved 0.907 AUC over its pooled held-out set, so the code can support more separation than the online local update extracts; that pooled offline number is not a per-duration matched comparison.

The next opt-in architecture experiment should change **local credit**, not trace amplitude or the evaluation threshold: keep the learned sensory/trace code and 17×17 field fixed, but give event-confirming and quiet-suppressing updates separately normalized local eligibility, with a matched shuffled-credit control and full timing/false-alarm evaluation. If that fails to restore held-out ranking at all three durations, move to a learned recurrent transition that propagates latent state spatially during blank input. Production M1A remains unchanged and is not ready for promotion.
