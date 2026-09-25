# Frozen blank-frame trace contains a local forecast, but decays across longer gaps

The [registered capacity diagnostic](2026-09-25-blank-trace-capacity-protocol.md) fit an **evaluation-only** translation-shared 17×17 linear readout from the saved 24-channel motion trace on the last blank frame to the next signed event map. It used 64 training-shape three-frame gaps, then tested 144 unseen-shape/multi-position three-frame gaps and 48 unseen-shape center-position cases for each changed gap. No learned M1A weight was changed; the offline decoder is not part of the online model.

| Gap-exit future-event F1 | Cases | Offline trace readout | Target reach within 5×5 / 9×9 / 17×17 |
| --- | ---: | ---: | ---: |
| Training three-frame gaps | 64 | 0.768 | 0.244 / 0.585 / 1.000 |
| Unseen three-frame gaps | 144 | **0.516** | 0.309 / 0.599 / **1.000** |
| Unseen early two-frame gaps | 48 | 0.262 | 0.480 / 0.809 / 1.000 |
| Unseen late four-frame gaps | 48 | **0.091** | 0.171 / 0.401 / **0.809** |

The held-out three-frame score passes the registered 0.20 F1 and 0.80 reachability capacity gate. It transfers across all four directions: up 0.562, down 0.626, left 0.365, right 0.472. Setting the trace to zero while retaining the fitted decoder yields zero F1. Thus the local decaying trace **does contain forecastable spatial information before reappearance**; the current online pathway misses it because it waits for an observed raw onset to activate its readout.

The same fixed decoder transfers poorly to changed timing. Early two-frame gaps remain geometrically reachable in the 17×17 field, but F1 falls to 0.262. Late four-frame gaps fall to 0.091, with 19.1% of target events now outside that field. The local trace also shrinks each blank frame. One separately registered evaluation-only control should remove *local amplitude decay* by normalizing trace channels at each site, then refit the same decoder and test the same held-outs. If this improves changed-gap transfer, a locally normalized continuously active readout is worth an online-plasticity experiment. If not, the primary next architecture hypothesis is a learned recurrent state transition that spatially propagates during absent evidence, with explicit drift/quiet controls. Do not promote this offline decoder or a fixed geometry-based motion model.
