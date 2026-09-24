# Closed-loop learned forecasts retain a short horizon, but drift

The frozen correlation-input latent circuit was rolled forward without new camera evidence or weight updates. Its predicted primitive activity was thresholded at 0.5 and fed back as the next local input; the fixed correlation control received its thresholded predicted events in the same way. Each rollout restored the true online state before the next observed frame. All horizons used the same t=3..10 anchor frames.

| Scene family | Learner +1 | Learner +2 | Learner +4 | Fixed +1 | Fixed +2 | Fixed +4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unseen single pattern | 0.658 | 0.549 | 0.539 | 0.962 | 0.962 | 0.949 |
| Two independent movers | 0.671 | 0.530 | 0.521 | 0.968 | 0.968 | 0.968 |
| Crossing movers | 0.540 | 0.406 | 0.338 | 0.748 | 0.668 | 0.495 |
| Speed changes | 0.597 | 0.440 | 0.257 | 0.851 | 0.720 | 0.458 |

The learned model sustains some future visual activity for four frames on clean constant translation, far above the random-code control (approximately zero beyond one frame). Its predictions drift under interactions and changes in motion. The fixed low-level sensory control remains better at every measured horizon; in simple translation it is exceptionally strong even as an autoregressive predictor. This diagnostic does not support promoting the present code as a full world model.

The next opt-in candidate should give latent units a persistent, locally corrected state rather than relying on event-site selection plus open-loop predicted input. Compare it with the current learner and fixed control on crossings, missing evidence, noise, and +1/+2/+4 horizons. A useful improvement must preserve the established one-step transfer while narrowing the interaction and missing-evidence gaps. No production code changed.
