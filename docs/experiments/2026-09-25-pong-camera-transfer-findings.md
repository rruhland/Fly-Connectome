# Native Pong camera timing starves the adjacent-frame visual code

**Status:** Zero-shot opt-in evaluation under the [camera-rate protocol](2026-09-25-pong-camera-transfer-protocol.md), with [machine-readable results](2026-09-25-pong-camera-transfer-results.json). The model received only event-camera maps. No Pong identity, ball/paddle position, collision, reward, or simulator state entered inference or learning; production remains unchanged.

The locally learned visual code and hidden transition were trained only on 96 generic moving-pattern episodes, then frozen. Four held-out Pong seeds each supplied 120 32×64 event-camera frames at native physics-frame sampling and at a four-physics-step sampling control. After warmup, each condition scored 464 frames.

| Sampling | Frames with camera events | Frames with adjacent-frame correlation/latent code | Learned next-latent F1 on active frames | Persistence F1 |
| --- | ---: | ---: | ---: | ---: |
| Native, one physics step | 287/464 (61.9%) | **33/464 (7.1%)** | **0.000** | 0.000 |
| Four physics steps | 464/464 (100%) | **456/464 (98.3%)** | **0.405** | 0.310 |

At the native rate, event pixels are present but usually too far apart in time for the immediate-previous-frame coincidence input. The visual code is silent on most frames, so the recurrent transition receives too little sensory state; camera-quiet frames average only 0.25 active hidden sites. The four-step control restores a dense enough local code to expose some zero-shot prediction, although its 0.095 F1 gain over persistence falls just below the registered 0.10 transfer threshold.

This does not show that the learned transition is adequate at native timing. It isolates an upstream temporal-support failure first. The next bounded experiment should replace only the fixed adjacent-frame sensory coincidence with a short decaying local event trace, then evaluate native-rate code coverage and next-latent F1 with the *same frozen* sensory dictionary and transition. If that wakes the code but does not improve forecasting, retrain generic local dynamics on variable-cadence experience before discussing production promotion. Do not solve this by feeding privileged Pong state or hardcoding objects.
