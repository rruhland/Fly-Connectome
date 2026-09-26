# Learned observation transfers across noise, not yet across motion regimes

**Decision:** Keep the locally learned observer opt-in. It generalized to two unseen corruption levels, but the frozen observer-plus-transition system did **not** pass the [registered transfer criterion](2026-09-26-observation-model-transfer-protocol.md) across motion regimes or native Pong. No production M1A change is justified.

The 64-stream generic training, local rules, weights, and eight-frame intensity cadence were frozen from the successful [observation experiment](2026-09-26-learned-observation-model-findings.md). The transfer audit used changed-position/speed/shape, separated and crossing moving visual patterns, raw Pong camera streams at two cadences, and held-out generic scenes with lighter/heavier corruption rates. No transfer input affected training or calibration.

| Zero-shot stream | Learned observer top-32 / top-8 | Raw field top-32 / top-8 | Shuffled observer-credit top-32 |
|---|---:|---:|---:|
| Changed position | .173 / .151 | .178 / .143 | .186 |
| Changed speed | .089 / .073 | .100 / .079 | .118 |
| Changed shape | .344 / .233 | .337 / .242 | .296 |
| Two separated entities | .170 / .116 | .172 / .106 | .169 |
| Crossing entities | .172 / .130 | .193 / .120 | .302 |
| Pong, native cadence | .347 / .072 | **.483** / .049 | .113 |
| Pong, four-step cadence | .314 / .049 | .330 / .052 | .100 |
| Held-out lighter event noise | **.658 / .486** | .333 / .145 | .276 |
| Held-out known event noise | **.635 / .447** | .231 / .104 | .282 |
| Held-out heavier event noise | **.541 / .367** | .146 / .080 | .281 |

The observation rule appears reusable as an event-reliability front end under this family of noise: lighter and heavier corruption were absent from training, and it retained strong next-event gains. On clean changed-position, speed, separated, and crossing tests, its true-event credibility remained about .96. Their low forecast scores were already present in the raw distributed field, so suppressing noise cannot fix its motion-dynamics transfer. The transition field learned from fractional-pixel generic trajectories and is weak on 1–2-pixel-per-frame step motion. This tempo mismatch is an inference from training/evaluation conditions, not yet proven as the sole cause. On native Pong, true-event credibility fell to .734 and top-32 dropped by .136 despite a small top-8 gain, exposing additional visual-domain shift in the observer.

The next broad experiment should train the same local observer and transition rules on **unlabeled multi-cadence generic visual experience**, with held-out positions/shapes/scene seeds and no Pong data. Compare against the original frozen model on these transfer groups and the original noise test. This tests whether experience coverage, rather than another mechanism, repairs motion transfer. If it does not, revisit the distributed temporal representation instead of tuning observation thresholds or local decays.

Reproduce with `.venv/Scripts/python scripts/run_observation_model_transfer.py` (about 37 seconds here); [raw results](2026-09-26-observation-model-transfer-results.json) include each case, event credibility, and state/forecast mass.
