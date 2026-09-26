# Local visual-context memory: promising clean M1A.5 proof, not acceptance

**Decision:** Preserve the opt-in context-conditioned tracklet state as a promising architectural proof. Do not promote it to production M1A or claim M1A.5 complete. It passes the [predeclared clean context gate](../plans/2026-09-26-m1a5-history-state-benchmark.md), but its multi-entity identity, reacquisition confidence, noisy operation, and sensor-teacher transfer are not ready. No backpropagation, hidden-state teaching, object labels, or Pong concepts enter its learning rule.

The benchmark places a small visible mark near a moving visual pattern. Across experience, the mark's relative side predicts a transverse path change while the pattern is hidden. The mark and moving pattern are generic binary visual geometry; the held-out cases change position, shape, speed, direction, and polarity. A fixed tracklet scaffold supplies causal local proposals. On later visible reacquisition, a locally stored mark-side trace credits a single learned transverse displacement ratio. The scaffold and relative-side extractor are engineered low-level primitives, so this experiment does **not** establish discovery of a general latent representation from arbitrary visual input.

| Event-only held-out split | Aligned local association | Shuffled association | Frozen association | Strong fixed kinematic controls |
|---|---:|---:|---:|---:|
| Context change, top-8 / top-32 (16 cases) | **16 / 16** | 0 / 0 | 0 / 0 | 0 / 0 |
| Constant motion, top-32 (16 cases) | **16** | 16 | 16 | 16 |
| Two moving entities, top-64 (16 hidden entities) | **16** | 16 | 16 | 16 |

Sparse-frame inference produced the same counts on these clean scenes. The aligned model learned cue effects about −0.402 and +0.402 from 16 observed reacquisitions per cue. Balanced cue/outcome shuffling drove them near zero (−0.020 and +0.020), which supports a real experience-dependent association rather than a fixed kinematic forecast.

Matched continue/disappear histories have identical input and output through the hidden decision frame, as they must. Without an absence update, the model assigned mean strength .946 to the continued target and .906 to the counterfactual vanished target two frames after expected reveal. A local visual-support update reduced vanished strength to .176 and put all 16 vanished cases below .5, while continued strength fell to .705 and only **10/16** continued cases stayed above .5. The update preserves clean hidden localization but reacquisition confidence remains inadequate and is not probabilistically calibrated.

The [observer-teacher audit](2026-09-26-event-only-observer-teacher-findings.md) found that corrupted context scenes collapse even with the privileged observer and improve only modestly with sparse frames. Thus the clean 16/16 result is a proof that local context can influence a hidden visual state, not a robust generic vision/world-model result. The current two-dimensional relation is also specialized to horizontal approaches and above/below context. Rotation, multiple unrelated cues, changed hidden dynamics, longer gaps and environmental transfer remain untested.

Reproduce with `.venv/Scripts/python scripts/run_local_context_memory.py`; exact results for both inference sensor arms, frozen and shuffled controls are in [the result JSON](2026-09-26-local-context-memory-results.json). Production M1A is unchanged.
