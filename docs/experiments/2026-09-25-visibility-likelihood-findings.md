# Local visibility likelihood does not yet calibrate uncertain evidence

**Status:** Two opt-in heads tested; neither qualifies for signed-event emission or production promotion. The locally learned [hidden transition](2026-09-25-local-hidden-transition-findings.md) remains a separate, successful result. Protocols: [site head](2026-09-25-local-visibility-protocol.md) and [bounded field revision](2026-09-25-field-visibility-revision-protocol.md). Results: [site](2026-09-25-local-visibility-results.json), [feature audit](2026-09-25-visibility-feature-audit-results.json), [field](2026-09-25-field-visibility-results.json).

Both heads froze the learned transition and trained a single shared probability unit with delayed local camera-event error, without backpropagation, object labels, gap labels, or Pong semantics. They saw 480 training episodes from the four training shapes and were evaluated on unseen shapes and positions. Every one of the 144 two-versus-three and 144 three-versus-four pairs produced **identical probabilities from identical visual prefixes**, as required by the causal test.

| Held-out metric (lower is better) | Site head | Site fixed hazard | Field head | Field fixed hazard |
| --- | ---: | ---: | ---: | ---: |
| Local site/field Brier | **0.121** | 0.192 | **0.154** | 0.236 |
| Global any-event Brier, all frames | **0.196** | 0.227 | **0.183** | 0.237 |
| Global quiet-frame Brier | 0.363 | 0.377 | 0.397 | 0.326 |
| Matched 50/50 pair Brier (ideal 0.25) | 0.319 | **0.301** | 0.289 | **0.263** |
| Matched pair mean event probability (ideal 0.50) | 0.259 | 0.538 | 0.320 | 0.556 |

The site head learns useful local evidence likelihood but underpredicts ambiguous reappearance. A narrow [credit audit](2026-09-25-visibility-feature-audit-results.json) explains the sign of its elapsed-evidence weight: the fraction of active candidate sites with a nearby next event is 0.933 during ordinary visible motion, 0.761 at two-gap exits, and 0.638 at three-gap exits. The event may return somewhere in the moving field while many exact candidate sites are negative. At three-gap exits, thresholded candidates cover only 54% of event pixels and are absent in 42/144 cases. Its local target therefore trains against a different question from whether *any* event will return.

The field revision reads continuous subthreshold state in overlapping 16×16 local receptive fields and raises event-pixel coverage to 95%. It improves aggregate Brier but produces too many quiet-frame false alarms and still predicts only 0.320 probability on intrinsically 50/50 matched histories. A local field label plus a maximum over fields is not automatically a calibrated probability for the whole visual field; the dominant visible frames can mask poor gap timing in aggregate scores. The shuffled-time controls were worse overall, but that does not rescue the failed calibration and quiet criteria.

Stop adjusting this visibility head or its scalar threshold. The evidence supports a learned motion/world-state transition, while the observation process remains unresolved. The next decisive experiment should test the hidden state itself on independent moving patterns, crossings, speed changes, and noise. If it transfers, M1A can expose a learned latent forecast with explicit uncertainty while observation likelihood is redesigned at the correct causal scale; it should not claim precise visible-event forecasts. If hidden state fails those scenes, work on the motion representation before returning to visibility. Production remains unchanged.
