# Fourfold visual exposure does not rescue prediction

The same opt-in diverse-shape circuit received 384 instead of 96 unlabeled training episodes per learning stage. Its first 96 episodes matched the earlier run; additional episodes crossed missing position and contrast combinations. The learned code, local prediction rule, offline decoder method, and 48 unseen-shape evaluations were unchanged. The expanded run took **374 seconds** on this host.

| Measure | 96 episodes | 384 episodes |
| --- | ---: | ---: |
| Held-out transition-code direction probe | 30/48 | 34/48 |
| Training local event F1 | 0.238 | 0.226 |
| Held-out local event F1 | 0.223 | 0.221 |
| Training offline learned-code event F1 | 0.476 | 0.432 |
| Held-out offline learned-code event F1 | 0.324 | 0.314 |
| Held-out offline random-code event F1 | 0.160 | 0.189 |

The preregistered 0.50 held-out decoder, 0.30 slow-speed decoder, and 0.35 local forecast gates all failed. Slow-speed decoder F1 rose from 0.202 to 0.261, but fast-speed F1 fell from 0.392 to 0.348. At the exploratory threshold 0.3, held-out decoder F1 changed from 0.440 to 0.431. All future events remained locally reachable. The fixed low-level event-correlation control remained at 0.971 F1.

Extra experience improved the post-hoc direction probe without improving next-event prediction. The current event-site transition code and its local predictive update should not be promoted to M1A. The next opt-in architecture experiment should change what **local temporal state** each latent unit can access, while preserving unlabeled learning, local credit, no backpropagation, and generic evaluation. It should compare learned and frozen/random state at both speeds; otherwise the same fixed sensory primitive will keep outperforming it. No production code changed.
