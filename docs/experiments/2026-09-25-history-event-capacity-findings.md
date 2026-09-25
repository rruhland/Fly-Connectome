# Local raw polarity unlocks future-event capacity in learned history units

The [registered frozen-code diagnostic](2026-09-25-history-event-capacity-protocol.md) recreated the same unlabeled learned and random history dictionaries. An evaluation-only sparse 5×5 linear readout fit t=11 ON/OFF events from t=10 source activity on 64 generic interrupted training-shape cases. It then tested 144 held-out interruptions across three unseen shapes, four directions, two speeds, two contrasts, and three positions. The readout weights were never used to train or change the visual code; they are not an M1A learning rule. The run took 304 seconds.

| Exact future-event F1 | Learned unit only | Random unit only | Learned unit × current raw polarity | Random unit × polarity | Learned × polarity, history reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Readout training | 0.000 | 0.099 | **0.959** | 0.632 | 0.000 |
| Held-out 144 cases | 0.000 | 0.067 | **0.923** | 0.544 | 0.000 |
| Matched opposite-motion pair | 0.000 | 0.000 | **0.643** | 0.353 | 0.000 |

The learned polarity-conditioned readout reached **0.962 up**, **0.915 down**, **0.933 left**, and **0.886 right** held-out F1 at the registered 0.5 threshold. All future events were spatially reachable from active sources. The same learned history units without locally observed raw polarity failed even on readout training examples. The random dictionary also benefited from polarity but trailed the learned code by 0.379 held-out F1. Resetting the trace removed all source activity at reappearance and collapsed prediction to zero.

This is strong evidence that the learned history code and current sensory sign **jointly** contain transferable local future-event information. The previous online emitter indexed weights only by history-unit identity, forcing raw ON and OFF contexts to share a kernel; a direction-readable unit by itself is insufficient for exact sign and pixel prediction. The offline least-squares readout is an upper bound and must not be promoted or described as online learning. Next test a **polarity-conditioned local emission bank** with the same delayed local error update, training episodes, threshold, and random/reset controls. If that succeeds across all directions without false alarms or clean-motion regression, prepare a concrete production architecture revision for user approval. No production M1A code has changed.
