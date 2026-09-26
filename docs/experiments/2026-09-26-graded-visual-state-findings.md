# Graded visual-state Stage 1: clean continuity, noisy overbirth

**Decision:** Reject this soft entity-file implementation for M1A. Do not tune its confidence, support, or refresh thresholds on these held-out streams. The registered state-reliability criterion failed, so timing, Pong transfer, and measured T4/T5 ablation of this candidate are deferred. Production M1A is unchanged.

The [registered protocol](2026-09-26-graded-visual-state-protocol.md) compared an opt-in recurrent set of soft signed spatial hypotheses with the prior event-only files and hard intensity refresh. It used the original 64 unlabeled generic streams only to fit the low-level diagonal sensory affinity. The four held-out clean/corrupted generic scenes and stationary noisy scene matched the prior [refresh experiment](2026-09-26-absolute-refresh-findings.md). All forecasts were scored against the same clean next camera event. The aligned and shuffled arms received intensity every eight frames; the shuffled arm received an unrelated held-out scene.

| Held-out stream and arm | Wrong live-count fraction | Mean live files | Births/episode | Top-32 next-event recall |
|---|---:|---:|---:|---:|
| Clean, prior event-only files | .023 | 2.56 | 2.5 | .215 |
| Clean, prior hard refresh / 8 | .248 | 2.33 | 4.0 | .305 |
| Clean, graded aligned / 8 | **.000** | 2.58 | 2.5 | .466 |
| Clean, graded event-only | .252 | 2.30 | 2.5 | **.549** |
| Clean, graded shuffled / 8 | .944 | 4.64 | 5.0 | .350 |
| Corrupted, prior event-only files | .966 | 11.11 | 12.75 | .050 |
| Corrupted, prior hard refresh / 8 | .621 | 3.59 | 30.0 | .253 |
| Corrupted, graded aligned / 8 | **.865** | 6.11 | 28.5 | **.331** |
| Corrupted, graded event-only | .975 | 13.57 | 29.0 | .237 |
| Corrupted, graded shuffled / 8 | .987 | 7.59 | 29.75 | .211 |
| Stationary noisy, graded aligned / 8 | .375 | .75 | 12.0 | .500 |

The aligned arm's clean count and shuffled-control degradation show that spatially aligned intensity is informative. The signed soft support and local continuation also produce useful forecast signal. But forecast gain does not rescue an unreliable autonomous state: the preregistered corrupted wrong-count target was below .621, and this model reached .865. It promoted 24.25 of its 28.5 average born hypotheses per corrupted episode to live status. A provisional false component can be matched by subsequent noisy event evidence before the next intensity sample; the intensity correction then reduces confidence gradually, but more false components emerge. That is the same entity-list overbirth problem expressed with graded weights. Clean forecast also fell from .549 event-only to .466 with intensity, suggesting the current absolute update interrupts some useful event-time continuation.

This does **not** establish that graded sensory evidence is useless. It rejects a variable list of entity hypotheses whose own event matches can confirm them. The next architectural comparison should remove file birth/identity entirely and test a distributed recurrent predictive visual field, with spatially local state and plasticity and no privileged object slots. It must show stable next-event prediction under the same corrupted and clean streams, and should be evaluated for emergent segmentation only after prediction works. That is a different representation family, not another file-lifetime or gate variant.

Reproduce with `.venv/Scripts/python scripts/run_graded_visual_state.py`; exact measurements are in [the result JSON](2026-09-26-graded-visual-state-results.json).
