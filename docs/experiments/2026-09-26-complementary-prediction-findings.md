# Fast sensory and persistent entity forecasts are strongly complementary

**Status:** Confirmed opt-in architectural comparison under the [registered protocol](2026-09-26-complementary-prediction-protocol.md). The original generic discovery result was re-evaluated alongside the complete frozen transfer sets. No production M1A architecture, game-specific feature, object label, reward, or backpropagation was added. The [raw results](2026-09-26-complementary-prediction-results.json) retain per-case scores and overlap counts.

The fast locally learned field and the online entity files predict different useful pixels. Equal reciprocal-rank fusion of their full-frame signed forecasts substantially improved top-32 next-event recall at the **same 32-site budget**. Replacing the entity forecast with one from an unrelated event removed the gain, so it is not simply extra prediction volume. The entity file used the previously learned diagonal sensory affinity; no fusion weights were trained or tuned on the held-out cases.

| Evaluation set | Fast only | Entity files | Equal fusion | Shuffled entity fusion |
| --- | ---: | ---: | ---: | ---: |
| Generic held-out (16) | .360 | .331 | **.523** | .251 |
| Calibration (16) | .588 | .789 | **.862** | .318 |
| Changed position (16) | .588 | .789 | **.862** | .318 |
| Changed speed (16) | .480 | .857 | **.913** | .255 |
| Changed shape (16) | .542 | .843 | **.899** | .353 |
| Two separated entities (8) | .312 | .789 | **.829** | .239 |
| Crossing entities (2) | .354 | **.823** | .807 | .406 |

Fusion beat the better individual source on 9/16 generic, all 16/16 changed-speed, all 16/16 changed-shape, all 8/8 separated, and 1/2 crossing cases. Its generic top-32 precision rose to .069, versus .047 fast-only and .043 entity-only. The two sources' top-32 sets overlapped by only **1.72 pixels per scored event** on average in generic scenes. That low overlap explains why their predictions can cover more future evidence together.

The important counterexample is **temporal/spatial precision at smaller budgets**. Generic top-8 recall was .278 for fusion versus .293 for entity-only. Changed-speed top-8 fell from .571 entity-only to .398 fused; changed-shape from .674 to .495; separated entities from .526 to .317. Equal ranks give many fast-field sites too much priority among the strongest eight. Fusion also does not repair the prior static-noisy false births or four clean-crossing identity-switch frames. Reciprocal ranks are an engineering probe of information complementarity, not a biological/local readout to promote unchanged.

**Decision:** Preserve the two streams and test one shared, locally plastic arbitration rule on unlabeled generic training streams. A sparse, competitive rank can be treated as a low-level activation normalization; each source/rank channel should learn its own next-event reliability directly from local predictive error, with time-shuffled credit as a control. Evaluate whether that learned readout preserves the broad top-32 gains while restoring entity-like top-8 focus across speed, shape, and multiple entities. If it cannot, do not tune a global blend on transfer cases; reconsider the readout representation. Production promotion remains premature until noisy false hypotheses and crossing identity are addressed.

Reproduce with `.venv/Scripts/python scripts/run_complementary_prediction.py` (about 20 seconds on this machine).
