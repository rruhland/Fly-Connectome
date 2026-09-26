# Absolute visual evidence helps noise, but hard reset destabilizes clean entity state

**Status:** Completed opt-in sensory architecture comparison under the [registered protocol](2026-09-26-absolute-refresh-protocol.md). The [raw results](2026-09-26-absolute-refresh-results.json) include per-scene signed next-event scores, active-frame count errors, and birth counts. The model received generic camera contrast derived from the uncorrupted observed frame only on refresh frames; this is an **additional sensory channel**, not an event-only achievement or an oracle entity mask. No production change, Pong information, label, reward, or backpropagation entered the experiment.

| Evaluation | Event only | Absolute refresh every 8 frames | Every-frame ceiling |
| --- | ---: | ---: | ---: |
| Clean generic top-32 next-event recall | .215 | .305 | **.347** |
| Clean generic wrong file-count fraction | **.023** | .248 | .294 |
| Clean generic mean births/episode | **2.5** | 4.0 | 4.5 |
| Corrupted generic top-32 recall | .050 | .253 | **.314** |
| Corrupted generic mean live files | 11.11 | 3.59 | **2.61** |
| Corrupted generic wrong count fraction | .966 | .621 | **.370** |
| Corrupted generic mean births/episode | **12.75** | 30.0 | 20.5 |
| Stationary noisy wrong count fraction | 1.000 | .625 | **.312** |

The independent absolute observation carries information the noisy event stream lacks: even a refresh every eight camera frames cut mean false live-file load by about two thirds and restored much of the clean next-event forecast. But **replacing** contrast and deleting a file when its center neighborhood had no matching signed contrast was not a valid state update. On clean held-out streams, count error rose from 2.3% to 24.8% at eight-frame refresh and 29.4% with refresh every frame; births increased as real files were removed and recreated. Under corrupted input, eight-frame refresh created 30 files per episode on average despite reducing the number simultaneously live. The stationary noisy every-frame arm averaged only .69 live files for one true shape and had zero next-event recall. A later observed true entity can be lost if a hard local check and imperfect event association disagree.

**Decision:** The event-only input is insufficient for robust stationary/noisy scene memory in this test, but this hard-reset implementation fails clean-state preservation and should not be promoted. A larger architecture should assimilate event and occasional absolute contrast as **graded local evidence** in a persistent recurrent state: support should raise confidence, absence should lower it gradually and permit reacquisition without changing identity, and local prediction error should learn the timing and reliability of each source. Compare that state with event-only and hard-reset controls under clean, corrupted, and multi-cadence visual transfer. Do not sweep refresh period or deletion radius; this experiment already identifies the structural flaw. The original two-stream clean learning remains opt-in evidence, not a production M1A claim.

Reproduce with `.venv/Scripts/python scripts/run_absolute_refresh.py` (about six seconds on this machine).
