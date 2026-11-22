# Reusing Pong motion evidence for later SO-101 returns

A motion coordinate fitted to native Pong return feedback produced equivalent later-stroke learning curves on the archived physical SO-101 opportunities. Across the 4-, 8- and 16-trial checkpoints, Pong changed the strict legal-return rate by −0.05 percentage points relative to the same target learner with a cold metric. A paired bootstrap over held target donors gave a 95% interval from −0.16 to +0.07 percentage points. The difference from the permuted-label Pong fit was −0.12 percentage points, with interval −0.80 to +0.60. [Design](source-metric-design.md) · [Complete curves and selections](source-metric-curves.json.gz)

## Native Pong feedback

The fit paired prior archived RGB mover and actor tracks with native Pong return events by launch seed and the digest of every executed action. Each feature vector used observations ending eight camera frames before the last sample preceding the scored crossing. A ridge fit learned how to combine relative lateral offset, lateral slope, projected offset and time to the actor plane. The held source set gave the true-label fit AUC 1.00 and Brier error .034, versus .484 and .232 for a fit to a fixed permutation of the same labels. There were 49 eligible approaches from eight training launch seeds and 37 approaches from nine disjoint held seeds. Causal input to the predictor was pre-contact optical motion. The native public return scorer supplied the source training label. [Source examples, permutation and frozen fit](source-metric-checkpoint.json.gz)

## One target reward at a time

Physical branch data supply the counterfactual outcomes at each practice cue. All methods get a fixed target-trained opening head and the same physically branch-tested stroke menu and lateral rule. The later learner samples one feasible stroke per practice cue and updates from its strict return outcome. The source fit scales the evidence given to target contexts that differ along the fitted Pong motion coordinate. It neither selects commands on its own nor contributes direct arm rewards. The true- and permuted-label coordinates use the same scale and weighting. [Learner](../../scripts/experimental/tt23_source_metric.py) · [Physical training branches](../tt22/branch-motor.md)

| Later physical outcomes observed | 0 | 4 | 8 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Pong metric | 28.9% | 28.2% | 26.8% | 23.9% | 24.7% | 26.9% |
| Cold metric | 28.9% | 28.2% | 26.9% | 24.0% | 24.4% | 26.8% |
| Permuted-label Pong metric | 28.9% | 28.1% | 26.9% | 24.2% | 24.4% | 26.9% |

Each holdout omitted an entire donor point, including its opening branch. The target evaluation contains 45 later cues from 17 distinct donors. The 32 paired reruns varied cue order and sampling randomness and reused those same physical outcomes. They supply no new independent target starts. Through these budgets all learners remained below their shared opening-derived zero-trial accuracy. Pong changed 21 of the 1,440 matched predictions relative to cold at 16 trials. It changed 237 relative to permuted source labels. [Fold-by-fold outcomes](source-metric-curves.json.gz)

## Fresh full points

After 16 eligible target practice outcomes under a single fixed order, each learner played six fresh matched physical points against the existing cooperative five-axis rival. All three produced a legal opening and one rival reply on all six starts. The resulting alternating return counts were `[2, 2, 2, 2, 2, 2]` in each condition. The observed optical prefixes and physical arm commands matched until each first later cue. All 18 full two-arm command tapes reproduced the exact event and outcome records. When another policy independently issued the same complete command sequence on captured RGB observations, its separate physical replay reused that observation record. [Fresh physical receipt](source-metric-live.json.gz) · [Runner](../../scripts/experimental/tt23_source_metric_live.py)

Run `python -m scripts.experimental.tt23_source_metric` from the repository root to regenerate the source and archive curves. Run `python -m scripts.experimental.tt23_source_metric_live` for the fresh-point cohort, with `MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` in the project environment.
