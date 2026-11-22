# Learning when to hold from native Pong duration

The earlier 264 native scenes supplied zero cases where lateral position and observed motion had opposite signs. Randomly delaying the choice added 13 such cases among 38 detected training scenes, and six among 24 detected held scenes. Native Pong kept its original dynamics and camera. Each set of three physical trials followed the same center servo until the same delayed decision. Some points ended before a decision. [Experimental design](abstain-design.md) · [Native controls and frozen model](abstain-source.json.gz)

The model estimates two differences in log-duration relative to holding center. It reuses the anonymous actor-relative motion features, motor frame and reflected kernel. Holding stays available as the zero-advantage action. The two bandwidths and three uncertainty penalties were compared by holding out each entire source group. Early and delayed held groups each contributed half the parameter-selection score. The selected setting kept the original bandwidth and used zero additional uncertainty penalty. The corresponding mean gain in the source folds was .460 seconds over center. The fixed shuffle permuted the three durations within each source scene. [Learner](../../scripts/experimental/tt23_abstain.py)

| Native holdout | Policy | Mean duration | First returns | Holds |
|---|---|---:|---:|---:|
| Earlier independent holdout | Learned | 3.271 s | 15/48 | 2/48 |
| Earlier independent holdout | Shuffled | 2.801 s | 9/48 | 35/48 |
| Earlier independent holdout | Center | 2.695 s | 8/48 | 48/48 |
| New delayed holdout | Learned | 3.081 s | 7/32 | 8/32 |
| New delayed holdout | Shuffled | 2.921 s | 6/32 | 27/32 |
| New delayed holdout | Center | 2.661 s | 5/32 | 32/32 |

The learner moved on every detected delayed holdout. Eight launches had no detected decision context and defaulted to center. The earlier ordinal rule reached 3.348 seconds on the early holdout and 3.081 on the delayed one, with the same first-return counts. The delayed collection spent 240 physical native points and 88,160 native ticks, about 735 simulated seconds. [Earlier ordinal model](direction.md)

## Fresh simulated arm openings

The source model was frozen before the new 64-start course. Each launch supplied the same eight-frame stereo prefix to all policies, and the evaluator physically played and replayed all three motor choices. The table gives contact starts and strict legal return starts. [Frozen launch plan](abstain-plan.json.gz) · [192 verified physical options](abstain-arm.json.gz)

| Policy | Same-sign contacts / legal | Opposite-sign contacts / legal | All contacts / legal |
|---|---:|---:|---:|
| Learned hold or direction | 31 / 12 | 12 / 4 | 43 / 16 |
| Within-scene shuffle | 10 / 5 | 28 / 20 | 38 / 25 |
| Center | 10 / 5 | 28 / 20 | 38 / 25 |
| Earlier ordinal direction | 31 / 12 | 14 / 4 | 45 / 16 |

The observed lateral position and motion had opposite signs on 28 of 30 detected opposite-sign release draws and zero of 31 detected same-sign draws. The learned model moved in all 61 detected scenes. It differed from the ordinal rule on three opposite-sign launches and yielded the same legal-return counts in each family. The shuffled control held every launch. The learned model failed the advancement criterion on the opposite-sign course and tied the earlier ordinal rule on both. The evaluation ended at the opening stage. [Scored choices](abstain-results.json.gz)

```sh
python -m scripts.experimental.tt23_abstain
python -m scripts.experimental.tt23_abstain_arm
```
