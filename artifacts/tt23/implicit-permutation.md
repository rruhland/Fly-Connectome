# How often do shuffled Pong rewards reproduce the online arm gain?

The five real Pong fits and the original pooled shuffle gave the same directional preferences on all 32 physical starts. A separate randomization audit asked how often five independent shuffled fits would jointly match that performance. On the fixed arm bank, **123 of 4,096** shuffled groups matched or beat the real fits on both capped launches to ten and own legal returns. Of those, 121 tied exactly on both outcomes and two improved both. [Receipt](implicit-permutation.json.gz)

| Policy or reference | Launches to ten, capped at nine | Own legal returns in eight launches |
|---|---:|---:|
| Mean of five real Pong fits | 4.798 | 14.666 |
| Fixed pooled Pong reward shuffle | 4.798 | 14.666 |
| Flat prior | 6.089 | 8.749 |
| Mean across independent shuffled groups | 6.052 | 10.043 |
| Middle 90% across independent shuffled groups | 5.295–6.788 | 7.336–12.833 |

| Matching or beating the real fits | Count out of 4,096 | Fraction | Add-one Monte Carlo tail |
|---|---:|---:|---:|
| Capped launches, lower is better | 129 | 3.15% | .03173 |
| Own legal returns, higher is better | 123 | 3.00% | .03027 |
| Both outcomes | 123 | 3.00% | .03027 |

Each draw independently permuted the seven observed durations within every native start in each of the five disjoint Pong training sets. The original reflected direction rule rebuilt the five fits. The audit applied each fit separately to the same four eight-launch arm streams, preserving the 256 paired exploration sequences per stream, the shared arm and coach, and the update from only the selected point’s elapsed time. It averaged scores over exploration draws, physical streams, and the five fits. Points reaching ten were identified by the strict physical referee. A sequence that never reached ten was assigned nine launches as in the original analysis. [Native direction fit](direction.md) · [Arm bank](jog-bank-design.md) · [Online learner](implicit-online-design.md)

The audit exactly reproduced the real and fixed shuffled results for each of the five fits on all four streams and reproduced the pooled tie. It reused the original online evaluator across equivalent eight-launch prior patterns. The comparison was added after observing the original shuffled tie. It estimates how often independently permuted native rewards attain the observed result conditional on these 32 physical starts. The exploration sequences and repeated source fits share the same physical starts. An independent physical confirmation is a separate test. [Completed online results](implicit-online-results.json.gz) · [Audit runner](../../scripts/experimental/tt23_implicit_permutation.py)

Run `python -m scripts.experimental.tt23_implicit_permutation` in the repository’s MuJoCo environment.
