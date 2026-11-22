# Learning stroke value from the clock

The duration labels helped identify whether an archived point continued for another 35 frames. At 60 and 100 frames the advantage over shuffled labels narrowed, and holding out whole source cohorts reversed it. The fixed archival criterion for a native trial was unmet. [Fit, source hashes and held predictions](duration-value-archive.json.gz)

The ordinary games, lateral-policy experiment and learned-coach comparison supplied 62 previously verified complete points, including the 41-, 37-, 27- and 23-return rallies. The six earlier archives supplied 55 verified points. Identical own command and stereo histories for the same launch and serving side were counted once. The resulting 100 points came from 67 different launches. There were 339 distinct eligible decisions from 63 of those launches. The 200 decisions from the three newer sources comprised 45 ordinary game decisions, 81 in the lateral-policy experiment and 74 from the coach comparison. [Ordinary games](../tt22/ordinary-game.md) · [Lateral-policy rallies](../tt22/lateral-rallies.md) · [Learned coach](coach.md)

At each archived selection, a command-driven nominal arm reconstructed the paddle state. The existing stereo velocity estimate or causal inverse flight fit initialized the ball. The nominal simulator replayed the recorded near-arm command segment and forecast 44 frames in a scene with the known table, ball and actuator dynamics. The motor segment length came from the recorded stroke's proposed timing. Four decisions needed zero padding because the actual point ended before the end of that command segment. The feature vector used the previous actor-relative trajectory descriptors and local 3D motion moments. This reuses the geometric stroke library that generated the archived actions and the known physical model required to forecast them. [Decoder and fit](../../scripts/experimental/tt23_duration_value.py)

The training target used only the number of frames from the motor decision to the point's end. It mixed three bounded survival values of the form `1 - exp(-frames / scale)`, with scales of 32, 72 and 150 frames and weights of .55, .30 and .15. The kernel used the same eight-component projection and bandwidth rule as the earlier trajectory value. Each launch carried equal total fitting weight. The shuffled control kept the same inputs, distances, weights and fitted support with a permutation of the duration targets. Referee calls, collisions, points won and legal returns supplied no training labels.

| Held unit | Labels | Duration Spearman | Per-launch mean absolute value error | 35-frame AUC | 60-frame AUC | 100-frame AUC |
|---|---|---:|---:|---:|---:|---:|
| Launch | Duration | .263 | .158 | .786 | .613 | .571 |
| Launch | Shuffled | .162 | .177 | .696 | .591 | .540 |
| Source cohort | Duration | .114 | .172 | .722 | .546 | .507 |
| Source cohort | Shuffled | .180 | .177 | .721 | .579 | .556 |

The per-launch and per-cohort constant baselines had mean absolute errors of .200 and .200. The four source cohorts were the earlier archives, ordinary game, after-service and coach-held. Every fold excluded all training examples from the held launch seeds. In the newer coach group, the duration model gave a 60-frame AUC of .647 versus .558 for shuffled labels when the other launch groups were available for fitting. Excluding whole source cohorts tested the model against larger changes in the motor policy and partner.

On the same 200 newer decisions, the fit from the earlier archives alone achieved .362 Spearman and .692 AUC at 60 frames. Adding the other newer launches to the training folds lowered these to .213 and .613, while improving mean per-launch absolute error from .174 to .161. The larger archive improved calibration of the scalar estimate and weakened the ordering of longer outcomes.

Eight pairs of different archived motor options from six launches shared the exact preceding near and far command history and near stereo measurements. The real-label fit ranked the longer outcome first in five pairs. The shuffled fit did so in seven. The labels measure the whole later point, including later decisions by both arms. Same-prefix ordering is therefore an archived selection check and the sample is small. [Matched options and their terminal times](duration-value-archive.json.gz)

The live condition required at least .70 AUC at 60 frames, a .15 Spearman gain over the shuffle and lower mean per-launch error than both controls. It also required a Spearman advantage when holding out whole source cohorts. The archive analysis took 67 seconds and consumed zero new target play. The receipts retain the fitted and shuffled checkpoints. [Artifact](duration-value-archive.json.gz)

```sh
python -m scripts.experimental.tt23_duration_value
```
