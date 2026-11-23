# Next-release validation of motor updates from wrist play

The prediction guard preserved the original Pong map on one drive. On the other drive it admitted a candidate that predicted the next release's encoders better and then scored fewer rebounds on the separate held releases. Across the eight matched held drops, the guard averaged **2.25 consecutive rebounds**, versus **2.875** for frozen Pong and **1.625** for the ungated update.

| Four unexposed drops per drive | Frozen Pong | Ungated updates | Guarded updates |
|---|---:|---:|---:|
| Drive 0 | 2.50 | 2.00 | 1.25 |
| Drive 1 | 3.25 | 1.25 | 3.25 |

Drive 0 admitted one candidate. Its median joint prediction error on the next physical release fell from 19.14 to 14.68 mrad over eight nonoverlapping windows. The fifth-to-first singular-value ratio was 5.96%. On the four held drops the admitted map won one comparison, tied one, and lost two against frozen. Drive 1 admitted no candidates and reproduced the frozen held results. Its last check had enough command spread and improved from 10.21 to 9.53 mrad, a 6.6% reduction.

## Guard and chronology

The branch started from the saved Pong map. It ran six ordinary physical releases per drive with the current map active throughout each release. The committed rolling estimator used that branch's own preparation, play, and terminal encoder events to fit at each boundary. The next release compared the resulting candidate with the active map. Both predictors saw the same real commands and the same returned encoders. A candidate could take effect only after that whole release.

The single fixed check required eight nonoverlapping eight-tick windows with measurable movement in every joint and a fifth-to-first command singular-value ratio of at least 5%. The candidate then had to reduce median joint prediction error by at least 10% and 0.45 mrad. Eight of ten next-release checks fell short of the command coverage requirement. One of the remaining two met the prediction threshold. The last fitted candidate from each drive remained pending because the held games were reserved for evaluation.

Every proposal records its own input prefix digest and physical sample count. All twelve guarded practice tapes replayed exactly. On drive 0 the four held tapes were executed and verified with the admitted map. On drive 1 the unchanged map reused the exact four frozen results. No earlier ungated coefficients entered the guarded learner, and the guard used neither physical wiring labels nor rebound scores.

The guarded learner received 837 and 1,284 actual public command and response pairs. They contained 2.47 and 2.42 s of ordinary visual preparation, followed by 11.45 and 18.96 s of actual play. The six releases added no probe time. The earlier ungated test and its source retention remain independently recorded. [Ungated continual wrist course](rolling-motor-wrist-report.md)

Reproduce with `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_rolling_motor_wrist_guard` after the original wrist course. [Guarded observations and physical tapes](rolling-motor-wrist-guard.json.gz)
