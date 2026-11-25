# Where the ordinary rally and the earned movement diverge

The existing school earned 18 real returns across 40 practice forks. Its failure to transfer is not just a shortage of retrieved motions. Even with the winning joint tapes already in the candidate set, the standard optical simulator called only two of them legal. It predicted separation on all 18. The difficulty was predicting where a blade contact would send the ball, not recognizing that the paddle reached it.

## Ordinary training history

In the 128 continuously randomized TT46 training points, the opponent legally offered the next ball in every point. The receiver scored 756 times and ended every chain with its own failed return. Fifty endings were touched misses in the precise TT57 low-face, height and ordinary-search envelope. Sixty-two were untouched. The remaining 16 were touched outside the height or search envelope. At own score ten through nineteen, the receiver failed on 24 of the 178 opportunities, including 13 school-envelope and seven untouched endings. These descriptive counts do not assume independent strokes or predict the score of a changed controller. [Frozen history and stage counts](ordinary-history.json)

All 62 untouched offers had a measured receiving-line crossing and a normal optical search. Fifty-four projected below 1.000 m and none reached 1.030 m. Forty were within 10 cm of the centerline and none was more than 30 cm wide. The normal searches never found a nominal legal path on 51 and never found a robustness-voted path on 54. Seventeen ended via fixture contact. Thus rescuing touched balls of the TT57 class still would not cover the larger untouched class. [Independent referee recount and missed-offer geometry](physical-check.json)

## Fixed-tape recognition

All 40 practice forks were physically replayed to the original decision. The live RGB and encoder prefix initialized a separate nominal plant. The same 80 already-issued tapes ran in each evaluator condition, rebuilding references from the signed, saturated commands. No new physical treatment or held result enters this diagnosis. Substitutions replace the ball's position and translation velocity, spin, bounce latch, or the attained joint and paddle state. All 16 combinations are recorded alongside rollout from the original exact plant. Privileged states never enter an actor.

| Evaluator initialization | Correct accept, out of 18 scored | False accept, out of 62 failed |
|---|---:|---:|
| Standard RGB and encoders | 2 | 15 |
| Exact body, otherwise standard | 2 | 15 |
| Exact bounce, otherwise standard | 2 | 14 |
| Exact spin, otherwise standard | 1 | 10 |
| Exact position and translation velocity | 8 | 6 |
| Exact translation and spin | 15 | 0 |
| Original exact native plant | 15 | 0 |

The three exact-native rejects were real net clips. Only one got a next legal opponent answer. Sixteen of the 18 scored and got an ordinary legal answer, and all 18 scored tapes matched both prior incoming events and passed strict CAD clearance. The real measured contact radius had a 74.54 mm median, and 12 of the 18 contacts were within 74–76 mm of the blade center. Near these rim contacts, millimeters of translation and tens of radians per second of unknown spin can change the predicted direction. A better body estimator would not address the discrepancy on these forks. This identifies an interaction of observation and motor margin. It does not establish that correcting either input alone will improve whole games. [All scored model substitutions](practice-factors.json.gz) · [Mode counts and residual quantiles](report.json) · [Measured contacts and net outcomes](physical-check.json)

## Release-cross-fitted public uncertainty gate

The alternate scores a candidate across correlated translation and spin residuals learned from other practice releases. It uses only current stereo estimates, exposure age, the inferred rebound phase, and the established encoder-initialized twin at decision time. All rows of the evaluated release are excluded from its uncertainty estimate. It does not require the arbitrary uncorrected nominal estimate to be legal. It needs support from two different calibration releases. Neither physical receipt nor recorded opponent state can select a candidate. This is still recognition of fixed previous tapes, not retrieval of a new feasible command.

| Gate | Correct selection | Wrong selection |
|---|---:|---:|
| Standard nominal-plus-offset votes | 0 | 12 |
| Residual mixture with 0.12 vote fraction | 6 | 19 |
| Residual mixture with 0.35 vote fraction | 2 | 11 |
| Residual mixture with 0.50 vote fraction | 2 | 6 |
| Residual mixture with 0.80 vote fraction | 1 | 3 |

All six correct choices at the permissive threshold also received an ordinary legal answer. The false-choice burden is too high to authorize a whole-game change. Even the 0.80 setting selected the wrong tape three out of four times. Do not weaken the stock veto on this evidence. The calibration shows that other observationally plausible states can expose valid approaches, but candidate teaching needs a physical tolerance margin across nearby ball states. On untouched, very low central shots it also needs a different preparation path that reaches the ball and produces a returnable departure. Do not count any of this as a held rescue or progress toward the 100-return milestone. [Every cross-fitted vote and exclusion](cross-fitted-gate.json.gz) · [Threshold comparison](gate-report.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt60_school_bottleneck history
python -m scripts.experimental.tt60_school_bottleneck audit
python -m scripts.experimental.tt60_school_bottleneck report
python -m scripts.experimental.tt60_school_optical_gate fit
python -m scripts.experimental.tt60_school_optical_gate audit
python -m scripts.experimental.tt60_school_optical_gate report
python -m scripts.experimental.tt60_school_report
python -m pytest -q tests/test_tt60_school_bottleneck.py
```
