# A physical motor response learned in native Pong

The receiving SO-101 arm made **43 native Pong returns** in ordinary visual play, versus **2** when it held the paddle still. It returned at least one ball on 17 of 18 points. The active controller won one point, lost none and reached the eight-second cap in 17. The stationary condition won two and lost 16. The arm made its ordinary game movements throughout. There were no separate calibration moves.

The screen controller in this first study extrapolated the approaching ball from raster observations. A separate study below used a readout trained on native game rewards. The native ball, opponent, contacts and scoring used the installed Pong engine. Only the paddle plant changed. One fixed encoder projection drove the screen paddle from the physical arm. The source controller moved along one requested actuator direction and saw the native raster. Two evaluator command matrices agreed exactly on that direction and differed on the other four. The source commands, physical encoder histories and point outcomes were reproduced in all 36 checks across the two installations.

A causal filter and one response vector were fitted from the logged command and encoder pairs, using source prediction error to choose the filter. The other columns kept their original identity values. Matched pairs recovered the exercised response to 0.0098% relative error. Circularly shuffling the encoder sequences within points raised that error to 134.1%. The learner received neither true matrix.

## Immediate wrist transfer

| Matched wrist cohort | Stale actuator map | Matched Pong pairs | Shuffled Pong pairs |
|---|---:|---:|---:|
| One-direction change, mean consecutive rebounds | 8.60 | 14.10 | 1.90 |
| One-direction change, at least ten consecutive | 4/10 | 9/10 | 0/10 |
| Dense change, mean consecutive rebounds | 1.50 | 1.20 | 1.00 |
| Dense change, at least ten consecutive | 0/10 | 0/10 | 0/10 |

The matched estimate beat stale on eight of ten one-direction drops. It was frozen before any wrist rewards. The arm switched to the normal physical bouncing serve without an intervening identification move. The bouncing actor and wrist optics were unchanged across conditions.

The ordinary Pong command matrix had singular values 4.274, 0, 0, 0 and 0. The unaltered wrist policy's commands had singular values 4.935, 3.973, 3.231, 2.995 and 2.009. Both source matrices therefore generated the same histories for this Pong controller, while their physical responses differed under the five-direction wrist controller. The supported result transfers actuator response through a shared physical arm interface.

The original identity map was kept in separate memory and explicitly recalled on returning to the unaltered context. It reproduced all six earlier recorded stock action tapes and contact events. On ten new unaltered stock drops it averaged 5.00 rebounds with 3/10 ten-rebound streaks, versus 1.10 and 0/10 when the new map was deliberately used for that context. All 80 new target tapes replayed exactly.

Reproduce with `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_pong_arm`. The `--source-only` and `--held` flags run the individual stages.


## How much play does a reward-trained Pong player need?

An archived player previously trained from native point rewards drove the physical joystick in eight native games. Its screen images, readout coefficients and motor requests came from the original learned-player code with the adapter attached at its paddle port. It made eight legal returns across four games and scored four wins and one loss. The zero-readout control made zero returns, with three wins when the opponent missed. All eight active source games reproduced their command and encoder sequences. [Native training](tt24_joint_pong.md) · [Runner](../../scripts/experimental/tt24_played_pong_prefix.py)

Only the first 30 physical control ticks of the **first** source game went into the shortest motor estimate. The fitter predicted encoder displacement from that one stream of normal actions, then froze its five measured effects for the one exercised motor direction. The target already had the donor-trained bounce readout and calibrated wrist vision. It started the 12 independent five-second wrist games immediately, with zero target teaching or inspection. Every target action tape reproduced its score.

| Experience used to estimate the changed direction | Mean longest streak | Games reaching ten |
| --- | ---: | ---: |
| Original motor map | 7.83 | 5/12 |
| First 0.50 s of learned Pong play | 12.33 | 9/12 |
| First 2.00 s of learned Pong play | 12.42 | 9/12 |
| Same 2.00 s with encoder timing shuffled | 1.08 | 0/12 |

The early estimates recovered the changed column with relative errors of 0.70% and 0.16% under the independent evaluator. A half-window shift can pair motion with opposite commands, explaining why the shuffled arm scored below stale. This establishes fast transfer of a motor response from a reward-trained Pong player on one independently held wrist cohort. The shared hardware and prior five-joint bounce readout are essential parts of the test. [Source histories, fitted matrices and all 48 rebound tapes](played-pong-prefix.json.gz)
