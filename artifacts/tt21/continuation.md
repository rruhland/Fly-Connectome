# Learning a continuation stroke from physical rally credit

The player kept the TT20 unmarked SO-101 clamp, wrist and two fixed native RGB views, factory stroke catalogue, inherited flight and rebound models, and strict referee. It considered a one-frame lead, nominal phase, and one-frame lag for the local flight estimate used to choose a continuation stroke. The opening stroke and recovery commands stayed fixed. During play the player identified a later stroke from its own earlier plan or completed serving sequence and the existing camera-only return gate. [Receiver](../../scripts/experimental/tt21_continuation.py)

The reset curriculum used every first legal rival reply in the archived TT20 frozen-policy game, comprising four receiving starts and two serves. Each drill began at the recorded regular opening and re-executed the original learner commands up to the rival contact. The physics reconstructed the ball and arm state through actual contact. The drill then relinquished the archived commands and used new native RGB. The referee supplied credit in priority order for a legal continuation, physical contact, and a point win. The seed, replayed measurements, and referee events stayed on the evaluator side. [Source game](../tt20/curriculum-frozen-2082473549.json.gz) · [Trial](../../scripts/experimental/tt21_continuation_trial.py)

The ordinary-practice comparison assigned the same three candidate phases to six predetermined normal starts. The reset practice first showed that a fixed shift could gain a contact and lose a legal return. The final learner kept nominal strokes when the dry planner predicted a legal landing. It selected a fallback using only practice replies where the nominal plan failed that criterion. Assessment froze both selections and compared them with the nominal phase on 16 fresh paired openings per opponent, alternating two receiving points and two learner serves. Each assessment started from an ordinary reset and ran until the referee ended the point. The stronger opponent used its own search to drive physical rate-limited paddle motion. [Rival implementation](../../scripts/experimental/tt20_opponent.py)

## Practice outcomes

Six reset starts supplied a legal rival reply in all 18 candidate trials. Practice took 1,368 control frames, including 759 frames of physical opening replay and 609 of live continuation. Six ordinary starts supplied two legal rival replies per phase. They used 1,015 control frames in 16 distinct live executions. Two further candidate trials had identical commands by construction because the continuation planner never ran. All 36 recorded branches reproduced the referee events on replay.

| Phase applied to every continuation | Reset legal returns | Reset contacts | Ordinary legal returns | Ordinary contacts |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 2 | 2 | 0 | 1 |
| One frame ahead | 0 | 2 | 0 | 1 |
| One frame earlier | 1 | 3 | 0 | 0 |

Among the four reset starts where the nominal model supplied no legal stroke, only the earlier estimate produced a contact. It failed to land legally. This selected the guarded one-frame adjustment. Ordinary practice selected the nominal phase. Three nominal practice failures had own-half bounce positions of 0.28 m, 0.31 m and 0.40 m from the centerline. The model found no candidate for those replies at any observed planning frame. Both nominal legal continuations bounced within 0.04 m of the centerline. [Practice receipt](continuation-practice.json.gz)

## Uninterrupted held points

Legal continuation and point-win counts matched across all three policies on 32 paired starts.

| Opponent | Held points | Legal rival replies | Legal continuations | Points with a second learner return | Point wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 16 | 5 | 1 | 0 | 5 |
| Stronger search | 16 | 5 | 2 | 1 | 5 |

The standard continuation followed the learner’s serve. The guard changed commands on standard start `987089346`. Both strokes contacted the ball and lost the point. The other 31 command tapes matched their frozen controls. The stronger opponent produced five legal reply opportunities and the guard left the executed strokes unchanged. All 96 recorded held branches replayed exact scores and physical events. [Held receipt](continuation-held.json.gz)

## Reproduction

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt21_continuation_trial
python -m scripts.experimental.tt21_continuation_trial --stage verify
```
