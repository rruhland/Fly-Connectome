# Receiving practice on a limited physical serve family

The lesson narrowed the standard SO-101 rival serve release to the central 28 percent of its seeded ranges. Longitudinal placement, pace, lateral placement and spin still varied from point to point. The ordinary launcher released the initial ball, the strict referee required both service bounces, and the separate stronger rival controlled a physical paddle for any replies. The table, clamp, arm, actuators and score rules were unchanged. [Evaluator](../../scripts/experimental/tt21_receive_lesson_trial.py)

The learner started with an empty strike memory and saw one unmarked outside RGB stream. Both policies used the same factory dry-motion library and inherited flight and rebound priors. Practice explored six perturbations of the proposed stroke, varying timing, total motor amplitude and wrist pitch. The receiver retained its own command tapes with enough optical credit from the outgoing arc and the ordinary point reward. It used camera estimates of the incoming arc to retrieve a nearby example and align the remembered stroke to the new arrival time. [Policy](../../scripts/experimental/tt21_receive_lesson.py)

The fixed training budget was 24 unique tosses. Learned and cold each played 20 fresh matched seeds with the same physical rival. The memory stayed frozen through assessment. The learner attempted a single receiving stroke and recovered toward home while the rival played out the point. An additional 12 pairs of full-spread serves were prespecified if the limited family gained at least three net legal first returns.

## Practice

All 24 practice tosses met the service rules. Across 1,225 control frames, the learner made 23 contacts, 10 legal first returns and won one point. The memory retained 12 examples. Ten landed legally according to the independent referee and two received enough optical credit despite missing a legal landing. Eight practice swings used a previously retained example. [Practice receipt](receive-lesson-practice.json.gz)

## Fresh limited-family tosses

| Policy | Legal first returns | Contacts | Legal rival replies | Point wins |
| --- | ---: | ---: | ---: | ---: |
| Cold factory proposal | 10/20 | 18 | 7 | 3 |
| Learned stroke memory | 12/20 | 20 | 12 | 0 |

The learned policy gained six matched returns and lost four. It recalled a stroke on 19 tosses. The stronger rival landed a legal reply after all 12 learned-policy returns, compared with seven of the ten cold returns. The net gain of two fell below the prespecified full-spread threshold. All 24 practice scenes and 40 held branches reproduced exact scores and events on physical replay. [Held receipt](receive-lesson-held.json.gz)

## Relative dry-model follow-up

A separate 12-pair schedule reused the same frozen memory. On each shared RGB estimate, the factory model compared the stock proposal and the actual recalled tape, including any time spent waiting for the remembered onset. Both faced the same three sampled trajectories. The receiver used a recalled stroke if it was legal in at least two dry trajectories, had more predicted legal outcomes than stock, and lowered mean dry cost by more than 0.25. The held dry predictions were prospective and the receiver collected zero additional target practice.

The guard compared both proposals on all 12 fresh tosses and selected stock every time. Both policies made nine legal first returns and won seven points, with identical command tapes. All 24 branches replayed exactly. The safeguard therefore preserved the stock receiver on this schedule. [Guarded receipt](receive-lesson-guarded.json.gz)

## Reproduction

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt21_receive_lesson_trial
python -m scripts.experimental.tt21_receive_lesson_trial --stage guarded
python -m scripts.experimental.tt21_receive_lesson_trial --stage verify
```
