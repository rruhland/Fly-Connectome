# More frequent camera rechecks did not rescue three known opening misses

On three diagnosed opening misses, checking every fresh image after a post-bounce failure increased search refreshes from 12 to 19 and did not produce a legal return. The extra attempts added nominal paddle contacts, but no additional candidate passed the nominal landing test. [Per-point outcomes and exact refresh ticks](joint-cadence-results.json)

Both runs used the same frozen guarded controller, optical readout, joint fragment bank, independent plant, proposal budget, and measured openings. The only difference was when the planner ran. The fixed controller checked every five ticks. After a failed post-bounce search with no live incumbent, the adaptive controller checked on fresh frames every two ticks for up to eight ticks. A stable, well-supported forecast could wait seven ticks. The three releases were previously observed failures, not a new held benchmark. [Diagnostic roster](joint-cadence-plan.json) · [Implementation](../../scripts/experimental/tt38_joint_cadence.py)

| Same three failures | Legal near | Search refreshes | Total native trials | Search CPU seconds |
| --- | ---: | ---: | ---: | ---: |
| Fixed cadence | 0 | 12 | 379 | 1.40 |
| Fresh-frame retry | 0 | 19 | 590 | 2.13 |

The extra searches found no legal nominal action on releases 13628 or 7775. Release 9203 had six nominal-legal candidates in each condition. Neither had enough optical agreement to replace raw feedback. These data rule out speeding up this particular schedule as a sufficient remedy for the chosen failures. They do not rule out combining faster rechecks with better candidate generation or more accurate observations.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt38_joint_cadence plan
python -m scripts.experimental.tt38_joint_cadence fixed
python -m scripts.experimental.tt38_joint_cadence adaptive
python -m scripts.experimental.tt38_joint_cadence report
```
