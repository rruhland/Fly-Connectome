# Continuing solo practice after table receiving

The second practice course started from the retained four-drop checkpoint and its verified post-table bounce record. The same deployed file stayed available throughout the new practice. Fifteen additional releases used the wider held release generator. Every other release also increased lateral position spread by 23%, lateral velocity by 60%, spin by 35% and drop height by 34 mm. These are simulated challenge conditions with the same clamped arm and camera. [Practice and validation tapes](tt29_bounce_continued.json.gz)

Wrist RGB and joint encoders drove the arm. During practice only, paired fixed RGB supplied triangulated ball centers and finite-difference motion labels. The learner refit the wrist motion map at drops 5, 10 and 15 and blended the new readout at 35% with the previous estimator. It also tried five local reference trims and selected a +0.045 rad wrist-roll trim using rebound rewards from its own completed drops. The frozen challenger used 429 valid optical samples. Table rewards played no part in this solo update. [Continued training implementation](../../scripts/experimental/tt29_bounce_continued.py) · [Prior table and bounce retention](tt29_bounce_retained_receive.json.gz)

The candidate had to win more paired validation streaks than it lost, add at least two qualified rebounds, and match or improve median streak. It also had to match or improve total rebounds on the difficult validation subgroup. Those eight validation seeds and a further ten held seeds were disjoint from one another and from every earlier bounce course. The ten held releases were played after the promotion decision and include five difficult drops.

The gate rejected the update. The incumbent made 20 qualified rebounds and reached a 14-rebound streak. The challenger made seven and reached a two-rebound streak. Both made two rebounds on the four difficult releases. One ordinary release accounted for the paired streak loss. The deployment copy has the exact incumbent hash. [Deployment checkpoint](tt29_bounce_continued_deploy.json)

| Untouched assessment | Releases | Incumbent rebounds / longest | Challenger rebounds / longest |
| --- | ---: | ---: | ---: |
| Ordinary | 5 | 28 / 16 | 5 / 1 |
| Difficult | 5 | 7 / 3 | 5 / 1 |
| All | 10 | 35 / 16 | 10 / 1 |

The challenger lost four paired streaks and won none. Both checkpoints had an overall median streak of one. The practice course used 861 control ticks, or 14.35 simulated seconds. All 51 command streams from practice, validation and held assessment reproduced their physical referee outcomes. The retained checkpoint remains the deployed result. [Independent held tapes](tt29_bounce_continued.json.gz)
