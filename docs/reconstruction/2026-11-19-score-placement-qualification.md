# Qualifying score-trained Pong placement

The first score-trained policy won 24/32 points after its own return, compared with 3/32 for inherited center aim and 14/32 for fixed exploration. It passed the scoring gain criterion and fell short of the later-return criterion against the stronger opponent. [Portable v1 evaluation receipts](../../artifacts/m1b/2026-11-19-score-placement-qualification.zip)

This comparison is prepared for a scoring policy trained from terminal game scores after timing qualification. Each score policy receives the same actual carried motor and timing checkpoint. Motor and timing updates stay live, and scoring updates stay frozen throughout qualification. The environment keeps the actuator settings constant during each score and retention game.

The independent schedule pairs 32 starts against the opponent at speed 0.35 and 20 starts against the opponent at speed 0.65. Each cohort balances the opening serve direction. The score-focused cohort compares the frozen trained score policy, the inherited center policy, and a frozen untrained policy with fixed exploration. The retention cohort compares the trained policy with the inherited center policy. All games end at a score or 12 seconds. The runner assigns policy randomness independently of each native serve seed and shares it between matched arms.

The score endpoint is a point won after at least one own return. A point won by a natural serve contributes zero and remains in the planned denominator. On the same starts, the trained policy must improve on inherited center by at least 15 percentage points with a one-sided exact paired p-value of at most 0.05. The report shows the comparison with fixed exploration separately. The stronger opponent measures ordinary interception. First returns must reach at least 90%, and later returns must reach at least 95% on 30 or more opportunities. Both rates must be within five percentage points of the matched center policy.

Preparation pins fresh schedules before exposure to the candidate. The evaluation records the frozen source and checkpoint hashes, every game receipt, natural serve wins and the first-contact clock parity of each pair. [Independent qualification runner](../../scripts/experimental/run_pong_score_qualification.py)

## First held-out comparison

The scoring checkpoint came from 96 practice games after timing had fully qualified. It received 78 terminal-score updates. The independent test started from the exact restored Stage 2 motor and timing state for every frozen score-policy arm. Its schedules shared no starts with practice or development evaluation.

| Policy at opponent speed 0.35 | Points won after an own return | Natural-serve wins | First returns | Later returns |
|---|---:|---:|---:|---:|
| Trained score policy | **24/32** | 4 | 28/28 | 15/15 |
| Center aim | 3/32 | 4 | 28/28 | 63/63 |
| Fixed exploration | 14/32 | 4 | 28/28 | 43/43 |

The trained policy gained 21 paired own-hit wins and lost none against center aim, an increase of 65.625 percentage points with one-sided exact paired p = 0.000000477. It gained 11 and lost one against fixed exploration, an increase of 31.25 percentage points with p = 0.00317. Each pair shared the same first-contact clock. The exposure to later balls varied because the score-trained policy ended more points early by winning them.

Against the opponent at speed 0.65, the trained policy returned 19/20 first and 30/33 later balls. Center aim returned 20/20 first and 49/49 later balls. The trained policy passed the first-return criterion and its 90.9% later-return rate fell below the 95% criterion. Four trained episodes ended in a loss. Neither arm scored or received a natural-serve win in this cohort.

The independent comparison used 136 native games and took 273.608 seconds. The locked spec SHA-256 is `b7fb9cf12a45f8f2d0bede7c0eb7bc4962bb7f6523bf71f66776c1afb8ca0834`, and the frozen score checkpoint SHA-256 is `484059d03112dc06b18c946d8da244e0d229d5e63ee3da08ebef20fd94c762eb`. The archive contains the exact evaluation runtime and paired game receipts.

The follow-up registers 32 fresh starts at each opponent speed, with the same score, first-return, and later-return thresholds. Its locked spec SHA-256 is `5b17fef333c3fe14664cc142744c1dac92775543c9a42765c538e8d21144c73c`.

Before dispatch, an addendum pinned the public pursuit guard and carried forward the byte-identical score array and timing state. The guard uses camera observations to estimate a lower bound on the opposing paddle’s pursuit speed. It cancels an edge placement when the observed pursuit or approach makes the shot risky. Trained, center and fixed-exploration arms all use the same guard. The addendum leaves the original schedule and thresholds intact. Its SHA-256 is `6c57ef5269831051b93f77eb4cb3a79be2e08bf37223800c7d8d2645a2d45023`.
