# Does a stronger native Pong player accelerate 3D bounce learning?

The qualified Pong player's observed motion accelerated early reward learning in the 3D bounce worlds relative to zero initialization and shuffled Pong motion. An independent random initialization of equal magnitude performed similarly to coherent Pong. The experiment therefore leaves open whether native Pong knowledge adds value over an equally large random start in this receiving architecture. [Full paired result](pong-world-strong-learning.json.gz)

## The runnable source policy

The older October 29 relational-memory policy returned 32/32 opening and 147/148 later balls on its fresh cohort. The documented frozen checkpoint and required ancestor assets are absent from this workspace. This test uses the portable November 19 score-trained, risk-guarded actor. Its independent qualification returned 32/32 opening and 65/66 later balls against a 0.65-speed opponent and won 18/32 points after an own return against a 0.35-speed opponent. Every source game in this study uses the verified archived score, motor and timing checkpoints. [October result and checkpoint paths](../../docs/plans/2026-10-29-relational-memory-outcome.md) · [November qualification](../../docs/reconstruction/2026-11-19-score-placement-qualification.md) · [Portable source and receipts](../m1b/2026-11-19-score-placement-confirmation.zip)

The November actor inherits the frozen visual tracker, ordinal scene binding, action-axis convention, intercept and wall-reflection rules, and a visual guard that can cancel risky learned placements. It carries motor order and timing states learned in earlier physical-action recovery experiments. Motor and timing estimation remain live during source collection. Its frozen shot-placement readout came from 96 native points, 573.65 simulated seconds, 28,716 camera samples and 78 terminal-score updates over supplied offsets. The new tapes measure the combined behavior of these learned and supplied components. [Motor and timing lineage](../../docs/reconstruction/2026-11-19-timing-recovery-qualification.md)

The instrumented collector matched the entire recorded action stream, contact clock, score and duration on six originally qualified launches, with three per opponent and 2,753 camera samples in total. It then observed 48 additional source-fit and 24 held points, each split equally between the two opponent speeds. The fresh launches are disjoint from the actor's archived score training and confirmation schedules and the earlier duration-policy collection. Replaying the six qualification and four fresh tapes with the actor's unused contact input set to zero reproduced their records exactly. Every issued command came from the archived actor. The transfer readout uses camera tracks, the known direction of the paddle's action axis, and measured paddle silhouette size. Source contact and point labels are used for reporting the actor's behavior. [New launches, optical cues, action digests and native outcomes](pong-world-strong-native.json.gz)

| New native exposure | Points | Simulated seconds | Camera samples | Opening returns | Later returns | Points won after own return | Optical cues |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source fit | 48 | 435.62 | 21,790 | 46/46 | 64/65 | 17 | 1,449 |
| Native holdout | 24 | 201.78 | 10,093 | 20/20 | 31/31 | 6 | 739 |

Six points ended before an incoming opportunity and stayed in the 72-point denominator. The actor returned every ball it faced on the 36 stronger-opponent points. The 23 points won after an own return all came against the slower opponent.

## Cross-game motion readout

The fit relates current native visual gaps, motion and paddle offset to the paddle center the actor actually reaches. It uses every third sample with a five-frame uninterrupted optical history, within a supplied near-encounter region of 0.42–4.5 paddle half-spans in depth and 5.2 half-spans overall. The target is the mean of the observed paddle positions 0.24 through 0.42 seconds later, relative to its first observed pose. The readout uses only camera-derived quantities. The observer also records the issued signed alias according to the actor's action belief.

The registration broadcasts the source tangent direction to both 3D tangent axes. It maps paddle silhouette half-span to a 0.076 m blade radius and maps the median absolute source normal speed to 2.1 m/s. The bounded output is expressed in receiving paddle coordinates. Only 12 of 42 tangential-position coefficients are initialized from the source. All vertical, tilt and tangent response-to-tilt coefficients begin at zero. The shuffle exchanges future observed displacements within coarse strata for present pose, separation and ball approach. It matches the coherent initializer's coefficient norm. An independent random start also matches that norm and sparsity. A fourth initialization sets all weights to zero. [Frozen readout and held native prediction](pong-world-strong-initial.json)

Among 339 native held cues with at least 3 mm of motion after conversion to receiving coordinates, the coherent fit predicted actual future pose within 19.86 mm RMS. Predicting no movement gave 25.39 mm and shuffle gave 25.62 mm. The coherent fit predicted correction direction on 82.3% and shuffle on 69.9% of those cues. Twenty of the 24 held games contributed cues. [Held motion audit](pong-world-strong-initial.json)

## Matched receiving-world test

The test registers 24 independent 3D learning streams. Within a stream each arm trains from the same random perturbations and launches, with a fixed matched exploration variance. After initialization all 42 coefficients may change. Every update uses 384 scored simulated launches, of which 96 use wider conditions. Each evaluation uses 128 nominal and 128 stress launches per stream, held from updates. The predeclared endpoint averages consecutive rebounds across those profiles and updates 1, 2, 4 and 8. Paired intervals resample entire learning streams. There are zero physical-arm practice episodes in this test. [Registered schedule and budgets](pong-world-strong-plan.json) · [Full paired learning and exposure receipt](pong-world-strong-learning.json.gz)

| Initializer | Update 0 | Update 1 | Update 2 | Update 4 | Update 8 | Update 16 |
|---|---:|---:|---:|---:|---:|---:|
| Coherent Pong motion | 1.75 | 2.81 | 5.44 | 9.31 | 12.29 | 13.96 |
| Shuffled Pong motion | 1.71 | 2.68 | 4.97 | 8.89 | 12.15 | 13.83 |
| Zero cold | 1.55 | 2.42 | 4.59 | 8.54 | 11.80 | 13.73 |
| Norm-matched random | 1.75 | 2.84 | 5.42 | 9.18 | 12.29 | 13.88 |

Entries are mean consecutive rebounds across nominal and stress holdouts. Coherent Pong increased the predeclared early average by 0.627 rebounds versus zero cold (paired 95% interval 0.501 to 0.765) and by 0.293 versus shuffled (0.183 to 0.408). It led zero in 24/24 streams and shuffle in 18/24. The coherent initializer differed from norm-matched random by 0.034 rebounds (−0.050 to 0.111). The early advantage over zero and shuffle appeared on both profiles. By update eight the matched budget was 3,072 scored launches per stream and arm. Elapsed simulated training frames averaged 619,992 for Pong, 560,746 for zero cold, 586,502 for shuffle and 615,873 for norm-matched random because longer-lived rollouts run longer. The evaluation matches launch and perturbation budgets.

The deployable checkpoints contain the exact eight-update weights from the prospectively designated first learning stream. For an independent replication across training streams, the bank additionally provides the next five consecutive registered streams and all their controls. Fresh replays matched every learning update and evaluation exactly across the six streams. The re-exported first stream matched all four original checkpoint hashes. The reported endpoints here are from the cheap 3D worlds. [Six-stream checkpoint bank and replay hashes](pong-world-strong-bank.json)

## Reproduction

Use the table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_pong_world_strong --audit
python -m scripts.experimental.tt24_pong_world_strong --workers 2
python -m scripts.experimental.tt24_pong_world_strong_bank
```
