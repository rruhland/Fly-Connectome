# Overhead feedback directions in solo wrist practice

Archived reward-relevant edits did worse than matched random search and stationary hold on both solo test sets. The search selected a moving archived-subspace policy in one of three independent replicas. All three reward-shuffled and random-search replicas selected moving policies. Directly running the overhead head yielded the lowest mean on both solo sets.

| Wrist policy | Standard stock mean / max, 24 drops | Changed stock dynamics mean / max, 12 drops |
|---|---:|---:|
| Archived subspace | 2.25 / 4 | 2.33 / 3 |
| Shuffled source rewards | 2.58 / 6 | 2.17 / 3 |
| Shuffled joint outputs | 2.46 / 4 | 2.58 / 3 |
| Random subspace | 2.50 / 4 | 2.50 / 5 |
| Overhead head | 1.21 / 2 | 1.33 / 3 |
| Stationary hold | 2.46 / 4 | 2.58 / 3 |

Scores are the longest consecutive run of qualifying physical rebounds. The experiment compared 20 scored solo practice drops per search per replica. All 216 held solo action tapes replayed exactly. After solo search, the separately stored overhead head reproduced all 72 archived donor tests, where it averaged 3.39 completed pairs per drop. It averaged 3.72 pairs on 36 new donor starts, with a maximum of 9 and zero for stationary hold. All 36 new overhead tapes independently reproduced their physical pair counts and terminal frames.

## Method
The source archive records 480 signed pairs of candidate feedback edits. Each candidate was scored on six physically simulated drops across three hinged donor arms. The source objective required a descending lower-racket hit followed by an ascending impact on a separate overhead racket. Two consecutive lower-racket hits ended the game. Source observations came from exact simulated geometry. The target used the existing calibration from public RGB and vendor links to estimate ball and paddle motion from its own wrist images and joint encoders.

For each of 48 source updates, the assay normalized each difference between opposite candidate edits, weighted it by the difference in completed source pairs, and summed the contributions across candidates. It kept six eigenvectors of the covariance across those estimates after subtracting the mean covariance from 64 within-update reward permutations. It used the same feature layout as the source archive, including the relative-pose and velocity interactions in the existing readout. The archive also reflects the original source learner's biased exploration around its current best policy.

The three search controls used one reward permutation per source update, a permutation of the true subspace across the five joint-output rows, and a random orthonormal six-dimensional space. Each solo learner started with zero weights and used the same edit magnitudes and small full-space perturbations. Each of three replicas spent 14 practice drops testing signed edits and six drops selecting among the stationary policy and the two update winners. Within each replica all four methods shared practice starts and matched random draws. The selection saw only scalar rebound reward. The separate direct-head baseline spent zero solo practice drops.

The frozen policies ran on 24 fresh standard stock drops and 12 fresh drops on the same stock geometry with shifted ball and actuator dynamics. Held starts were independent of practice and validation. Exact headless action replay matched the contact-event record on every held solo drop. Source retention reran the archived overhead head from separate storage after solo learning and checked its original held roster plus 36 new donor starts.

The checkpoint contains the selected policy and subspace for each replica and search condition. The compressed receipt stores practice scores and candidate parameters, held action tapes and contact metrics, and source-retention tapes. Reproduce with `MUJOCO_GL=egl python -m scripts.experimental.tt24_crossgame_subspace` from the repository root.
