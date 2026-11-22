# Choosing physical strokes with a learned optical trajectory value

## Source value

Six existing policies replayed native Pong with randomized rotations, reflections, palettes and command IDs. The existing anonymous role and actuator discovery reconstructed actor and mover tracks from those images. The value learner used actor-relative directions, displacements and speeds from short approaching trajectories to predict the logarithm of native episode time remaining after the sampled future. Duration was the only label. Short gaps in the decoded motion were interpolated and the last visible velocity extended missing tails. The source receipt includes each chronological optical track and the action digest. Forty-eight favorable and unfavorable original-policy tapes reproduced their installed action digests and duration on held native launch seeds. [Source receipt](value-source.json.gz) · [Motion descriptors](../../scripts/experimental/tt22_value.py)

| Pong future | Learning / held paths | Held Spearman, duration labels | Held Spearman, shuffled labels |
|---|---:|---:|---:|
| 24 frames | 30 / 24 | .494 | .274 |
| 44 frames | 27 / 18 | .626 | −.274 |

[44-frame source receipt](value-source-extended.json.gz)

## Shared motor proposals

The SO-101 uses the existing geometric timed-stroke generator. It forecasts incoming crossings at supplied depths, offers pitches, advances and timing offsets, and reaches the resulting blade poses with nominal inverse kinematics. Its supplied swing scheduling and rearming use incoming direction and court-relative position. Both geometric placement and uncapped lateral placement using the frozen Pong duration gains enter the same candidate set. A deterministic stride limits the set to 112. The nominal MuJoCo twin starts from commanded body state and a native RGB mover estimate. The learned value of actor-relative position, displacement and speed is the sole ranking criterion among generated proposals. The far robot keeps its ordinary physical cooperative control and the official referee scores play. [Motion-only player](../../scripts/experimental/tt22_value_player.py) · [Matched runner](../../scripts/experimental/tt22_value_trial.py)

## Pong-only transfer

On six matched complete points, the 24-frame Pong model and the shuffled model each made a first contact on all six, and neither made a legal learner return. The actual Pong-ranked point ended on the learner's half every time. Its selected dry trajectories moved farther away from the actor after reversal, reaching a median longitudinal position near y = −0.28 m by forecast frame 24, compared with roughly −0.71 m for the shuffle. Matching first candidate digests confirmed the same proposal set. All 12 point outcomes and referee tapes replayed exactly. [Pong-only physical receipt](value-live.json.gz)

## Calibration from earlier physical experience

The calibration donor replayed previously issued commands into an independent nominal arm. At each archived choice, the chronological native RGB supplied the initial mover state. The recorded command tape supplied the continuation. Generic local 3D positions, velocities and curvature joined the actor-relative descriptor. The scalar label was the actual learner return count attributable to that shot, plus 0.30 times the log of one plus actual survival seconds after frame 44. The archive includes shots from the earlier geometrically ranked behavior policy and from the Pong-only development points. Those behavior selectors provided past experience. Every live ranker in this study used its learned value on the shared proposal set.

The calibration fit 115 shots from whole earlier launch seeds, including 86 legal returns and 29 nonreturns, and assessed another 24 shots from held-out archive seeds. A fixed weak Pong prior contributed 0.12 times its standardized value to the hybrid. A target model fit the residual. The target-only and shuffled branches used the same descriptors and regularizer. Every fresh evaluation launch was disjoint from the archive. [Calibration receipt](value-calibration.json.gz)

| Ranker | Archived held return AUC | Archived held reward Spearman |
|---|---:|---:|
| Target + weak Pong prior | .871 | .599 |
| Target-only | .871 | .597 |
| Pong-only | .214 | −.523 |
| Target labels shuffled | .707 | .351 |

## Fresh calibrated physical play

All four rankers used identical first proposal sets on four disjoint native points against the strict physical anchor opponent. The hybrid and target-only models chose the same first move every time and converted the same point. The shuffled model converted a different point. Across this bank, archival return discrimination carried over to one first-return success and yielded no advantage over shuffled labels. All 16 action tapes, terminal outcomes and referee events replayed exactly. Every physical cohort used a native stereo receiver and the same slowed launch family with .30 of standard launch spread and .82 of speed. [Physical receipt](value-calibrated-live.json.gz)

| Ranker | First contacts | Points with first legal return | Points with a second legal learner return | Longest alternating run | Points with 10 alternating returns | Wins |
|---|---:|---:|---:|---:|---:|---:|
| Target + weak Pong prior | 4/4 | 1/4 | 0/4 | 2 | 0/4 | 0/4 |
| Target-only | 4/4 | 1/4 | 0/4 | 2 | 0/4 | 0/4 |
| Pong-only | 4/4 | 0/4 | 0/4 | 0 | 0/4 | 0/4 |
| Target labels shuffled | 4/4 | 1/4 | 0/4 | 2 | 0/4 | 0/4 |

The largest target-only prediction in the first candidate selections was 2.52, above the largest archived training reward of 1.66. That launch ended on a long-ball fault. The Pong-only branch again ended on the learner's half on every launch. The archive measured trajectories that earlier policies chose to execute. The live trial measured the result of maximizing the value over up to 112 alternative nominal trajectories, where the observed extrapolation and the arm/ball forecast can both affect the chosen action.

## Bounded value from the same earlier experience

A Gaussian average of earlier measured rewards replaced the quadratic score. The model projected standardized path features into eight unsupervised principal components. Its bandwidth came from the median fifth-nearest training example. A small zero-reward prior lowers the estimate in regions with sparse examples. The bounded estimate shrinks smoothly without adding an acceptance rule. The control used the same feature transform and bandwidth with permuted rewards. On archived held data, real versus shuffled labels gave return AUC .871 versus .721 and reward Spearman .693 versus .286. Both used the shared candidate motor bank that includes the Pong lateral option. This bounded ranker used target outcomes alone as labels.

The first bounded trial drew four fresh matched starts. The learned ranker yielded 7 learner returns and 7 rival returns, compared with 4 and 4 for the shuffle. The alternating sequences by launch were 0, 4, 2, 8 versus 0, 2, 0, 6. All eight action tapes replayed exactly. Each ranker connected on all four starts. [Bounded trial receipt](value-supported-live.json.gz)

| Bounded pilot ranker | Points with first legal return | Points with a second legal learner return | Longest alternating run | Points with 10 alternating returns | Wins |
|---|---:|---:|---:|---:|---:|
| Measured rewards | 3/4 | 2/4 | 8 | 0/4 | 0/4 |
| Shuffled measured rewards | 2/4 | 1/4 | 6 | 0/4 | 0/4 |

An independent eight-start replication used the same frozen model. The measured model landed 16 learner and 16 rival returns. The shuffled model landed 14 learner and 11 rival returns, and won three points. The longest alternating sequences were eight and six respectively. All 16 points replayed exactly. [Replication receipt](value-supported-replication.json.gz)

| Bounded replication ranker | Points with first legal return | Points with a second legal learner return | Alternating returns by launch | Points with 10 alternating returns | Wins |
|---|---:|---:|---|---:|---:|
| Measured rewards | 8/8 | 5/8 | 8, 4, 4, 2, 6, 4, 2, 2 | 0/8 | 0/8 |
| Shuffled measured rewards | 7/8 | 6/8 | 3, 2, 4, 4, 6, 3, 3, 0 | 0/8 | 3/8 |

## Updating from ongoing play

The online ranker starts from the frozen bounded model and stores only its own selected forecasts. When the native point ends, it assigns each preceding forecast the learner's actual return count before the next stroke and its observed remaining point time. It appends those examples before the next start. The new cases receive weight four relative to one for each archived case. The unsupervised metric and bandwidth are determined by the donor fit. The control starts from the same parameters and retains them for the independent matched sequence. The receipt stores the chronological updates, their wall time, each full scored point, and the reloadable terminal model. [Online runner](../../scripts/experimental/tt22_value_online.py)

The policies issued the same commands and earned the same referee events before the first update. Over eight complete points, the learner added 17 selected shots to the 115 archived examples. The update computation took 12 ms in total, excluding play. Updating lengthened four paired rallies and shortened two. The online policy made a second learner return in two points, versus three for the frozen policy. Both connected on all eight starts. All 16 points replayed exactly, and replaying the updates recovered the saved model. [Online receipt](value-online.json.gz)

| Ranker | Learner returns | Rival returns | Points with first legal learner return | Alternating returns by launch | Points with 10 alternating returns | Wins |
|---|---:|---:|---:|---|---:|---:|
| Online | 10 | 9 | 4/8 | 0, 8, 0, 0, 0, 2, 7, 2 | 0/8 | 1/8 |
| Frozen | 7 | 6 | 3/8 | 0, 3, 6, 4, 0, 0, 0, 0 | 0/8 | 1/8 |

## Changing the optical flight estimate

The flight-fit comparison freezes the measured-reward ranker in both branches. Both read the same calibrated stereo RGB, know the factory body and use command history to initialize the independent arm model. They use the same crossing depths, stroke geometry, motor-rate limits and candidate cap. The fitted branch uses up to 21 past optical positions and a nominal ball-and-table model to estimate velocity and spin. It keeps the currently measured position and falls back to the stock estimate when the fit lacks support. The changed estimates can produce different motor proposals. The bounded learned model ranks the resulting trajectories in both branches. [Flight-fit runner](../../scripts/experimental/tt22_value_inverse.py)

The fit was used in five of nine selected strokes. The fitted estimate shortened four paired rallies and tied two. Both branches made contact in each point and won twice. All 12 points replayed exactly. [Flight-fit receipt](value-inverse.json.gz)

| Flight estimate | Learner returns | Rival returns | Points with first legal learner return | Alternating returns by launch | Points with 10 alternating returns |
|---|---:|---:|---:|---|---:|
| Inverse fit | 4 | 3 | 3/6 | 0, 0, 0, 2, 1, 4 | 0/6 |
| Stock | 15 | 14 | 4/6 | 6, 0, 0, 7, 8, 8 | 0/6 |

## Revisiting wide returns

The two eight-return points stopped after wide far-arm replies. Stereo observed the next crossing at about 40–44 cm from center, outside the standard 33 cm proposal range. On those two points and another development stall, an extra motor deck becomes available when the standard generator has no proposals. It uses the same nominal flight and body models with a wider lateral search, shallow blade orientations and small offsets inward from contact to use the blade's edge. The frozen model still ranks the resulting trajectories. The replay checks that the earlier issued commands match the recorded point through the frame before the first new proposal. These openings were chosen after inspecting the earlier results. [Wide-return runner](../../scripts/experimental/tt23_value_wide.py)

The arm issued the extra swing in all three points. The lengths stayed at 8, 8 and 4, and the extra swings made no additional contact. All six points replayed exactly. The nominal pose solver found candidate blade positions within 4 mm of the proposed static anchors on the two eight-return points. At the normal decision time, none of the 112 simulated swings on either point reversed the ball. Simulating one of those decisions five frames earlier yielded 44 reversals and one crossing to the far half. The ranker's selected forecast reversed the ball and stopped short of the net. [Wide-return receipt](value-wide-development.json.gz)

## Preparing the blade earlier

The highest-scoring native Pong coefficient vector supplies continuous lateral motion in the ready phase. It combines the reconstructed blade location, the current stereo ball location and the lateral crossing forecast while the inbound ball is between y = +.62 m and −.30 m. The factory inverse kinematics maps this request to a blade ready pose at the usual depth and height, with a lateral target capped at 29 cm and rate-limited joint commands. Both branches keep the same subsequent value model and late stroke search, including the duration-based proposal variants and the edge deck. The point-score-trained coefficients enter only the ready motion. Evaluator blade and ball positions are logged separately from the controller's RGB inputs. [Ready-pose runner](../../scripts/experimental/tt23_value_ready.py)

On the same three archived wide feeds, preparation was enabled after the last far-arm contact. The earlier issued commands matched each recorded point. At the usual late decision, the blade had moved 15.1, 24.7 and 28.9 cm laterally. It remained near the center in each stationary replay. The preparation reached two of the three wide feeds and neither new contact produced a legal return. The alternating lengths stayed at 8, 8 and 4. All six points replayed exactly. [Ready-pose development receipt](value-ready-development.json.gz)

In three fresh complete receiving points, Pong preparation produced alternating lengths of **0, 10 and 4**, compared with **2, 4 and 2** for the stationary control. The near and far arms each scored seven legal returns with preparation and four in the stationary condition. Both conditions lost all three points. All six points replayed exactly. [Three-start receipt](value-ready-fresh.json.gz) · [Matched native replay](value-ready-matched.mp4) · [Video receipt](value-ready-matched.json)

### Fixed validation

Across twelve fresh matched starts, Pong preparation lengthened one rally, shortened seven and tied four. Both conditions made a first legal learner return on 10/12 starts. The learner returned a second ball on 6/12 starts with Pong preparation and 9/12 with stationary preparation. Both had zero ten- or twenty-return points. All 28 ordinary and mirrored points replayed exactly. [Frozen design](value-ready-validation-design.md) · [Block 1 receipt](value-ready-validation-1.json.gz) · [Block 2 receipt](value-ready-validation-2.json.gz)

| Block and branch | Learner returns | Rival returns | Alternating returns by launch | Points won |
|---|---:|---:|---|---:|
| 1 · Pong | 11 | 11 | 2, 2, 6, 4, 6, 2 | 0/6 |
| 1 · Stationary | 17 | 17 | 6, 4, 8, 6, 6, 4 | 0/6 |
| 2 · Pong | 8 | 8 | 4, 0, 0, 4, 6, 2 | 0/6 |
| 2 · Stationary | 10 | 8 | 4, 5, 0, 0, 7, 2 | 2/6 |
| Total · Pong | 19 | 19 | Longest 6 | 0/12 |
| Total · Stationary | 27 | 25 | Longest 8 | 2/12 |

The prepared branch started moving a median 8.5 frames before late stroke selection among 30 selected strokes with preparation. It spent 1.35 seconds computing preparation across the twelve points, excluding the late ranker and rendering. At the first late decision, the median absolute blade offset from center was 1.5 cm. The late selector chose a different initial pitch in nine pairs. The development replays showed a larger change in blade position and improved contact on wide feeds. The fixed fresh comparison showed fewer sustained returns with the same preparation policy.

The four mirrored-control points lasted 8, 4, 0 and 4, versus 2, 6, 0 and 6 for Pong preparation and 4, 6, 5 and 7 for stationary preparation. They produced eight learner returns, compared with seven and twelve respectively. The cumulative absolute preparatory joint-rate effort was 50.9 for mirrored and 55.5 for Pong preparation, in sums of normalized per-frame axis commands. The direction comparison varied by launch, and stationary preparation had the most learner returns. [Mirrored schedule and receipts](value-ready-validation-design.md)

## Reproduction

Run from the repository root using its Python environment. Source fitting replays the archived native duration policies. The first physical run supplies deliberate failed-shot examples to calibration. The calibrated and supported runs draw fresh launch banks and reject overlap with their archived inputs.

```sh
tt22_native=$(mktemp -d)
c++ -O3 -std=c++17 -shared -fPIC scripts/experimental/tt3_components.cpp -o "$tt22_native/components.so"
export TT3_COMPONENT_LIBRARY="$tt22_native/components.so"
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_value_source
python -m scripts.experimental.tt22_value_source --extended
python -m scripts.experimental.tt22_value_trial --count 6
python -m scripts.experimental.tt22_value_calibration
python -m scripts.experimental.tt22_value_trial --calibrated --count 4
python -m scripts.experimental.tt22_value_trial --supported --count 4
python -m scripts.experimental.tt22_value_trial --replication --count 8
python -m scripts.experimental.tt22_value_online --count 8
python -m scripts.experimental.tt22_value_inverse --count 6
python -m scripts.experimental.tt23_value_wide
python -m scripts.experimental.tt23_value_ready
python -m scripts.experimental.tt23_value_ready --fresh 3
python -m scripts.experimental.tt23_value_ready --block 1
python -m scripts.experimental.tt23_value_ready --block 2
python -m scripts.experimental.tt22_value_movie artifacts/tt22/value-ready-fresh.json.gz \
  --mode pong --compare stationary --seed 488245766 --minimum 10 \
  --output artifacts/tt22/value-ready-matched.mp4
```

To verify a saved physical receipt without rerunning ranker inference, pass its path to `python -m scripts.experimental.tt22_value_trial --verify`. The replay executes both sides' archived commands and compares the exact strict referee reward, terminal outcome and event log.
