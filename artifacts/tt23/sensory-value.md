# A Pong-trained value for observed movement

Short visual transitions predicted remaining duration in held native Pong. The same value had no eligible matched arm comparison at the intended feedback frame. At the secondary frame it ordered the physical jog options about as often as a tie. The assay ended before an online trial. [Design](sensory-value-design.md) · [Receipt](sensory-value.json.gz)

## Native Pong

The original RGB trace archive supplied six policies on each of 18 fit launches and 12 held launches. A delayed action probe identified the paddle from pixels. The model used seven-frame transitions with three paired sightings of the paddle and the moving region. Features measured the lengths and angles of observed motion and relative position. The label was the bounded remaining episode duration. A clock spline estimated elapsed-time effects, and a small ridge model learned the residual from vision. The shuffled control reassigned residual labels within elapsed-time bins. Each original launch had equal fitting weight. [Native trace archive](../tt22/value-source.json.gz) · [Implementation](../../scripts/experimental/tt23_sensory_value.py)

| Held source model | Per-launch mean absolute error | Same-launch, same-frame duration concordance |
|---|---:|---:|
| Observed sensory transition | .131 | .687 |
| Shuffled residual labels | .238 | .412 |
| Clock only | .168 | .500 |

The fit saw 1,836 complete observation windows. The held error used 1,792 windows. The paired ordering used 1,789 comparisons from eight held launches with outcome differences, averaged within launch first. A clock-only prediction ties alternatives at the same frame.

## SO-101 jogs

The four completed archives supplied 32 verified physical starts and all 96 distinct command tapes for their three registered jog alternatives. The same frozen value read the real stereo ball positions and the nominal paddle geometry produced by replaying issued joint rates. The fixed feedback window used frames 18, 21, and 24, just before the continuation controller took over. The secondary window used frames 24, 27, and 30. Each comparison required all three observations for all alternatives, with the same clock for every option. The target's legal returns and rally length entered only the evaluator. [Physical bank design](jog-bank-design.md) · [Receipt](sensory-value.json.gz)

| Window ending at | Observable option windows | Starts with every option visible | Starts with different learner returns | Sensory concordance | Shuffled | Clock |
|---|---:|---:|---:|---:|---:|---:|
| Frame 24 | 33/96 | 0/32 | 0 | — | — | — |
| Frame 30 | 70/96 | 11/32 | 5 | .467 | .567 | .500 |

At the primary landmark, 91 options had a sight at frame 18, 47 at frame 21 and 75 at frame 24. At frame 30 the five informative starts gave 11 legal-return comparisons. The model achieved the same concordance on strictly alternating rally length. Two comparisons on one start crossed the ten-return threshold. The sensory estimate ordered one correctly, and the shuffled estimate ordered both correctly. None of the 33 windows from complete frame-30 starts fell within the native reference support, compared with 1,681 of the 1,792 held Pong windows. Support uses the 95th percentile of training motion's nearest-neighbor distance across distinct Pong launches.

The primary gate required ten complete starts whose options differed in legal returns. Missing observations prevented that early comparison. The later measurements gave no ranking advantage. This tested a value of measured outcomes after an issued jog. A live selector would need such feedback to update after the selected action.

Run `python -m scripts.experimental.tt23_sensory_value` in the repository's MuJoCo environment. The receipt records the hashes and verified counts of the bank snapshots used.
