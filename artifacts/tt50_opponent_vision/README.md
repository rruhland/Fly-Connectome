# Can the arm see where the opponent will hit before contact?

Each of 24 previously recorded outgoing shots was replayed from the same physical state six times. Only the opponent's four continuously drawn aim quantities changed. The two RGB cameras saw the resulting paddle movements. Future supervision was the ball later observed in stereo on its way back. Sixteen entire releases trained the readout and eight different releases tested it. The actuator never received an aim, simulator position, segmentation buffer or a future frame. The researcher chose fixed far-court image windows and a low-chroma foreground filter for this camera rig. [Physical forks](observations) · [Full held evaluation](evaluation.json)

| Outgoing ball's location | Median time before opponent contact | Ball-only lateral error | With opponent image | Same outgoing ball with mismatched opponent image |
|---|---:|---:|---:|---:|
| −0.45 m | 0.60 s | 12.0 cm | 13.3 cm | 19.4 cm |
| 0.00 m | 0.45 s | 12.5 cm | 14.1 cm | 20.9 cm |
| +0.45 m | 0.29 s | 12.3 cm | 11.2 cm | 17.7 cm |
| +0.85 m | 0.12 s | 12.4 cm | 19.1 cm | 26.5 cm |

The images contain genuine information about differences between replies to the same ball. At the middle gate, subtracting the mean within each matched outgoing situation reduces the lateral error from 11.5 to 8.2 cm when the incoming stroke is visible. The absolute figures in the table include error in that mean. The first two test releases looked substantially better than the full eight. Choosing silhouette features and regularization using only grouped practice folds yielded 11.3 cm at the earliest gate and 9.1 cm at +0.45 m. Early height estimates seldom recognized the most difficult low returns. [Grouped practice selection and held results](group-evaluation.json)

A native intervention tested whether this visual signal could choose an earlier receiving posture. Across six matched offers, stock and three integrations all returned the same three. Earlier preparation was physically valuable on two taught balls. Starting from a previously demonstrated posture, a late camera-driven joint readout returned both and the opponent answered both. The stock finisher missed both from those same postures. Acquiring the suitable posture from early visual evidence is the unresolved part. [Physical results and commands](../tt50_conditional_prep/README.md)

The observational timing test follows work on anticipating a table-tennis opponent from partial racket motion. The comparison of early commitment against waiting matters because moving to the wrong side carries a motor cost. [Wang et al., Learning Anticipation Policies for Robot Table Tennis](https://publications.ri.cmu.edu/learning-anticipation-policies-for-robot-table-tennis) · [Anticipatory action selection for human–robot table tennis](https://www.sciencedirect.com/science/article/pii/S0004370214001398)

## What the camera can learn before impact

A stereoscopic view of the far paddle and the earlier ball path supplied an advance estimate of the future return. On 50 new replies from selected difficult source strokes, six arrived both low and left. The earliest gate placed three of the six in its highest-risk quarter. Successive gates placed four and five there as the far paddle moved. On 48 ordinary fresh replies there was just one low-left case. The calibration model was frozen using 35 ordinary source games and separate difficult source contexts. [Report](risk-evaluation.json) · [Frozen coefficients](risk-model.npz) · [Difficult source roster](difficult-plan.json)

Learning from the later camera observations, including misses, improved the receiving-location forecast on a separate 24-game course. At the gate near 0.45 seconds before opponent contact, the arm had 203 forecasts before contact. The carried online calibration reduced mean lateral error from 11.6 to 9.4 cm and height error from 5.6 to 4.1 cm. Calibration started in those 24 games alone reached 9.8 and 4.7 cm. Two predictions made after far contact were excluded from the pre-contact metric. Improved forecasts still have to support useful motor decisions in actual games. [Per-gate and per-stage errors](continual.json) · [Practice camera observations](practice-episodes) · [New-games camera observations](untrained-episodes)
