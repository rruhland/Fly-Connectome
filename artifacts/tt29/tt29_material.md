# Camera-based material adaptation in regulation rallies

The adaptive controller landed five learner returns across three shifted points, compared with six for the frozen controller. Both controllers reached the six-return cap on all three stock points. The adaptive controller gained one legal learner return on launch 7271 after the material shift and lost one on each of the other two launches. [Live counts, commands and native contacts](material-adaptation.json.gz)

| Plant and controller | Live points | Learner returns | Alternating legal returns | Points lost | Points reaching the cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock, frozen | 3 | 9 | 18 | 0 | 3 |
| Stock, adaptive | 3 | 9 | 18 | 0 | 3 |
| Shifted, frozen | 3 | 6 | 12 | 3 | 0 |
| Shifted, adaptive | 3 | 5 | 10 | 2 | 1 |

## Matched physical points

Each branch played the same three measured openings with the same opponent seeds and the same 40 saved stroke examples. Both branches received 60 Hz external stereo RGB and near-arm encoders. The examples were frozen during the assay so changes in action memory could not explain the contrast. Each point ended at the native referee's decision or after six alternating, physically struck legal returns. The initial recorded opening was excluded. Launches 7271 and 7345 also appeared in the checkpoint course. Launch 4527 was new to that course. The adaptive 7271 command tapes reproduced every native event on both plants. [Camera checkpoint](camera-action-memory.json.gz) · [Experimental code](../../scripts/experimental/tt29_material.py)

The shifted fixture increased ball mass by 8%, table damping from 24 to 31 and paddle damping from 25 to 33. It reduced paddle friction from 0.54 to 0.47. The adaptive pupil inferred a table coefficient from recent stereo rebound fits, then used it in the independent dynamics model for planning. It also reranked up to eight candidate strokes under two possible paddle responses. The learner received camera frames and encoders. The evaluator applied the material changes and read the native referee events for scoring. [Perturbation definitions](../../scripts/experimental/tt29_conditions.py) · [Experimental code](../../scripts/experimental/tt29_material.py)

## What the images identified

The frozen branches estimated median vertical rebound ratios of 0.820 on stock and 0.817 after the shift. The isolated native calibration predicts about 0.965 and 0.940 at a 3 m/s normal impact. The fitted damping saturated near 42 in both branches. The same optical estimator that predicts usable incoming flights gave closely overlapping observations for these two table settings. Its instantaneous bounce fit and the finite native contact measure differ enough to swamp the material signal in this assay. The return comparison tests the combined table fit and paddle reranking. [Optical samples and native event tapes](material-adaptation.json.gz)
