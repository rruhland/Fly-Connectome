# A source-calibrated readout for wide and ordinary releases

The encoder-conditioned readout retained the original policy on all 24 fresh ordinary releases across the two arms. It reached a second rebound on 9 of 24 wide releases, matching the independently evaluated broad-trained policy's count. The original head reached no second rebounds on those wide releases.

| Fresh cohort | Actor | Second qualifying rebound | Mean consecutive | Longest |
|---|---|---:|---:|---:|
| Wide stock | Context-conditioned | 3/12 | 1.25 | 4 |
| Wide stock | Original | 0/12 | 0.83 | 1 |
| Wide stock | Broad-trained | 3/12 | 2.00 | 13 |
| Ordinary stock | Context-conditioned | 11/12 | 13.00 | 18 |
| Ordinary stock | Original | 11/12 | 13.00 | 18 |
| Ordinary stock | Broad-trained | 9/12 | 2.00 | 5 |
| Wide altered | Context-conditioned | 6/12 | 1.67 | 6 |
| Wide altered | Original | 0/12 | 1.00 | 1 |
| Wide altered | Broad-trained | 6/12 | 2.00 | 6 |
| Ordinary altered | Context-conditioned | 8/12 | 3.42 | 12 |
| Ordinary altered | Original | 8/12 | 3.42 | 12 |
| Ordinary altered | Broad-trained | 11/12 | 3.67 | 13 |

The interpolation shortened some wide runs. On stock seed 890891530 the broad-trained policy made 13 consecutive rebounds and the 93% blend made 4. The conditioned stock policy preserved all ten ordinary streaks of at least ten from the original head. Ordinary altered preservation matched the original action tape on every seed, where the broad-trained head had more second rebounds.

The initial encoder angles go through the existing public vendor-kinematic estimator. The two thresholds came from 192 unscored ordinary and 192 unscored wide resets across three source donor morphologies. The blend is zero below the largest estimated ordinary tilt, 3.176°, and reaches one at the first quartile of the wide source distribution, 8.353°. A cubic smoothstep of the horizontal normal magnitude interpolates the two sealed parameter matrices in between. That single matrix then controls the entire run using the unchanged live wrist observer and own joint encoders. Every action follows feedback from its current observation.

This test used the original bounded training pass and an additional unscored source encoder sample. The four held sets were matched across policies and disjoint from the earlier physical tests and gate calibration. There was no target fitting or model selection. All 101 distinct held action tapes replayed exactly. When the blend was exactly zero or one, the conditioned matrix equaled the corresponding reference and reused its verified action tape.

Reproduce with `MUJOCO_GL=egl python -m scripts.experimental.tt24_tilted_start_context`. [Original broad training report](tt24_tilted_start_report.md)
