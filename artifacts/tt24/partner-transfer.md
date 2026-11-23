# Can physical rallies on three arms teach a fourth arm to bounce alone?

The source is a two-racket vertical volley game. A five-joint arm meets the ball from below. A second simulated racket tracks it from above, moves vertically, and exchanges momentum through physical contact. The learner earns a point each time its return reaches the upper racket after a legal near return. A second consecutive touch by either side ends play. Training therefore cannot earn credit for bouncing a ball twice on the same paddle. This source game is an intermediate transfer benchmark distinct from regulation table tennis and native arcade Pong. [Source arena](../../scripts/experimental/tt24_overhead.py) · [Fast two-racket referee](../../scripts/experimental/tt24_overhead_native.cpp) · [Independent physical replay](../../scripts/experimental/tt24_overhead_audit.py)

Both source arms started with zero motor weights and learned all five joint commands from rewards on three hinged donor designs. They received metric ball and paddle observations and 30 short body-identification probes. The first game used a 100 mm radius tracking opponent. The second used a 48 mm opponent with slower movement and a smaller lateral range. The opposing face had a different height on each release. The motor readout used the same relative position, velocity and nonlinear features as the donor-bouncing controller. Neither motor readout included a stroke path.

| Source and held target | Larger opponent | Smaller opponent |
| --- | ---: | ---: |
| Source reward practice episodes | 6,048 | 8,190 |
| Additional source validation episodes | 3,480 | 4,704 |
| Mean completed near/far pairs over 72 new source starts | 3.39 | 3.50 |
| Best completed near/far pairs | 10 | 15 |
| Source starts with at least three pairs | 48/72 | 45/72 |
| Stationary completed pairs | 0 | 0 |
| Stock solo wrist mean streak, 16 new drops | 1.31 | 1.63 |
| Matched stationary stock solo mean streak | 1.81 | 2.56 |
| Changed-body solo wrist mean streak, 16 new drops | 1.44 | 1.19 |
| Matched stationary changed-body solo mean streak | 1.63 | 1.50 |

The upper racket was removed for solo play, the ball was released from the standard solo distribution, and the pretrained motor head controlled every joint from the first frame. The target wrist observer used eight previously recorded unlabelled optical calibration drops. Neither target received motor practice. Perfect-state target diagnostics also struggled, averaging 1.06 and 1.38 on stock. This rules out wrist sensing as the sole cause on these starts. A Python evaluator replayed all 144 learned source trajectories from the original motor commands and independently reproduced near and far collision counts. Each of the 192 solo condition trajectories also replayed exactly. [Wide opponent training](overhead-source-training.json.gz) · [Stock and altered transfers](overhead-source-held.json.gz) · [Source replays](overhead-source-replay.json.gz) · [Small opponent training](overhead-precision-training.json.gz) · [Small opponent transfers](overhead-precision-held.json.gz) · [Small opponent replays](overhead-precision-replay.json.gz)
