# Transient joint feedback on difficult donor releases

A second observation-dependent motor readout decays over 16 ticks at the start of a release. Five donor bodies supplied 500 screening, 8,976 training and 3,600 validation drops. Half of the training starts in each sampled donor batch came from the lower-scoring 45% of its screening distribution. Source validation utility increased from 0.420 to 0.531. The original motor head still controlled steady play. [Source code](../../scripts/experimental/tt24_acquisition.py) · [Source training and selection](acquisition-training.json.gz)

| Target, 24 fresh starts each | Original sum of longest | Transient feedback | Same correction permanently active | ≥10, original to transient |
| --- | ---: | ---: | ---: | ---: |
| Stock | 390 | 429 | 60 | 15 to 16 |
| Arm with tilted elbow and altered gravity | 114 | 89 | 35 | 4 to 3 |
| New geometry and dynamics | 147 | 113 | 55 | 6 to 4 |
| Widely varied starting joint angles | 34 | 33 | 22 | 1 to 0 |

Only stock improved. The original head remains the general reference. The target learner used wrist images and encoders, with zero target motor practice. Every physical action tape reproduced its referee outcomes. [Stock](acquisition-stock.json.gz) · [Tilted](acquisition-tilted.json.gz) · [New geometry](acquisition-novel.json.gz) · [Wide poses](acquisition-wide.json.gz) · [Checkpoint](acquisition-source.npz)
