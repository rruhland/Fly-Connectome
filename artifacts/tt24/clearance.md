# Whole-body contacts on the stock SO-101

All 48 archived command tapes from the clean-state gantry carry audit replayed exactly with additional contact logging. Terminal collisions were grouped by the actual contacting geometry. [Substep contacts, states, and exact replay evidence](clearance-contacts.json.gz)

| Archived policy, 16 drops each | Arm link | Hand or jaw | Paddle handle | Camera mount | Table clamp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original analytic prior | 0 | 2 | 1 | 1 | 2 |
| Gantry-refined prior | 1 | 2 | 0 | 4 | 3 |
| Stationary | 0 | 0 | 0 | 0 | 0 |

All 16 obstruction stops occurred on a descending flight following a blade contact. Their median incoming vertical speed was about −2.37 m/s. The median ball distance from the blade axis at these stops was 142 mm for the original prior and 179 mm for the gantry refinement, against a 76 mm blade radius. All 16 stationary tapes ended at the table.

From the first blade touch to the next contact, on intervals of at least 65 ms, median horizontal ball travel was 95 mm with the original prior, 142 mm with the gantry refinement, and 28 mm with the stationary arm. At first touch the median blade tilt was 5.0°, 4.8°, and 0.9°, respectively. The active modes therefore exposed the body to a substantially wider return corridor. This was measured before the terminal obstruction could influence the ball.

## Local clearance correction

The correction uses the fused wrist RGB observer, reported joint angles and speeds, and a separate static vendor and tool geometry proxy. It projects the visible sphere forward at two short horizons. When the predicted gap to a moving surface is below 14 mm, it tries a small joint-target correction with a preference for preserving the requested blade pose. The live fixture and referee stayed identical. The clean-ball comparison supplied true sphere position and velocity to the correction alone and used the same public policy for the requested blade pose. [Servo implementation](../../scripts/experimental/tt24_clearance.py)

| New paired stock drops, 16 per condition | Mean rebounds | Max consecutive | Body, handle, or fixture stops | Handle stops |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 2.63 | 5 | 1 | 1 |
| Public servo | 1.44 | 4 | 3 | 2 |
| Public servo with clearance correction | 1.38 | 4 | 4 | 1 |
| Clearance correction with clean obstacle state | 1.44 | 4 | 1 | 0 |

The public correction intervened 15 times across seven drops. One handle stop became a camera-housing stop, and one table stop became a clamp stop. With clean obstacle state, both handle stops became table stops. The rebound mean stayed at 1.44 with clean obstacle information. The observed lateral return drift remained about 11 cm in the active servo conditions and about 3 cm with the stationary arm. [Paired data, corrections, and all 64 exact action replays](clearance-servo.json.gz)
