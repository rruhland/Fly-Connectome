# 67 consecutive arm returns against the reactive random opponent

The centered table-clamped SO-101 played 109.04 seconds from one physical launch and made **67 legal arm returns**. The far paddle returned 68. All 6,543 issued joint commands were replayed independently. The replay reproduced every referee event, checked all 69 requested goals against the continuous random draw law and found no CAD penetration in 52,337 physics substeps. Minimum clearance was 15.19 mm. Median outgoing speed across the two players was 5.22 m/s. The last arm contact sent the ball onto its own half. [Command tape](live-record-1-67-tape.json.gz) · [Independent physical replay](live-record-1-67.json)

This was the first assigned point in a prospective 96-game course. We chose the motor memory immediately *before* the earlier 63-return game because that memory also performed well on 22 other releases. The first 24 games here use new measured launch recordings, each with its own independent opening and reply streams. The two fixed external cameras and joint encoders drive the arm. Its incoming-ball estimate, earned motor memory and calibrated physics model contribute to the actions. This opponent draws continuously across the established reachable region and physically reacts to each actual stroke. [Course and exclusions](protocol.json) · [Starting memory and prior validation](../tt51_point_transfer/README.md)

| Same starting motor memory and same opponent stream | Own returns | Weight updates during the point |
| --- | ---: | ---: |
| Frozen | 15 | 0 |
| Learning | 67 | 67 |

The two arms issued exactly the same first 1,067 commands and had already made eleven returns before the first command changed. Both returned the next physically identical ball, then their different strokes produced different rallies. The learning arm retained 288 rewarded motor segments during its point. It returned both low-wing arrivals. It received no low-central arrival. The result establishes that updating inside this particular point changed its length. Whether the resulting skills persist across games is measured on the remaining course. [Matched point and coverage](first-point-learning.json) · [Frozen actions](first-frozen-tape.json.gz) · [Motor memory after the rally](after-first.npz)
