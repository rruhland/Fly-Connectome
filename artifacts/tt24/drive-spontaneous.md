# What ordinary play reveals about five unfamiliar actuators

Four wrist-play streams exposed the portable carried arm to a previously unseen dense five-channel command mapping, introduced and restored during ongoing releases. The arm received its ordinary wrist image and encoders. Its initial postures came from the ordinary, nearly level distribution. The same original feedback head chose every nominal five-joint movement before and after each switch. One arm observer learned from ordinary motions using fast and retained forward models fitted to earlier contact-free and stock trials. A second observer fitted single-frame physical command estimates to encoder transitions from 24 generic 100-tick contact-free recordings and assessed them on eight further recordings. It added brief independent exploration when it detected a mapping error. [Carried arm](carried-wrist.md) · [First actuator learner](../../scripts/experimental/tt24_carried_drive.py) · [Single-frame alternative](../../scripts/experimental/tt24_motor_innovation.py)

A direct inverse fitted from single-frame observations was too noisy. Held estimates had 0.21–0.24 RMS normalized action error per axis. The exploratory method repeatedly misidentified the same actuator interface, storing 8–12 supposed mappings per stream despite only two actual interfaces. This cost the arm even the original skill. The older encoder learner made occasional recoveries on the remapped arm, but its restored scores also lagged. The point scorer and complete contact histories reproduced under replay in all four streams for both learners and controls.

| Actual scored rebounds in four streams | First fully changed pair of drops | Second fully changed pair | Two fully restored drops |
| --- | ---: | ---: | ---: |
| Learned from ordinary movements | 8 | 33 | 33 |
| Single-frame command identification | 5 | 6 | 4 |
| Frozen motor decoding | 7 | 8 | 116 |
| Supplied wiring inverse | 156 | 159 | 116 |

The receiving policy had been trained to bounce on three different donor arms. The earlier four-demonstration arm had learned its wiring more readily under these tests. Rapid recognition in a weaker motor policy did not carry over automatically to the stronger readout. [Earlier in-flight motor recall](body.md#switching-motors-during-an-ongoing-bounce) · [Stream 1](carried-drive-0.json.gz) · [2](carried-drive-1.json.gz) · [3](carried-drive-2.json.gz) · [4](carried-drive-3.json.gz)
