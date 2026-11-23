# TT24 direct joint control and paddle bouncing

The end-clamped SO-101 moves through five continuously commanded joints. The jaw keeps its grip. The physical judges count legal table-tennis returns and separated upward paddle bounces. All learner updates in this pass use non-backprop methods.

| Question | Most informative result | Evidence |
| --- | --- | --- |
| Can a pupil control table tennis without a supplied stroke at execution? | After 48 earlier demonstration points, a direct stereo pupil made a second legal return on 8/24 new starts and five on 2/24. Both fifths made ten alternating returns. The pupil controlled every frame. | [Report](rally-bank.md) · [Watch ten](rally-bank-held.mp4) |
| Can it identify a changing body quickly? | On 32 fresh streams with mixed motors, different loads and latency, the learner restored accurate five-joint reaching on 31/32 under each change. Typical recovery was 3.2 simulated seconds. | [Body and game scores](body.md) |
| Does experience with one camera aid another? | A single paired calibration clip let a wrist-only pupil reuse an outside-camera policy and make the second return on 3/16 new starts with an angular-size cue. | [Cross-eye tests](eyes.md) |
| Can wrist control improve repeated bouncing through reward? | A direct five-axis reward-trained wrist policy averaged 3.36 rebounds on fourteen fresh drops, versus 2.50 stationary and 2.43 blind. Earlier target practice cost 216 scored drops plus 10 motor probes. | [Training and replay](motor-results.md) · [Watch the paired control](motor-position-witness.mp4) |
| Does identifying the ball depend on its paint? | A motion-based front end with the frozen control readout averaged 3.79 rebounds, reaching eight. A cyclic RGB permutation preserved every action on fourteen drops. Eye-to-blade association remained weak. | [Anonymous vision](tt24_anonvision.md) |
| Does native Pong help the changed arm learn? | In a locked 24-drop replication on the stock arm, a Pong-informed wrist update earned 84 rebounds, versus 61 with flat credit, 54 with shuffled credit and 79 with the earlier arm frozen. The arm had extensive prior target practice. | [Intervention receipt](recurrence-arm.json.gz) · [Fresh replication](recurrence-arm-replication.json.gz) |
| Is the apparatus physically capable of longer runs? | A privileged physical controller sustained 14–15 on all sixteen stock drops, and ten or more on fourteen of sixteen alternate-arm drops. The separate short-arc control reached 20 and 19. | [Fixture ceiling and traces](bounce-ceiling.md) |

The reward-only policies, demonstration pupils, empirical body decoder and privileged controls have different training histories. A short route from native Pong alone to consistently long wrist-only runs across hardware remains open. [Prospective comparison definitions](design.md)
