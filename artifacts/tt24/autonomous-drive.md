# Discovering a motor change without a switch announcement

Four streams of 16 wrist games switched between stock and altered physical bodies, changed command mixing and later restored earlier interfaces. In two stages the wiring could switch during a release after 70 control ticks. A persistent monitor compared measured joint angles with an estimate from the current command map. After four frames of sufficiently unexpected movement, the arm could inspect its motors at the next break. The setup bench preserved the new mapping and physical body. The arm then resumed scoring drops with its original frozen wrist motor skill. Detection and identification read own commands and encoders only. No switch signal, map identity or wiring matrix entered them. The initial wrist image still selected between two previously calibrated optical models. [Monitor and paired physical runs](../../scripts/experimental/tt24_autonomous_drive.py) · [Inspecting the five motors](motor-settle.md)

| 64 sequential releases | Actual rebounds | Releases with ≥10 consecutive | Charged inspection time |
| --- | ---: | ---: | ---: |
| Recalibrate after a prediction error | 594 | 24 | 18.6 s across 18 inspections |
| Also recognize earlier maps during play | 595 | 24 | 11.6 s |
| Frozen initial map | 376 | 14 | 0 |
| Correct inverse supplied on every frame | 938 | 42 | 0 |

Recognizing an old map from its recent ordinary movements reduced setup time by seven seconds while maintaining the aggregate score. The simpler monitor earned 155 rebounds during the four streams' three-drop return to the first changed mapping, versus 15 frozen. On the final two drops after returning to the original interface the monitor earned only 46 versus 112 frozen. The monitor often needed a failed release before an inspection restored control. The result combines data across physical games and a separate, charged, contact-free inspection opportunity. [In-play recall implementation](../../scripts/experimental/tt24_inplay_bank.py) · [Example complete course](autonomous-drive-0.json.gz) · [Same course with in-play recall](inplay-bank-0.json.gz)

## Can previous Pong play teach when to inspect?

Four native Pong learners selected how much unexpected displacement to tolerate before investigating a possible control reversal. Actual native point rewards favored relatively high thresholds because every probe interrupted live play for half a second. On 200 held native points the fitted policy scored −93 net points versus −161 with shuffled source rewards and −138 with the fixed arm threshold. Fixed high thresholds of 2.3 and 3.3 scored −92 and −95. [Native source study](pong-inspection.md)

A fresh four-stream arm comparison applied the native tolerance directly using RMS encoder error in units of the permitted movement per frame. Each trained source threshold and its shuffled control went into its own stream. Inspection could be deferred to the interval between drops. The true native fits triggered only two inspections on 64 releases. Three native streams completed without making any inspection. Scores favored the more responsive alternatives.

| Arm policy | Actual rebounds | ≥10 consecutive | Inspections |
| --- | ---: | ---: | ---: |
| Native learned tolerance | 257 | 11 | 2 |
| Shuffled source tolerance | 468 | 22 | 31 |
| Previously used arm threshold | 422 | 20 | 20 |
| Fixed high tolerance or never inspect | 302 | 13 | 0 |

The arm has different tracking errors and a different cost for inspecting. The same units of permitted command travel did not make this decision rule transferable. Every distinct decision path yielded actual closed-loop play. When two branches chose the same motor matrix on the same release, their monitors could share the exact causal encoder and command history. The 129 unique physical games reproduced all encoder readings and physical referee events on replay. [Arm test](../../scripts/experimental/tt24_native_inspect_arm.py) · [Stream 1](native-inspect-arm-0.json.gz) · [2](native-inspect-arm-1.json.gz) · [3](native-inspect-arm-2.json.gz) · [4](native-inspect-arm-3.json.gz)
