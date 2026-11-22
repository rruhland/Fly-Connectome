# SO-101 sustained rally learning

A rally counts strict, consecutive legal returns from both robots after the serve. The best receiving controller uses native stereo RGB and issued motor commands. It estimates incoming flight, searches a nominal physics model for candidate strokes and can learn corrections from its own visible outgoing shots. The five-axis practice partner uses simulator state and a physical racket. All results on this page come from the simulator.

The longest verified rally has **46 returns, 23 by each robot**. A separate matched intervention changed the near arm alone: four practice points increased its legal returns from 22 to 55 on eight fresh starts. The wrist-only arm reached 19 and one external camera reached 25. [Watch 46 returns](tt23_sequence_coach_46.mp4) · [Watch the wrist-only player and its live view](single-eye-wrist-rally.mp4) · [Every start in the two practice comparisons](rally-practice.png)

| Question | Evidence |
|---|---|
| Does arm practice improve long rallies? | Seven optical corrections from four practice points took eight matched near-serve starts from 22 to 55 legal arm returns. Two matched points rose from 3 to 41 and from 1 to 37. [Point-by-point results](../tt22/lateral-rallies.md) |
| Does it happen in a complete match? | On the same first-to-11 schedule, practice increased legal arm returns from 19 to 28. The practiced arm reached 23, including 11 of its own. It lost 0–11. [Video and games](../tt22/ordinary-game.md) |
| Can the body adapt during a scored point? | Following a hidden motor remap, the wrist camera recovered the wiring in ten live control frames. The arm made 33 legal returns versus zero without the correction over 12 starts, reaching 10 and 14. Fixed stereo supplied the rally vision. [Remap study](../tt22/tt22_live_body.md) |
| What came from Pong? | On 176 disjoint matched opening serves, the same target-trained stroke touched 168 balls with Pong duration training and 65 with zero gains. Legal returns were 56 versus 36. [Source controls and learning curves](../tt22/source-transfer.md) |
| Does every phase benefit from that source policy? | Choosing each late stroke's workspace using native Pong did worse than holding the motor region near the arm. A transferred credit metric and a three-option cross-court selector also failed to speed up later strokes. [Lateral control](../tt22/lateral-rallies.md) · [Credit metric](source-metric.md) · [Cross-court learning](source-lanes.md) |
| Can a cheaper target motor learner improve late returns? | Under the same successful first stroke, a contextual scorer learned from physical branches earned a second return on 6/10 starts versus 1/10 for its context-free control. A separate ridge fit scored 5/8 on the learner's own state distribution and reached nine in a complete game. [First comparison](../tt22/branch-motor.md) · [Ordinary games and online-state curriculum](../tt22/tt23-motor-curriculum.md) |
| Do better camera predictions guarantee better shots? | The learned uncertainty model predicted held optical errors better and reduced actual near returns from 18 to 12 on six new starts. [Forecast and real-play tests](vision-and-risk.md) |

The outside work most directly relevant to these choices is Mülling and colleagues' demonstration of conditioning learned table-tennis strokes on the incoming ball; they used physical human demonstrations. Our experiments use recorded robot motor actions, optical observations and real simulated outcomes. Learning on states visited by the student also follows the distribution-shift concern described by Ross, Gordon and Bagnell. The local ridge and outcome learners here use no backpropagation. [Robot table tennis](https://journals.sagepub.com/doi/10.1177/0278364912472380) · [Learning on student states](https://proceedings.mlr.press/v15/ross11a.html)

## A more capable practice partner

From 87 earlier physical replies the five-axis partner learned which incoming flight conditions the SO-101 could return. On eight fresh starts it raised the near arm's legal returns from 19 to 44. Four rallies reached ten and two reached twenty, versus one and zero with the stock partner. On one start the same optical arm went from one alternating return to 27. The partner can use simulator state to rank its own physically legal shots. The learner sees its own camera frames. [Coach, matched services and full results](coach.md)

## Testing the source contribution across complete points

When the native Pong preference controlled only the first movement, it increased opening contacts from 11 to 23 on 24 points. Both the native and zero-gain arms reached ten twice; both first reached it on point 13. A duration-updated online selector gained less than one practice point on two disjoint twelve-point banks. A separate anticipation experiment gained a ten-return rally in a three-point pilot but failed to reproduce that advantage across twelve more paired starts. [Full points](tt23_sequence.md) · [Online selection and return to Pong](sequence-learning.md) · [Anticipation, video and replication](../tt22/value.md)

A separate test removed the supplied formula for moving to a predicted interception. A Pong paddle learned to choose among fixed displacements using point duration. On the first 64 arm starts the selected strategy beat holding position; a fixed source-label shuffle beat both, prompting independent source replications. [Discrete motor choice](anonymous-action.md)

The two approaches can be combined in a continuous point. On 24 scheduled starts the learned partner and neutral opening generated 22-, 46- and 30-return rallies. Native opening gains reached ten on the very first point. The neutral arm achieved the longer rallies overall. [Watch 46 legal returns](tt23_sequence_coach_46.mp4) · [Both opponent and opening interventions](tt23_sequence_coach.md)

A fresh carried-learning test used the improved partner and opening swing throughout. Pong ranking versus flat initialization led to 139 versus 91 legal arm returns across 48 matched points. The four paired streams first reached ten on points 2/5, 3/3, 1/3 and 1/6. Each online update used only the chosen point’s elapsed time. [All four learning curves](online-rally.md)

## Can a direction emerge from experience alone?

Giving the source learner seven fixed movement magnitudes reduced native Pong mistakes. Moving that whole choice to the arm often overstepped, even when feature smoothing improved native duration across five independent source batches. Online arm practice helped recalibrate the movement but a simple centered stroke was still quicker to its first landing. [Independent fits](anonymous-action.md) · [Smooth source scoring](smooth-jog.md) · [Online magnitude recovery](adaptive-jog.md)

A clock-only attempt to select later strokes from long successful table-tennis histories predicted the next second better than chance. Its ordering deteriorated on longer horizons when evaluated on a completely different group of recordings. [Continuation predictions](duration-value.md)


On eight independent time-to-ten confirmation streams using the supplied spatial rules, the Pong prior reached ten in 8/8 and flat in 6/8. Both succeeded on the same numbered point in all six common successes. On the two others flat was still unsuccessful after twelve receiving points. [Eight-stream confirmation](online-rally-confirm.md)

Five distinct native Pong fits each learned which direction to jog for the original arm releases. Compared with independently shuffled source durations the increase in mean arm openings was unlikely conditional on that bank (p=.029). The ordinary release generator couples where the ball starts with its flight direction. On a separate course varying the two draws independently, holding center got 36 openings and the native learner got 17. Adding delayed native observations and the explicit option to hold preserved native performance but the learner again moved on every detected arm start. [Pure direction](direction.md) · [Uncoupled flight paths](crossflight.md) · [Source hold-action experiment](abstain.md)


## Monocular and stereo rallies

On six matched legal services the wrist camera alone produced 27 near-arm returns and a 19-return rally. One fixed external camera produced 26 and reached 25. Stereo produced 39 and reached 25 on the same best start. When the court lines left the wrist image, the initial visual-registration algorithm lost its pose and earned zero returns. Using the nominal arm and its own commands to locate the moving camera restored single-eye ball tracking. All camera conditions shared the same physical serve and opponent. [Wrist camera with actual policy view](single-eye-wrist-rally.mp4) · [All six starts and RGB replays](single-eye.md)

## Transfer without an opening interception rule

On the 32-start three-action benchmark the Pong-initialized online chooser averaged 4.80 capped launches to ten versus 6.09 from flat initialization. The live eight-point deployment reached ten on the same point under both conditions. The single fixed reward shuffle tied the source on these visual contexts. Independent native height and angle draws improved native Pong performance on crossed motion but did not improve legal arm returns on crossed flights. [Anonymous online choices and retention](implicit-online.md) · [Independent source physics](decoupled-pong.md) · [Native visual value versus target visibility](sensory-value.md)
