# Learning to rally with the SO-101

[Watch fourteen legal returns](rally-14.mp4) · [Watch the practiced arm complete ten](learned-ten.mp4)

The near robot is the end-clamped SO-101 simulation with the gripped paddle and compact table. A return counts when the ball actually lands on the other half after legal physical contact. We exclude the serve. Ten returns therefore require five from each player in one uninterrupted point. The receiver uses two fixed outside RGB cameras in the main comparisons. [Counting and held-start protocol](benchmark.md)

| What played | Held starts or game | Result |
|---|---:|---|
| Two planners with simulator state, ordinary serve | 4 | 24, 10, 32, 22 returns. All four points replayed exactly. |
| RGB arm, improved physical four-axis partner | 12 matched starts | Baseline lengths 4, 8, 2, 4, 4, 4, 6, 8, 8, 7, 10, 4. After four optical practice points the frozen correction produced 4, 8, 2, 4, 8, 10, 4, 4, 8, 3, 8, 1. One ten-shot rally each. |
| RGB arm, five-axis partner whose feed target was fit from prior rallies | 6 wider, matched starts | Original target 4, 2, 10, 6, 10, 7. Learned target 4, 2, 10, 10, 10, 4. The partner learned the feed placement. |
| RGB arm, same learned five-axis partner, independent wider starts | 12 matched starts | Original near stroke ranker reached ten twice, with a longest of 14. The near ranker fit to actual practice successes reached ten once, with a longest of 11. Both also produced short points. |
| Full alternated-service game, online near calibration | 13 points | 2–11, 15 SO-101 legal returns, 19 rival legal returns, longest rally 8. The arm served legally 6/6. |

[Physical upper bound](cycle.md) · [Four-axis starting policy](rally-anchor-220381117.json.gz) · [Four-point checkpoint evaluation](benchmark-learned.json.gz) · [Five-axis opponent](tt22_opponent.md) · [Twelve-start wider comparison](paired-course.json.gz) · [Full game](match.md)

## What the arm learned

The near arm compared forecast landings with its own subsequent optical observations. Four practice points supplied seven usable observations. In a separate continuous six-start comparison the practiced arm produced 24 combined legal returns versus 20 for a control with the same edge and net scoring terms and zero learned correction. The online arm improved four starts, shortened two, and reached ten once, as did that control. On one matched start it went from four to ten. On another the correction went from ten to two. The videos and receipts distinguish these score changes from capacity that the unpracticed physics planner already had. [Training, checkpoint and controls](rally.md)

The five-axis rival moves a real simulated racket through a motorized wrist. On eight matched openings it returned 8/8 legally and brought the median lateral feed offset down from 22.8 cm under the previous cooperative rival to 3.7 cm. All 64 points in its comparison suites replayed from both robots' actual controls. Its fitted placement improved one six-point cohort. A further fit for incoming vertical velocity improved the requested physical trajectory while leaving total legal returns equal. [Opponent results](tt22_opponent.md)

## Pong transfer and retention

A learned anonymous image-motion predictor gave 11/16 first arm returns versus 12/16 from scratch. Pong-trained lateral preferences gave 5/8 versus 4/8 first returns with a bounded correction to the geometric controller. On an uncapped four-start comparison the counts were 1/4 for Pong, 0/4 for scratch, and 2/4 with the native score order permuted. One permuted-score point lasted fourteen returns. The stronger result for that control precludes attributing long rallies to the native score ordering. [Motion transfer](transfer.md) · [Lateral policy, comparisons and checkpoints](spatial.md)

Pooling arm rewards into the shared preference reduced first returns on a native Pong retest from 13/20 to 8/20. Weighting experience by the observed motion of the controlled body reproduced all 16 target choices and all 24 baseline Pong action and outcome tapes from a single post-arm store. [Retention audit](spatial-context-audit.json.gz)

The main long-rally receiver uses calibrated stereo, color proposals, a body model and a stroke search that scores contact and landings. It demonstrates physical rally capacity and some online correction. Learning the whole collision strategy implicitly from Pong requires a distinct comparison. Removing the candidate contact simulator in the first timed-stroke experiment produced contacts on 6/6 held starts for both the Pong and cold policy and only one legal return each. [Motor-transfer experiment](motor-transfer.md) · [Generic relation controls](relation.md)

## Seeing the rebound

At 30 Hz the existing four-frame optical velocity fit sometimes blended downward and upward travel across the table bounce. In two archived points a bounce-aware fit calibrated on seven optical bounces from four separate practice points cut the 90th-percentile vertical-speed error in the first two post-bounce frames from 4.42 to 0.10 m/s. A smooth aerodynamic fit reduced median pre-bounce 3D velocity error from 0.086 to 0.025 m/s. Only the evaluation compared estimates with simulator state. On four fresh ordinary serves in the separate optical-cycle harness, using the smooth pre-bounce velocity produced 13 combined returns versus 9, with no ten-return points. [Image precision](imaging.md) · [Bounce estimates and scored pilot](bounce-filter-pilot.json.gz) · [Ordinary-serve comparison](cycle-arc-live.json.gz)

Dense grayscale patches improved the chance that both cameras had a candidate on the paddle from 0/90 to 60/90 visible precontact frames. A transferred role selector still picked the wrong right-eye region. First-serve tracking during a quiet observation interval and sustained binding in a moving scene are distinct benchmarks. [Active camera probes](anonymous-babbling.md) · [Interaction ranking](anonymous-interaction.md)

## Playing from unlabelled motion

The arm learned the moving region's appearance from eight quiet opening frames. On four fresh points including two randomized ball colors, anonymous tracking supported eight legal arm returns versus one with the preassigned color. The longest exchange was eight alternating returns. All eight complete optical and motor tapes replayed. The high-performing controller still supplies a physical stroke objective. [Moving object discovery](motion-object.md)

In a separate direct control experiment, Pong duration training produced 47 physical contacts on 48 starts versus 19 for a neutral visual policy. On an independent 64 starts the numbers were 62 and 25. The first cohort's legal returns improved from 9 to 20 and the second tied at 18. Previous target-only practice supplied the arm stroke. A comparison with every unselected option and the six-stream training curves explains where the head start faded. [Source transfer and online practice](source-transfer.md) · [Lower-performing motor alternatives](other-pilots.md)
