# Wrist RGB in scored SO-101 play

## Full strict match replay

The SO-101 won 11–5 on the full-spread strict mini court in game `424520913`, with seven legal own serves and six legal returns. The 36.13-second replay runs from the opening point through the final score. It shows the broadcast camera alongside the wrist and both fixed RGB feeds the policy received. The scoreboard marks the server, the point result, the match clock and the timeline. Short score holds between points leave the match clock stopped. [Watch the full 11–5 game](so101-full-game.mp4) · [Original scored game](match-model-rollout-native.json.gz)

The replay covers all 16 points and 898 commanded frames. The 793 original optical capture ticks follow the archived schedule, with every-other-frame sampling on the opening phase and every-frame sampling during receive. An inset holds its last captured image between observations. The replay checks each point’s RGB digest, recovered ball coordinates, contributing views, referee events and outcome. The machine-readable receipt includes the input and video hashes and the video position of each point. [Full-game replay receipt](so101-full-game-replay.json.gz) · [Replay code](../../scripts/experimental/tt19_view_full_game.py)

## Matched camera comparison

The SO-101 wrist camera shows the gripper, handle, blade and part of the court in the same native image. In the matched game, the wrist-plus-stereo player scored 2–11 with three legal own serves. The wrist-only player scored 0–11. Both games used the same warm donor, 24 requested setup actions, and matched point starts and disturbance phases. The hybrid branch observed the blade on all 25 setup captures. The wrist-only branch observed it on eight. The referee and physics reproduced all 24 scored attempts. [Wrist image](view-wrist.png) · [Wrist and fixed game](online-29413852-warm-wrist+fixed-setup24.json.gz) · [Wrist-only game](view-wrist-only/online-29413852-warm-wrist-setup24.json.gz) · [Wrist-only checkpoint](view-wrist-only/online-29413852-warm-wrist-setup24-game1-player.json.gz)

| Live control views | Score | Legal own serves | Legal returns | Captures | Metric ball | Metric blade | Registered wrist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wrist and fixed stereo | 2–11 | 3/6 | 0 | 537 | 475 | 537 | 420 |
| Wrist only | 0–11 | 0/5 | 0 | 388 | 57 | 273 | 341 |

The wrist camera is rigidly attached to the modeled SO-101 camera assembly. Its local lens orientation places the jaws and paddle in the same field as the court. The simulated public lens specification supplies its nominal 93° vertical field of view. Surveyed colored posts identify world points in RGB. The tracker fits the moving camera pose from these points and rejects frames with insufficient geometric evidence. The ball, paddle, jaw tip and four paddle rim marks have visible colors. The rim marks provide known blade geometry for pose estimation. The wrist-only ball estimate uses the known ball size and rejects unresolved or highly uncertain detections. During gameplay the controller receives RGB, its prior actions and point rewards. MuJoCo ground truth is used by the external audit for error scoring. [RGB tracker](../../scripts/experimental/tt19_view.py) · [Native camera fixture](../../scripts/experimental/tt19_view_trial.py)

The 5.53-second game clip shows three selected points, including a legal robot serve and the ensuing rally, at the simulation’s 30 control frames per second. Its wrist and fixed insets replay the exact RGB bytes supplied to the policy on those points. The full-CAD spectator view uses the same checked action tapes. The second clip shows a legal return from a separate 6–11 game played with fixed cameras. Its wrist inset is an evaluator view. [Wrist and fixed game clip](view-game.mp4) · [Legal return clip](view-return/view-game.mp4) · [Return game receipt](stroke-gantry.json.gz) · [Replay code](../../scripts/experimental/tt19_view_replay.py)

## Observation audit on the same recorded motion

The three hybrid-game clips supplied 166 captures to each observation mode. The wrist saw both the tip mark and blade throughout these clips. Their centroid errors against the projected centers had medians of 7.75 px and 9.26 px, respectively. The moving wrist registered in 110 captures, with median optical-center error of 8.08 mm. In the hybrid tracker it contributed to 58 ball reconstructions. Blade-normal coverage was 140 frames in the hybrid tracker and 121 frames with the fixed pair alone.

| Views replayed | Ball frames | Ball median / p90 error | Blade frames | Blade median / p90 error | Capture and tracking median / p90 wall time |
| --- | --- | --- | --- | --- | --- |
| Wrist and fixed stereo | 142 | 5.1 / 13.8 mm | 166 | 3.6 / 5.7 mm | 219 / 323 ms |
| Wrist and one side camera | 62 | 5.7 / 15.9 mm | 74 | 11.7 / 26.3 mm | 161 / 235 ms |
| Wrist only | 11 | 21.1 / 112.1 mm | 84 | 7.4 / 25.5 mm | 94 / 152 ms |
| Fixed stereo | 142 | 4.8 / 12.1 mm | 166 | 3.8 / 5.6 mm | 129 / 186 ms |

This audit holds actions constant and scores optical estimates on frames where each estimator returned a measurement. The uneven coverage matters, especially for ball range from the wrist alone. Timing includes the selected native RGB renders and tracker work on a shared CPU with software rendering. It excludes the spectator view and video encoding. The measured pipeline exceeded the 33 ms simulation frame interval. [Frame-level observation audit](view-audit.json.gz) · [Legal return audit](view-return/view-audit.json.gz)
