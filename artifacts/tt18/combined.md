# Continuing to play during a change of arm, table and cameras

Two points into a scored game, a practiced gantry player encountered a smaller, uneven table, a heavier ball, a wrist load, different joint travel and swapped cameras. On the first schedule, carrying its motor fit cut the time to the next legal return from 250 to 114 control frames. On the second schedule the two arms returned the first ball at almost exactly the same time. Replacing the sliding gantry with an articulated arm exposed a larger gap. The experienced motor learner never returned a ball on that installation. [43-second game from the player's cameras](changing-table-and-views.mp4) · [Scores and frame counts](combined-summary.json.gz)

| Hardware after the change | Starting motor fit | Full game, arm–rival | Legal returns after switch | Frames to first return after switch |
| --- | --- | ---: | ---: | ---: |
| Compact gantry, schedule A | Carried and updating | 7–11 | 9 | 114 |
| Same | Cold and updating | 0–11 | 2 | 250 |
| Same | Carried and frozen | 4–11 | 8 | 137 |
| Same | Carried and updating, camera fits frozen | 1–11 | 0 | — |
| Compact gantry, schedule B | Carried and updating | 2–11 | 7 | 105 |
| Same | Cold and updating | 6–11 | 9 | 104 |
| Four-joint articulated arm | Carried and updating | 0–11 | 0 | — |
| Same | Cold and updating | 1–11 | 1 | 235 |
| Same | Choose among carried, updated and restarted fits | 1–11 | 0 | — |

All branches inherited the same game knowledge, stroke planner and color-based tracking. Cold describes the command-to-blade-motion fit. This comparison includes the effect of two scored opening points on each learner, so only the stale-camera comparison guarantees an identical opening action and image prefix to its carried control. The frozen-body control also omits the active learner's small exploratory command adjustments. No target-side dry movement was allotted. Legal returns count the arm's replies; legal own serves are counted separately. A game ends at 11 points with a two-point margin. Each recorded game includes the requested actions, the actions actually applied and the referee events. All twelve were replayed in the simulator. [Game runner and replay](../../scripts/experimental/tt18_combined.py) · [Independent counts](../../scripts/experimental/tt18_combined_report.py)

## Which adaptations helped in combination?

Fitting the camera from visible landmarks made an immediate difference on schedule A. The stale camera control saw the same first two points and then failed on every remaining return. Both calibrated-camera motor controls made returns. The adaptive motor fit won more points than the frozen fit, but the frozen fit served legally on four of seven own attempts after the switch and sustained two points. The updating fit served legally on three of nine and sustained one. That trade-off and the second schedule limit the case for a general motor advantage. [Carried game](combined-418336241-carried.json.gz) · [Frozen body](combined-418336241-fixed_body.json.gz) · [Stale cameras](combined-418336241-stale_view.json.gz)

Adding spatial bounce learning to the adaptive body learner produced the same final scores of 7–11 and 2–11. After the switch, legal returns fell from nine to five on schedule A and seven to two on B. Legal serving improved to six of eight on A and fell to zero of six on B. Independently useful camera and surface estimates did not automatically improve joint control. [First world-model game](combined-418336241-world.json.gz) · [Replication](combined-349051097-world.json.gz) · [Independent surface study](surface.md)

An alternative picked the motor fit that best predicted recent optical blade displacements. On the gantry it stayed with the frozen fit and duplicated the 4–11 game. On the articulated arm it used the restarted fit on seven point attempts and landed a legal own serve. It still made zero legal returns. Discovering that old motor predictions are inaccurate is easier than finding movements that meet a ball in time. These are single articulated-arm and two gantry schedules. [Articulated-arm comparison](combined-stock-96955769-choice.json.gz) · [Motor choices](../../scripts/experimental/tt18_body_choice.py)

## Checkpoint and retention

The first gantry's saved 7–11 player retained all 180 reference Pong action tapes, including 41 earned points. Use the command below to start a fresh scored schedule carrying its learned state. The runner starts the new game's score at zero and preserves the body estimate, stroke experience and registered views. The changing-table fixture and camera sequence are supplied by the evaluator. [Player checkpoint](combined-418336241-carried-player.json.gz) · [Pong retention receipt](pong-retention.json.gz)

```sh
python -m scripts.experimental.tt18_combined /tmp/tt18-followup carried --seed 184902217 --resume artifacts/tt18/combined-418336241-carried-player.json.gz
```
