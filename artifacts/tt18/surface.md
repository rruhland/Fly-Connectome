# Learning an uneven tabletop from scored play

A location-specific bounce estimate reduced mean absolute error on 59 matched incoming rebounds to **0.088**, versus **0.145** when treating the whole table alike and **0.110** with a fixed coefficient of 0.86. Error is measured in the ratio of outgoing to incoming vertical speed. In play the local estimate produced one win in four full games. The inherited player lost all four. [Paired physical and game results](surface-summary.json.gz)

The halves of each table looked alike. They differed in physical contact damping. Between the two games on each stream, the responsive and dead regions exchanged sides. The player carried its observations and received no notice of the change. It estimated a bounce from stereo positions before and after impact and used nearby previous bounces to forecast the next one. The forecast affected the existing flight tracker and stroke planner. No backpropagation, dry actions or referee contact signals entered that estimate. Both arms began with familiar geometry and camera calibration. [Learner](../../scripts/experimental/tt18_surface.py) · [Physical audit](plant.md)

| Held fixture | Inherited player | One shared bounce estimate | Location-specific estimate |
| --- | --- | --- | --- |
| Gantry, first table | 4–11, 10 returns | 6–11, 4 | 3–11, 7 |
| Same gantry, reversed regions | 11–13, 4 | 8–11, 3 | **11–8, 5** |
| Five-axis, first table | 2–11, 2 | 4–11, 1 | 2–11, 3 |
| Same five-axis, reversed regions | 6–11, 7 | 3–11, 8 | **13–15, 10** |

The local player made 25 legal returns and sustained five points across these games. The inherited player made 23 returns and sustained six. Final scores improved on the two reversed tables, while the average number of returns on a point did not consistently improve. The duplicated local and source-selected arms produced identical action tapes on all four held games.

## What transferred from parallel source play

Two independent source learners each played two consecutive games on different split tables. Their 5,286 control frames produced 53 useful stereo bounce observations. Each candidate estimator had to predict each observation from earlier observations in its own source stream. The lowest average error selected a spatial averaging width of 0.20 m and a decay time of eight observations. Only these two numbers transferred. Target learning started with an empty surface map. The source selection independently reproduced the prespecified local baseline, so this experiment shows a benefit from spatial memory but no advantage over that baseline from extra source play. [Source fits and hashes](surface-source-rule.json.gz) · [Gantry source games](surface-gantry-source.json.gz) · [Five-axis source games](surface-five-source.json.gz)

The physical comparison used the inherited player's exact trajectories for every estimator. The evaluator supplied the true location of each incoming contact only to query the learned map, and compared that prediction with the MuJoCo speed ratio at separation. Updates used only optical bounces observed strictly earlier. The playing policy had to estimate the next bounce position from RGB. This distinction separates learning a useful surface map from measuring whether the stroke controller can use it well. The full game receipts were replayed through the simulator, including point scores, actions applied and physics events. [Evaluator](../../scripts/experimental/tt18_surface_report.py)

Use the following command in the repository's table simulation environment. The worker count controls concurrency. Each worker owns one learner and each target sees a single chronological stream. Re-running into a fresh output directory preserves the published evidence.

```sh
python -m scripts.experimental.tt18_parallel /tmp/tt18-surface --workers 3
```
