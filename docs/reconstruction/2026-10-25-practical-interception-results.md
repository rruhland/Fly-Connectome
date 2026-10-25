# Practical interception continuation, October25

Work is ongoing under the user's instruction to achieve reliable efficient interception. This file records completed blocks, not a session stopping point. No backpropagation, privileged policy input or new architecture was introduced.

## Faster execution

The lean experimental episode runner preserves the existing evaluator, camera/vision, policy and simulator. Four retained hit/miss trajectories matched every checked record and terminal full state against archived and durable executions. Alternated single-process execution took4.129s versus6.679s (1.617×speedup). This check reexecuted720old transition images and8initial images, with no new candidate observations. Start/end frozen-state checks and compact episode receipts replace expensive per-frame journals; arbitrary-camera crash recovery is not supported.

Code is integrated on the continuation branch at`e9d72ef`. Proof: `/workspace/m1b-experiments/pong-fast-research-20261025/verification.json` and `parent-review.json`.

## Repeated-bank practice: no reliable gain

Root: `/workspace/m1b-experiments/pong-practical-interception-20261025-r1`.

Started from the carried RIGHTtheta6 learner. Six generations, four selected updates, same frozen vision and linear policy class. Eight mirrored directions and a combined proposal competed on a repeated ordinary-start bank, selected by weakest phase then average. The bank began with16starts per phase;14failures from the generation3 probe were subsequently added as explicit TRAIN curriculum.

| Comparison | Starting policy | Current policy |
|---|---:|---:|
| Generation3 fresh LEFT |12/16|11/16|
| Generation3 fresh RIGHT |12/16|7/16|
| Generation6 fresh LEFT |11/16|11/16|
| Generation6 fresh RIGHT |11/16|11/16|
| Ordinary32games: first return |27/32|26/32|
| Ordinary32games: total own hits |33|34|
| Ordinary32games: at least2hits |6/32|7/32|

The original bank improved from14/16both to14/16LEFT,16/16RIGHT while fresh performance deteriorated. Adding mistakes recovered some lost behavior, but did not yield a net fresh gain. On the expanded bank, the original policy scores15/21LEFT,20/25RIGHT using existing receipts; the endpoint scores15/21LEFT,18/25RIGHT. Thus the original policy dominates the endpoint on that same training set. Generation5's weakest-phase priority accepted one extra LEFT hit at the cost of three RIGHT hits. Recovery from a bad earlier update must not be reported as new skill.

Elapsed1222.358s (20.37minutes),358282training transitions,11379phase-probe transitions,18103ordinary-game transitions,4202initial images. Accounting is exact with no ambiguous/lost grants. About317transitions/s across the completed block including evaluations; this is not a matched parallel-speed benchmark. Compact receipts avoid the former multi-GB journal footprint.

## Next selected direction

Preserve strong earlier policies as selectable anchors on a common training bank; use broader representative starts as well as remembered failures; replace lexicographic weakest-phase priority with a smooth joint utility; and measure mutation size through forward action changes on actual legal policy inputs. Sol is implementing the next practical pass. Fresh outcomes, not the score of repeatedly practiced examples, will determine whether it helped. The target remains reliable interception in both opening cohorts, not winning or a statistical-gate count.
