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

## Additional runtime investigation, parked

A retained RIGHT miss was profiled with139old transition cameras and1initial (zero fresh candidate observations). Vision and tensor-rich copying dominate the remaining overhead. A guarded snapshot-copy alternative was implemented only on isolated branch`experimental/encoder-copy-20261025`, commit`1b4dddf`. Four focused alias/storage/cycle/mutation checks passed; two retained LEFT/RIGHT episodes matched every record and terminal graph/RNG (382old transition cameras+4initials across both implementations). Copy-only microbenchmark improved1.924×, but the two-episode aggregate under active-training contention improved only1.008×. This does not establish a useful end-to-end speed gain. The change is parked and not integrated. Evidence:`/workspace/m1b-experiments/pong-encoder-copy-review-20261025/verification.json`.

## Anchored, broader practice with action-dose scaling: still flat on fresh starts

Root:`/workspace/m1b-experiments/pong-practical-interception-20261025-r2`. Three generations completed in1255.164seconds (20.92minutes), including anchor comparison, legal feature capture, review and fresh evaluation. Source integrated at`28c7651`. Five earlier heads competed on78TRAINstarts; g4 from the first block was selected. Of390anchor outcomes,280were reused exactly and110were newly executed;32additional episodes supplied the legal512row feature buffer. Nine focused tests and the52transition/1initial retained capture-equivalence check passed.

The first combined update improved RIGHT from35/41to36/41 with LEFT unchanged25/37. The other two generations retained it. On64fresh starts:

| Opening | Original RIGHTtheta6 | Endpoint | Paired gains/losses |
|---|---:|---:|---:|
| LEFT |22/32|24/32|2/0|
| RIGHT |27/32|25/32|3/5|
| Combined |49/64|49/64|5/5|

This is not reliable improvement. No further blind mutation population is selected. The endpoint remains a useful common-bank candidate, not a promoted generalization success.

The completed block used371507TRAINtransition cameras,11144fresh-probe cameras,4248initials; all4248episode receipts complete, exact and independently summed by the parent. No ordinary-game rerun was spent on this flat interception result. The buffer showed substantial feature anisotropy (90%second-moment energy in25of214modes), but that is not a capacity ceiling.

A concrete next route uses the learner's own successful action trajectories. Generation1 already contained individual children that repaired5LEFTand5RIGHTstarts missed by its incumbent, although no single child retained all repairs. Sol is implementing a closed-form, regularized preference residual from such rewarded action differences, with successful-incumbent preservation and measured positive-direction action-change limits. Those are self-generated learning targets, not oracle actions. Eligibility is recomputed against the actual next carried policy; the early10repair count is not assumed unchanged. This remains within the existing linear head and uses no backpropagation. Fresh improvement still needs demonstration.
