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

## Directed reward-earned repair: fast LEFT gain, RIGHT interference

Root:`/workspace/m1b-experiments/pong-rewarded-residual-20261025-r1`; source integrated at`d108992`(with`5c5e78b`). One proposal completed in94.995seconds,9226TRAINtransition cameras+16932fresh-probe cameras+296initials. All296receipts independently summed and complete. Ten actual successful child attempts (5per opening) supplied113action-contrast rows;16carry successes supplied256zero-residual preservation rows. The same linear head was fitted by regularized closed-form regression, with no backpropagation or external teacher.

The raw fit made30.0%of requested actions preferred under phase/episode weighting. Forward rescaling by1.283raised this to54.2%, with functionRMS.06797,10.16%buffer action disagreement and8.59%preservation-row action changes. Thus this was an actual behavioral update, not merely nonzero weights. Whole-bank LEFT25→26/37,RIGHT36→35/41 caused harmonic selection.

| Fresh32starts per opening | Original RIGHTtheta6 | Carried practical-r2 | Directed update |
|---|---:|---:|---:|
| LEFT |22|23|27|
| RIGHT |26|26|21|

Relative to carry, LEFT gained4hits with0losses; RIGHT gained2but lost7. The LEFT improvement is worth retaining as a learned skill, but aggregate49→48/64does not meet the integrated interception goal. Do not promote this as a reliable combined learner. The next work targets interference/preservation, not another random population. Broader preservation of actual spoiled contexts and the optimizer's low-frequency restriction are the competing concrete explanations.

## Full-trajectory preservation: rejects the fitted direction

Root:`/workspace/m1b-experiments/pong-preserved-residual-20261025-r1`; source integrated at`e425b6e`with`9ae5482`. The previous64monitor starts were explicitly reclassified asTRAIN, producing69LEFT/73RIGHTcontexts. The carried r2head's cached baseline is48/69LEFT,62/73RIGHT. Ten original repair and16preservation episodes were reused; six newly successful donor attempts and seven spoiled RIGHT successes were captured. The fit used163positive and1152preservation rows, including896full-trajectory guard rows.

Half-margin preservation limited the fitted direction's rescaling from1.2646to.04068: actualfunctionRMS.001794, zero buffer action changes and zero requested repaired actions attained. Actual games retained7/7protected returns, but the whole142-start bank was unchanged. The proposal was rejected; no fresh evaluation was spent. Cost13963TRAINtransitions+155initials,46.564seconds, all receipts independently summed and complete. This rejects that fitted direction, not the entire policy class.

Next: solve the same regularized preference-fitting objective subject to the action-margin inequalities, instead of fitting first and shrinking afterward. The428variable contrast-coordinate quadratic program uses OSQP1.1.3, no backpropagation and no new captures. It must report feasibility/action effects and then earn its place through actual games. No further tiny-step tuning is selected.

## Constraint-aware repair: partial improvement carried forward

Root:`/workspace/m1b-experiments/pong-constrained-residual-20261025-r1`; integrated source`4b9298e`, experimental dependencyOSQP1.1.3. The same saved rows yielded a feasible428variable/1792inequality solve in1.085seconds, no new capture. Objective.08637at zero→.03273; maximum violation9e−17. Parent independently checked896guard choices and half margins. The actual proposed direction hasfunctionRMS.04003,2.34%buffer action changes,31.13%weighted requested-action attainment and.625%weighted preservation action changes. It is a different useful-sized direction, not another collapsed step.

Actual142-start TRAIN: LEFT48→51/69,RIGHT62→64/73;7/7protected returns retained. Fresh32/phase comparison: originalRIGHTtheta6LEFT20/RIGHT21; carriedr2LEFT21/RIGHT22; constrainedLEFT24/RIGHT20. Against carry this isLEFT4gains/1loss,RIGHT1gain/3loss, aggregate+1/64. No90%or reliable-generalization claim.

Cost12577TRAINtransitions+17240fresh-probe transitions+334initials, all334receipts exact and independently summed. Total231.616seconds includes solve, parent review and actual games. Reclassifying this64monitor set asTRAIN for the next pass yields206common contexts: previouscarry69/101LEFT,84/105RIGHT; constrainedhead75/101LEFT,84/105RIGHT. The constrained head dominates there and is retained prospectively as useful partial progress.

Next selected pass meshes reward-trained donors: query the earlierDIRECTlearner on the newcarry's26LEFTmisses; use the olderRIGHT-capable policy for contexts it returned that currentcarry misses; refilter existing self-repair examples against currentcarry. Protect currentcarry's4newly gainedLEFTfull trajectories and its7retainedRIGHTtrajectories (verify actions and recompute current logits before reusing them). One constrained fit and206-context execution, with new fresh comparison if useful. Old successful behavior that currentcarry misses is a repair donor, not currentcarry preservation. No random population or architecture alteration is selected.
