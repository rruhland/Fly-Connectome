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

## Learned donor meshing and finer spatial search

Meshing the earlier reward-trained LEFT and RIGHT policies improved the 206-context bank from 75 to 83/101 LEFT and 84 to 86/105 RIGHT. Fresh paired games changed from 23/32 LEFT, 22/32 RIGHT to 23/32 and 23/32. All 12 protected return trajectories survived. Cost: 123.10 seconds, 20,628 TRAIN and 17,060 fresh transitions, 435 initial cameras. This is a useful partial checkpoint, not reliable acquisition. Artifacts: `/workspace/m1b-experiments/pong-meshed-donor-20261025-r1`.

Expanding search from 214 to 566 coordinates using higher DCT modes of the existing camera features improved the next 270-context bank from 106 to 110/133 LEFT and 109 to 111/137 RIGHT. Fresh games were 25→25/32 LEFT and 26→25/32 RIGHT. All 13 guards survived. Cost: 154.09 seconds, 29,260 TRAIN and 16,870 fresh transitions, 520 initial cameras; parent independently summed all receipts. The deployed 3×4389 linear head and inputs are unchanged, with split mapping preserving the carried model exactly at initialization. Actual residual RMS .0846; requested action attainment 27.1%; no fresh integrated gain. Artifacts: `/workspace/m1b-experiments/pong-spatial-residual-20261025-r1`.

The latter checkpoint is retained for its practiced gains: 135/165 LEFT and 136/169 RIGHT on all now-observed contexts. Fresh probes become TRAIN only when explicitly reused. Next work prioritizes new reward-successful attempts: 23 of 30 current LEFT misses and 15 of 33 RIGHT misses have never been returned by any tested practical head. Successful donor traces also lost action coverage through 16-row truncation; new successful discovery traces will retain all contrasting actions and lightly preserve matching actions.

## Targeted local reward discovery: partial fresh gain

All 256 targeted weight mutations flipped their intended action on the recorded legal input; none hit the update cap. Actual games produced 17 successful trials covering 13 previously missed contexts (6 LEFT, 7 RIGHT). Successful trajectories differed from the carried policy on only 50 of 1,224 decisions. All contrasting rows were retained, and the 1,174 matching rows contributed a separate .25 soft preservation term.

The integrated policy retained only 3/13 discovered repairs (2 LEFT, 1 RIGHT), but improved the full 334-context bank from 135/165 LEFT, 136/169 RIGHT to 135/165, 141/169. Fresh comparison improved 24→25/32 LEFT and 25→27/32 RIGHT. All 16 protected trajectories returned. Runtime 255.85 seconds; 59,510 TRAIN transitions, 11,317 fresh transitions, 766 initial cameras; parent independently summed exact receipts. Root: `/workspace/m1b-experiments/pong-local-repair-20261025-r1`; selected checkpoint retained as a useful partial gain, not yet reliable interception.

Next: one fourfold repair-loss weighting fit on the same successful trajectories and original base, preserving hard guards and update cap. Compare with the useful onefold candidate rather than discarding it. This tests under-absorption using existing data. Parent also tried an offline squared-hinge margin QP on the previous DCT5 dataset; it exceeded the 30-second solver budget and produced no accepted candidate or camera observations. Parked.

## Stronger repair fitting and solver acceleration

Fourfold positive repair loss on the same 13 discovery trajectories improved their integration from 3/13 to 5/13. Relative to the useful onefold head, the common bank improved LEFT 135→139/165 (4 gains, no losses), RIGHT 141→143/169 (7 gains, 5 losses), keeping all 16 guarded returns. Fresh counts stayed 28/32 LEFT and 26/32 RIGHT; no fresh aggregate gain. Keep the practiced gain, and do not continue a weight sweep. Artifact: `/workspace/m1b-experiments/pong-local-repair-weighted-20261025-r1`.

The dense solver timed out after 30 seconds before any camera calls. An exact working-set solve, seeded by the prior fit's 71 binding constraints, completed in 1.23 seconds with 139 working constraints after three rounds. All 3,488 original inequalities were satisfied to numerical precision; primal/dual objective gap was about 1e-16. No approximate fallback was used. Parent independently verified all 1,744 protected action rows and split raw-weight mapping. This is successful fast recovery versus a timed-out dense solve, not a timing ratio between two completed solutions.

Total elapsed time including solver failure and recovery implementation: 377.73 seconds. Actual observation cost: 29,429 TRAIN transitions, 11,223 fresh transitions, 462 initial cameras, exact. Original clock was preserved and no extension was needed. New helpers retain feasible approximate iterates when a future numerical limit prevents strict convergence; any such use must be labeled and still pass full guard and objective checks.

## Repeated discovery and experience retention

The reusable step appended the last 64 fresh contexts as TRAIN (398 total), explored 16 failures per phase, and retained full earlier successful experience. New discovery found 8 contexts (2 LEFT, 6 RIGHT), with one overlap among eight older missed donor contexts. Exact conflict filtering omitted three unattainable preference rows, including one opening-only target episode; full donor memory was retained. The working-set fit solved in 1.03 seconds and matched 83.5% of the remaining desired actions.

Actual older-repair retention was 6/8 (3/3 LEFT, 3/5 RIGHT); new repair retention 4/8 (0/2 LEFT, 4/6 RIGHT). Practice improved LEFT 167→168/197 and RIGHT 169→179/201, keeping all 17 guarded returns. Fresh totals stayed LEFT 26/32 and RIGHT 30/32. This shows useful accumulation on practice, not reliable fresh improvement. Current candidate retained. Root: `/workspace/m1b-experiments/pong-local-repair-20261025-r2`.

Runtime 245.24 seconds, 64,526 TRAIN transitions, 11,111 fresh transitions, 823 initial cameras, exact; parent independently summed receipts. The exported learning-state.json carries the selected policy, cached outcomes, complete rewarded trajectories and validated success traces for the same runner's next step. Next direction under review: more discovery effort on LEFT and exploration beyond only low-margin decisions, with the fitting rule held fixed.

## LEFT-focused remembered skill and temporal exploration: fresh gain

The earlier reward-trained DIRECT policy returned 17/27 current LEFT misses (five cached successes reproduced, 12/22 newly queried successes). Local exploration then targeted only the remaining 18 LEFT and eight RIGHT misses, adding temporal-bin midpoints on LEFT: 342 distinct candidate episodes, all intended flips effective, supplied another seven LEFT and two RIGHT successful contexts. This combination was selected jointly: Astra proposed LEFT-focused broader timing; Sol proposed querying the cheaper remembered DIRECT skill first.

The single integrated policy improved practice LEFT 194→211/229 (18 gains, one loss) and RIGHT 209→210/233 (seven gains, six losses). Both practiced phases now exceed 90%. All 19 protected returns survived. Fresh LEFT improved 26→28/32 and RIGHT 25→29/32: six gains and no losses. DIRECT-donor absorption was 10/17; local absorption 5/7 LEFT and 0/2 RIGHT, with source overlaps recorded. This is meaningful fresh learning progress, but the small LEFT cohort is still below 90%. Current carry: `/workspace/m1b-experiments/pong-local-repair-20261025-r3/selected.npy`.

Cost: 257.62 seconds, 68,780 TRAIN transitions, 11,122 fresh transitions and 996 initial cameras, exact; parent independently summed receipts. Working-set fitting took 1.74 seconds. No architecture, inputs, backpropagation or teacher/oracle actions were introduced.

Next selected action is a fixed larger reliability check, not another learner change: two fresh cohorts of 64 starts per phase each, selected r3 versus the original RIGHTtheta6 carry. Report each cohort and the 128-per-phase aggregate; if both aggregate phases reach 90%, run 32 ordinary full games per head. Keep winning separate from interception, and do not tune or retry the validation cohorts.

## Larger frozen reliability check: genuine gain, LEFT still below target

The predeclared 128 starts per phase confirmed clear accumulated learning against original RIGHTtheta6: LEFT 92→113/128 (71.88%→88.28%), 29 gains and eight losses; RIGHT 104→117/128 (81.25%→91.41%), 18 gains and five losses. Current cohort counts were LEFT 57/64 and 56/64, RIGHT 63/64 and 54/64; the RIGHT variation matters, and the aggregate is not a uniform guarantee. No learner updates or cohort retries occurred. The ordinary-game trigger correctly did not fire because LEFT remained below 90%.

Cost: 145.79 seconds, 44,553 evaluation transitions and 512 initial cameras; zero TRAIN or full-game cameras. Parent independently recomputed paired counts and summed receipts. Root: `/workspace/m1b-experiments/pong-interception-reliability-20261025-r1`.

Continue learning rather than declare completion. The next selected pass explicitly reclassifies these 256 contexts as TRAIN, retains the original evaluation artifacts unchanged, and also adds the latest routine 64 probes (782 total contexts). The original comparison already demonstrates 13 useful forgotten behaviors: eight LEFT and five RIGHT current misses were returned by the original learned head. Capture those successes, query remembered DIRECT for new LEFT misses, and use broader local exploration where useful repair witnesses remain missing. Keep the fitting rule fixed; add eight representative RIGHT success guards, four per qualification cohort, plus ordinary newly acquired guards. Future reliability checks must use genuinely new contexts.

## Qualification failures become practice: next checkpoint reaches small-cohort target

The next pass explicitly imported the completed qualification as TRAIN, alongside routine probes, producing 782 practiced contexts. All 13 original-policy return witnesses reproduced; DIRECT supplied ten new successes from 20 queries and seven complete cached successful traces were reused without cameras. Local search on uncovered contexts found four LEFT and three RIGHT return witnesses. The objective and architecture stayed fixed; 33 success guards included eight representative newly qualified RIGHT successes.

Practice changed LEFT 352→360/389 and RIGHT 356→354/393. Fresh changed 28→29/32 in both phases. The candidate was retained; both practice and the small fresh cohorts now exceed 90%, but a larger genuinely new check is still needed. ORIGINAL witness absorption was 2/8 LEFT and 4/5 RIGHT; DIRECT absorption 7/17. The positive-target fit remained partial, and a possible future behavior-only guard margin was discussed but NOT implemented because actual performance justified validation first.

Runtime 370.59 seconds; 96,404 TRAIN transitions, 11,045 fresh transitions and 1,291 initial cameras, exact and independently summed by parent. Root: `/workspace/m1b-experiments/pong-local-repair-20261025-r4`. Next: reuse the already reviewed frozen reliability driver for two new 64-per-phase cohorts against original RIGHTtheta6, with the same conditional ordinary-game check. No checkpoint changes or cohort retries during evaluation.

## Second larger qualification and rejected relaxed-margin fit

Frozen r4 on 128 genuinely new starts per phase returned LEFT 113/128 (88.28%) and RIGHT 111/128 (86.72%), versus original 93 and 99. Cohorts were LEFT 57/64, 56/64; RIGHT 58/64, 53/64. This confirms accumulated learning but not the 90%-both target. No ordinary-game trigger. Cost: 142.38 seconds, 45,029 evaluation transitions, 512 initial cameras. Root: `/workspace/m1b-experiments/pong-interception-reliability-20261025-r2`.

A single same-data refit then relaxed confidence retention to `min(.5 * old_gap, 1e-4 * q_RMS)` while preserving actual guard actions in both QP and radial bounds. Solving took 1.42 seconds; requested-action attainment rose from 36.87% to 47.84%, and all 33 native guarded returns survived. Actual practice did not improve: LEFT 360→360/389, RIGHT 354→349/393. Rejected; no fresh cameras spent and local-r4 remains carry. Runtime 216.97 seconds, 68,119 TRAIN transitions, 782 initials. Root: `/workspace/m1b-experiments/pong-behavior-margin-20261025-r1`. No further margin/weight sweep selected.

Parent replayed four retained qualification misses for diagnosis (privileged state used only for offline inspection, never learning). Two LEFT misses were close with intermittent ball tracking; two RIGHT misses showed wrong-direction paddle motion, including one with accurate late ball tracking on all 15 inspected frames. This rules out missing vision as a sufficient explanation for every miss; four cases do not establish a global mechanism. Replay cost: 4.52 seconds, 382 OLD transitions and four OLD initials. Artifact: `/workspace/m1b-experiments/pong-practical-review-20261025/current-retained-misses-diagnostic.json`.

## Runtime improvement during the controller-search pivot

Parent profiling identified nested-vmap overhead in the 32-step mixture quantile calculation. Explicit broadcasting preserves the arithmetic/reduction order without a new dependency. Four retained LEFT/RIGHT hit/miss episodes, each checked twice with alternating implementation order, matched every action record, summary and terminal-state fingerprint exactly. Whole-episode measured speedup was 1.077×; the kernel-only improvement was larger and is not claimed as training speedup. Existing vision performance parity suite: ten tests passed, including online learning/serialization equality. Cost: 153 OLD transitions plus one initial for profiling; 1,600 OLD transitions and 16 initials for trajectory/timing checks; zero new candidate observations. Artifacts under `/workspace/m1b-experiments/pong-practical-review-20261025/quantile-runtime-*`. Integrate into subsequent training at a source-pinned boundary.

## Coherent eight-dimensional native search: large steps rejected

All 16 first-round candidates had lower harmonic return than the incumbent on the shared 64-game screen. The best 128-game finalist then returned 353/389 LEFT and 327/393 RIGHT on the complete practice bank, versus carry 360 and 354. No head accepted. Root and Sol jointly stopped the unchanged-radius second round; it had already begun, so its 26,879 additional transitions remain explicitly charged rather than described as a clean round boundary. Total: 593.30 seconds, 198,832 TRAIN transitions, 2,277 initial cameras, zero fresh observations, all exact. Root independently summed receipts. Source `bf40dc7` / main `3522589`; root `/workspace/m1b-experiments/pong-native-subspace-20261025-r1`.

Next selected bounded check reuses the same data/basis: half of the best 128-game direction, half of the best LEFT-improving direction, and one conservative (.05 legal contrast-RMS) reward-weighted combination of the top four mirrored 64-game pairs. Only a genuinely improved 128-game result earns a complete bank check and fresh comparison. This is an adaptive overshoot/combined-credit check, not a new blind radius draw. No further macro tuning follows a null.

The bounded backtrack also failed: half-best returned 56/64 LEFT and 53/64 RIGHT; half-LEFT 59 and 53; reward-weighted combination 59 and 52; incumbent 59 and 58. No candidate earned full-bank or fresh evaluation. Final combined root accounting: 231,980 TRAIN transitions, 2,661 initials, zero fresh observations, exact; parent receipt sums agree. Added probes cost 33,148 transitions and 384 initials. Total elapsed original clock including implementation pause: 969.57 seconds. Source `7bd3ea0` / main `1cd0be9`. Macro search closed; retain local-r4.

For the separate unselected action-credit option, parent checked a saved-policy simulator/vision snapshot continuation: cloning at transition60 and continuing93 steps reproduced all action records and the terminal graph. Copy took .039 seconds versus roughly one second for the suffix. Plain deepcopy first failed the pre-continuation graph check because Torch cache validity is process-local; existing context-cache restore helpers resolved this. Total proof cost including that failed check: 306 OLD transitions and two OLD initials, no new counterfactual actions or training labels. This is feasibility evidence only, not an interception gain. Artifact: `/workspace/m1b-experiments/pong-practical-review-20261025/snapshot-continuation-parity.json`.

## Single-action credit: real repairs discovered, radial cap blocks absorption

On64training contexts (24misses/eightsuccesses perphase),512alternative one-camera continuations produced nine mixed-outcome states. Two missed contexts perphase were rescued; one successful context perphase could be spoiled. The resulting five repair-state decisions are earned native-return preferences, not oracle action labels. Equal outcomes remain unordered; the highest-current-score successful action represents each preference against failed actions. Null intervention matched every saved suffix record and terminal graph.

The raw regularized fit made all five repair-state choices successful and satisfied every strict pair ordering, at .110252 legal contrast-output RMS. The .05 radial cap scaled it by .453506 and erased every repair choice, so the expensive bank/fresh check was skipped. This is an absorption restriction, not absence of causal learning signal. Trial47.005seconds;6,164baseline+6,417suffix cameras (including the null)=12,581TRAIN,64initials. Source `ae7732d` / main `9f7a07c`; root `/workspace/m1b-experiments/pong-action-credit-20261025-r1`. Per-worker bookkeeping RNG was not explicitly reset to the lean wrapper seed, but all64 deterministic cached native outcomes and the null continuation agreed; no rerun for bookkeeping alone.

Selected follow-on: one full native bank check of the already-defined raw optimum, with no new labels or fit tuning. Actual harmonic return with at most3pp phase regression decides selection; genuinely fresh paired128episodes follow only an improvement. Original capped result remains immutable. Source `efac81b`; root `/workspace/m1b-experiments/pong-action-credit-rawcheck-20261025-r1`, new explicit600second clock started23:00:03.355798411UTC.

## Raw causal fit deploys repairs but causes global interference

The uncapped native check returned LEFT360→362/389 (+16gains/−14losses), RIGHT354→305/393 (+14/−63). Rejected, no fresh evaluation. Three of four discovered causal repairs actually deployed (bothRIGHT, oneLEFT), so failure to reach the recorded repair states is not the principal explanation. Cost68,401TRAINtransitions,782initials,200.316seconds; parent summed all receipts. Combined discovery plus rawcheck:80,982TRAINtransitions,846initials.

A no-camera check on33existing current-success trajectories showed action changes on every trajectory, with median first divergence at frame0 in both phases. Early-quarter changes were25.3%LEFT/15.6%RIGHT; whole-trajectory19.2%/12.0%. These are comparisons on the same saved inputs, not executed new trajectories. Source: `existing-success-disruption.json` in the rawcheck root.

Selected next combines the cleaner causal credit with full success coverage: same14strict preference pairs, ridge.001 and sampled-state softanchor.25, plus unit phase/episode-balanced soft output preservation on33fullsuccess trajectories (3,738rows). No hard action/confidence/return veto and no radial shrink. Initial fit retains3/5repair-state choices, at.081472legalRMS, and changes10.44% of preserved-input actions. Actual native bank/fresh check is running; no performance claim from those fit statistics. Source `27ba385`; root `/workspace/m1b-experiments/pong-action-credit-soft-preserve-20261025-r1`, clock23:07:46.167924594UTC.

## Broad preservation improves practice, fresh transfer unresolved

The candidate returned375/389LEFT versus360 (+21gains/−6losses) and354/393RIGHT versus354 (+16/−16), retaining three of four causal repairs. Bank selection accepted it. The wholly unused fixed fresh probe was LEFT29→29/32 (2gains/2losses), RIGHT28→25/32 (1gain/4losses). These results do not establish transfer; preserve both r4 and the soft candidate. Runtime237.757seconds,67,987TRAIN+11,269freshtransitions,910initials; own completion passed.

Selected next is a fixed larger paired comparison of the soft candidate against r4, not another fit. Reuse reliability driver with an explicit labeled r4 baseline override:128newstarts perphase in two64cohorts,512native episodes, no retries or learning; ordinary32fullgames perhead only if soft reaches90%in both aggregate phases. The legacy `original_*` result fields are explicitly labeled baseline aliases for r4, not original RIGHTtheta6. Source `e1c21b7`; root `/workspace/m1b-experiments/pong-soft-vs-r4-reliability-20261025-r1`, clock23:14:50.499160732UTC.
