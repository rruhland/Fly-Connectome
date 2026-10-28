# Sustained-return continuation

## Progress at a glance

Same64 TRAIN starts, native opponent, normal learned action selection throughout:

| Policy | First returns | Later returns | Total hits | Games reaching4 returns |
|---|---:|---:|---:|---:|
| October25 carry |54/64|33/86 (38%)|87|4|
| Rewarded-sequence regression |53/64|49/100 (49%)|102|5|
| Soft action margins |51/64|84/127 (66%)|135|15|

The regression head also improved on32 fresh paired starts: later19/48→37/67 and hits48→68. The margin head has not had separate fresh qualification; its32 additional TRAIN starts gave26/32 first and36/58 later before further learning. All these policies still miss frequently. Current work aggregates opening and rally recoveries under the margin head.

## Starting evidence

The immutable October25 final head is the carry. Re-reading its existing32 full-game receipts (no new episodes) gives28/32 first returns and14/42 later returns. All32 ended in an agent miss. The matched r4 baseline had28/32 first and17/45 later returns. This reinforces that the prior milestone did not improve sustained play; the opponent was not ending those games early.

## Fresh TRAIN coverage, allocation1

Root: `/workspace/m1b-experiments/pong-rally-baseline-20261027-r1`. Same frozen carry, native.65 opponent, K1,20-second full-game cap,64 new unfiltered TRAIN seeds. No fit or candidate model.

| Incoming contact | Returns | Opportunities |
|---|---:|---:|
| First |54|64|
| Second |19|54|
| Third |9|19|
| Fourth |4|9|
| Fifth |1|4|
| All later contacts |33|86|

63 agent losses, zero opponent-miss endings, one timeout. The opponent returned all121 opportunities, including87 following our own hits. Maximum own-hit streak5. Allocation cost22,384 TRAIN transition cameras +649 explicitly counted parity replay cameras;64+1 initial cameras;71.74 wall seconds. The diagnostic logger reproduced the reference episode's records and terminal graph exactly, and weights/core sources remained unchanged.

Root independently analyzed the saved TRAIN traces, with **no additional camera executions**. A privileged diagnostic estimated reflected ball impact and reachable paddle bounds under constant extreme drive. At40 cameras before the53 later misses,32 remained approximately reachable; at20 cameras only13 did, and at5 cameras only1 did. Among the47 cases already incoming80 cameras before the miss, all47 were approximately reachable. These are diagnostic approximations, not an oracle policy, learning labels or proof of maximal reachability.

The nearest tracked visual entity was within3 pixels of the true ball in all53 later misses at each of the5/10/20/40-camera offsets; velocity was available in52–53 cases. This rules against wholesale ball disappearance at those sampled times, not against all visual errors. Some trajectories show the paddle pinned at the top for over a second while the policy continues to command up and the ball approaches below it. Others start moving correctly too late.

Current decision: test earlier and longer reward-verified action sequences, because single-frame interventions near terminal cannot correct most of these positions. Successful exploratory trajectories must subsequently be learned by the ordinary policy; finding a good scripted branch is only discovery, not deployed learning. Preserve current successful segments, not the failed suffixes. Root's diagnostic script/report are in `/workspace/scratch/analyze_rally_baseline.py` and `/workspace/scratch/rally-diagnostic-summary.json`; neither is a learner input.

## Recovery discovery, allocation2

Root: `/workspace/m1b-experiments/pong-rally-discovery-20261027-r1`. Twelve later misses selected from TRAIN, balanced by initial serve direction. Frozen prefixes were replayed exactly. At fixed earlier points, test one-camera alternatives in six contexts, then held commands of10/20/40/80 cameras, all three commands, stopping search after a successful next return per case. No hidden geometric target supplied actions or labels.

All12 cases had a reward-successful burst witness: four with20 cameras, six with40, two with80. The36 one-camera alternative trials in six contexts yielded zero returns;216 burst trials yielded12 successful witnesses. Exposure to duration was conditional on earlier failures, so this is not an unbiased duration comparison. Successful witnesses include the legal roll-in, override and released-policy continuation; they are whole reward-earned behaviors, not proofs that each forced action is individually optimal.

Cost24,959 actual transition cameras (4,240 prefix +20,719 suffix, including the exact null replay),12initial images,92.47wall seconds. No censored branches. A null later-miss replay retained previous hits but correctly received a next-return outcome of0 and matched the full native records/terminal graph. At this stage no learned candidate had been evaluated. Root approved one same566-coordinate analytic fit with broad successful-segment preservation, followed by64 native TRAIN full games. A separate untouched32-start paired full-game screen is reserved at `pong-rally-fresh-screen-20261027-r1`; it will not supply fitting labels.

## First learned update, allocation3

Root: `/workspace/m1b-experiments/pong-rally-imitation-20261027-r1`. Same566-coordinate analytic reward-earned sequence fit; no backpropagation, no new architecture, no held actions at deployment. Twelve successful witnesses supplied418 contrasting actions;87 baseline successful segments supplied soft preservation. Fit took0.263s and reproduced44.85% of weighted contrasting actions. This low absorption can reflect a regression objective that shrinks below action-choice boundaries; it is not yet evidence of insufficient representation.

On the same64 TRAIN full games, first returns54→53, later returns33/86→49/100 (38.37%→49%), total hits87→102. Games reaching2/3/4/5 returns changed19/9/4/1→27/15/5/2. The opponent returned136/136 opportunities; no opponent miss explains the improvement. Two timeouts versus one baseline. Only2/12 original witness contexts gained an extra deployed return. Thus there is partial closed-loop learning, but weak absorption and unresolved generalization.

Cost25,043 transition cameras +64initials,78.005s. Immutable October25 carry retained. Selected next: reserved32 paired fresh screen, then improve reward-earned action absorption or state coverage according to the result. Candidate SHA256 `979b986422b8ebcbd83abd53099c8fe70185ee644b2e523ef79689fa9f2f806a`. No global promotion or reliability claim.

## Fresh transfer screen, allocation4

Root: `/workspace/m1b-experiments/pong-rally-fresh-screen-20261027-r1`. Thirty-two reserved new starts, paired immutable carry versus first learned candidate, no fitting or retries. First returns29/32→31/32; later19/48→37/67 (39.58%→55.22%); total hits48→68. Games reaching2/3/4/5 returns14/5/0/0→20/8/6/3. All opponent opportunities returned; carry32 own misses, candidate31 own misses and1timeout. Paired bootstrap95% interval for extra mean hits/game was[0.125,1.125]; later-return-rate difference[0.0138,0.2994], noting policy-dependent later exposure. These are useful transfer gains, not reliable match-long control.

Cost28,142 transition cameras +64initials,86.677s. Root accepts r1 as a provisional useful sustained-return checkpoint, while retaining the immutable October25 control. Fresh data remains outside fitting. Next authorized allocation: one soft action-margin candidate using exactly the same TRAIN witnesses and successful segments, centered two-contrast coordinates, soft positive margins and soft preservation of earned action choices, analytic convex solves without backprop. This tests objective mismatch rather than expanding data and changing the objective simultaneously. Native64TRAIN reward determines whether it improves on r1; no parameter grid.

Sources informing this direction: [classification-based policy iteration](https://ijcai.org/proceedings/2003/226) motivates learning action preferences rather than numerical value targets; [Self-Imitation Learning](https://proceedings.mlr.press/v80/oh18b.html) motivates reuse of the learner's own successful experience. Our analytic classifier is not the paper's actor-critic implementation, and reward-successful sequences do not provide exact per-frame advantages.

## Action-margin learning, allocation5

Root: `/workspace/m1b-experiments/pong-rally-margin-20261027-r1`. Same original TRAIN data, same policy/vision, same normalization and ridge. Replaced numerical residual regression with soft pairwise action margins, solved by analytic active-set Newton steps (no autodiff/backprop). Two focused numerical derivative/fit tests passed. Fit converged in1.193s, reducing its convex objective1.63359→0.032897; weighted recovery-action agreement44.85%→83.69%. Soft preservation allowed16.56% changes on prior successful-segment actions, so deployed performance remains decisive.

On64 TRAIN games, first returns51/64 versus original54 and ridge53; later84/127 (66.14%) versus original33/86 and ridge49/100; total hits135 versus87/102. Games reaching2/3/4/5/6 returns34/24/15/8/3 versus ridge27/15/5/2/0. Six of12 witness contexts gained an extra deployed return. The opponent returned164/164 opportunities;56 own misses and8 timeouts. Cost31,298 transition cameras +64initials,94.367s. Head SHA256 `fe132e0dd2466494142ef4578b187e6856341dc1bb4a3803456ca16e7703b625`.

Root selected this as the next experimental TRAIN carry; the first ridge checkpoint remains the fresh-supported fallback, and the original export stays immutable. No fresh qualification claim for the margin head. Next authorized learning cycle: add32 new TRAIN starts, discover approximately24 rewarded recoveries spanning opening and later misses, aggregate successful experience, fit one same-margin update and evaluate on the expanded TRAIN bank. This follows learner-induced state coverage rather than repeating objective variants on the original12 witnesses.

## Aggregated recovery update, allocation6 — rejected

Root: `/workspace/m1b-experiments/pong-rally-aggregate-20261027-r1`. Added32 new TRAIN starts under the margin carry (26/32first,36/58later); combined96 baseline77/96first,120/185later,197hits. Found24/24 new successful recoveries (12opening/12later,17from new starts),15,460 search cameras, no censored branches. Seven remembered earlier first-return successes plus12 new opening recoveries covered all19 opening misses. Six unresolved older burst witnesses were also retained.

The same soft-margin fit converged in1.774s, but achieved only52.13% weighted requested-action agreement; changed18.68% of current successful-segment actions. Native96 fell to69/96first,53/119later (44.54%),122hits; only9/24 new witnesses gained an extra deployed return. Three timeouts, no opponent misses. **Rejected; fe132e0d margin carry retained.** Finding recoveries is currently easy; combining them without interference is not.

Costs:15,393 new TRAIN cameras,15,460 search,32,613 candidate;152initials. Learning-cycle wall156.305s, separate new-start collection49.712s. Root's saved-data label diagnostic found904 positive rows,854unique vectors, no exact positive-positive contradictory labels;9 exact positive-preservation conflicting vectors covered12 positive rows. This rules against trivial exact aliasing as the main explanation, not against nearby-state interference or unnecessary actions in rewarded sequences. Next decision compares excessive update size, objective underfitting and the566-coordinate optimizer restriction within the unchanged4389-feature policy. No new training architecture selected yet.

## Underfitting diagnosis and selected correction

Stored-data checks used no new camera observations. Halving the rejected update reduced requested-action agreement52.13%→20.21%, so simple damping was not selected. Removing preservation while retaining ridge0.001 only increased agreement to58.68%. Root challenged an interpretation of this as insufficient feature capacity: with the same566 features and positive data, a single positive-only diagnostic at ridge0.000001 attained99.96% weighted agreement in0.71s. It changed40.67% of preserved actions and is explicitly **not a deployable candidate**.

The diagnosis is strong regularization-induced underfitting; positive-only separability does not establish joint compatibility with all successful behaviors or generalization. Selected next: one full aggregate fit at ridge0.000001, retaining the original soft preservation/matching objectives, same data/normalization/margins/architecture, followed by96 native TRAIN games. No raw-feature recapture or feature expansion yet; no parameter grid.

## Weaker ridge deployment and recovery probe — rejected

Root: `/workspace/m1b-experiments/pong-rally-weak-ridge-20261027-r1`. Full preserved fit at ridge0.000001 converged in5.684s; requested-action agreement90.41%, successful-action retention81.84%. Native96 still regressed:69first,60/125later (48%),129hits versus77/120/185/197 for carry. Four timeouts, no opponent misses. Rejected. Cost33,444 cameras +96initials,102.843s.

Root performed two bounded follow-ups. A saved-data soft half-confidence preservation diagnostic attained64.08% desired actions and86.67% retained successful choices in5.46s; no game rollout or selected checkpoint. A24-context prefix-clamping probe kept the original recorded approach actions, then released the weak head at the successful discovery start with **no recovery override**:14/24 returns (8opening,6later), compared with11/24 extra returns under its ordinary full rollout (7opening,4later). Cost6,776 cameras +24initials,22.969s. This diagnostic cannot establish deployed performance; it shows prefix drift is only part of the recovery failure.

Selected next: one bounded raw-coordinate optimization test around immutable fe132e carry. Keep existing4389-feature linear runtime policy and vision; remove the566-coordinate fitting restriction. Recapture actual scaled sparse policy inputs only for needed reward-earned sequences, not invented raw features reconstructed from a projection. Use sparse analytic matrix-free Newton-CG, no autodiff/backprop; zero residual exactly preserves carry. One fit and96 native TRAIN games,600s execution budget, all recapture charged. This changes optimization freedom, not the deployed architecture. Parameter geometry differs, so results will not be described as a pure capacity proof.

## Raw-coordinate update, allocation8 — mixed and not promoted

Root: `/workspace/m1b-experiments/pong-rally-raw-optimizer-20261027-r1`. Actual sparse input capture cost35,328 cameras; root caught two seeds with both old burst and old first-return memories sharing a filename. Four source-qualified replay repairs cost974 more cameras. All valid captures retained. Total capture36,302 +106initials; unchanged native prefixes matched exactly.

Raw matrix-free fit reached its120s finite-downhill-iterate limit after29 steps, gradient norm2.80e-6. Requested-action agreement92.33%, earned-action retention89.44%, matching90.86%; this improves joint fitting over the566-coordinate fit but is not a convergence certificate. Native96:80first (carry77),96/170later56.47% (carry120/18564.86%),176hits (carry197),6timeouts, no opponent misses. Twelve of24 new witness contexts gained an extra return. **Not promoted as the combined carry.** Total79,182 cameras +202initials,372.857s; current best remainsfe132e.

Root benchmarked an exact optimizer improvement: restricting Hessian-vector products to the6,635 active rows among38,239 gave8.54× faster products with maximum output difference0. No new observations. Apply after the frozen allocation, and warm-start the same objective before deciding whether its residual numerical change warrants any further rollout.

Next learning mechanism: incremental four-repair update (opening/later and phase balanced), first removing forced-action chunks only when native reward verifies the recovery still succeeds. Fit these few repairs with all197 current earned-success segments; retain other witnesses for later instead of requiring simultaneous imitation of all37. This combines smaller learning increments and cleaner credit, not another parameter sweep. Same policy/vision, no backpropagation or privileged labels.

## Four pruned repairs, allocation9 — accepted incremental gain

Root: `/workspace/m1b-experiments/pong-rally-pruned-incremental-20261027-r1`. Four current rewarded recoveries selected with2/3/3/5 contrasting overrides, opening/later and initial phase balanced. Native-reward-verified chunk deletion retained all four returns while reducing total contrasting overrides13→5; forced requests20 per case→3/2/3/2. Cost2,203 cameras +4initials. An action-file dtype bug stopped the first launch before fitting; actions were validated integral/in-range and fit/native resumed from saved pruning on the same420s clock, with no repeated pruning observations.

Fit converged1.51s:100% requested-action agreement,99.52% earned-action retention,99.84% matching. Native96 improved77→80 first,120/185→134/201 later (64.86%→66.67%),197→214hits. All four targeted contexts gained returns:0→2,0→1,2→6,1→3. Thirteen timeouts, no opponent misses. Native cost49,379 cameras +96initials;211.837s total allocation wall including the recovery. **Root accepts selected.npy as the next provisional TRAIN carry.** No new fresh qualification yet.

The optimized same-objective continuation of the preceding raw fit converged7.36s with negligible action changes; no redundant native evaluation was run. This four-repair result combines cleaner credit, fewer simultaneous demands and raw weight freedom; it does not isolate one cause. Next: repeat current-policy discovery/pruning on new misses, preserve the newly earned214 segments, and evaluate the next small update. Use a reusable head-parameterized runner, not a new experimental framework per cycle.

## Repeated four-repair cycle, allocation10 — not promoted

Root: `/workspace/m1b-experiments/pong-rally-incremental-cycle-20261027-r2`, carried02b028f. Seven of8 sampled current failures had rewarded recoveries; four chosen pruned49→22 contrasting overrides. Fit converged2.94s,100% requested-action agreement,98.84% earned-action retention. All four target contexts improved, but native96 lost overall: first80→82, later134/201→125/197 (66.67%→63.45%), hits214→207;10timeouts, no opponent misses. Keep02b028f. Cost10,605 discovery +48,350 native cameras,104initials,186.649s.

Root implemented exact-action episode equivalence checking and validated against these two independently executed collections: all96 first-divergence predictions matched; the one fully unchanged episode matched updated q records, outcomes/events and raw inputs exactly. Only52 cameras would have been saved here: despite98.84% per-frame retention,95/96 complete trajectories changed actions. Median first divergence63cameras,11within5. Thus per-frame imitation percentages obscure widespread closed-loop effects. No new observations used for validation.

Selected next: one three-action reward-pruned repair with raw working-set action-choice guards on all currently earned successful prefixes, excluding old failed suffixes. Add matching actions from the one complete rewarded witness, so successful positive fitting can be checked for full trajectory reproduction. Diagonal ridge plus few positive hinge slacks keeps zero residual guard-feasible; lazy sparse guard generation avoids a dense8778-square Hessian. No hard old-q confidence locks; required guards are capped min(kappa,half oldgap). Check useful repair attainment before any native rollout. Major runtime architecture unchanged.

## Guarded single repair, allocation11 — gained19 returns without practiced losses

Root: `/workspace/m1b-experiments/pong-rally-guarded-repair-20261027-r1`. Protected32,049 currently earned/matching action rows (64,098 inequalities) and requested three changed actions in one pruned later-return witness. The first90s working-set solve was unfinished; its uncontracted iterate already preferred all three actions, while feasibility contraction lost two. This was not treated as an impossibility result. One warm continuation of the same QP/data converged to full feasibility in12.11s/five rounds:1,689 working rows, zero guard violation, no old/matching action changes, and exact agreement with all393 actions of the complete reward-earned witness. No new training observations or architecture changes in solving.

The original420s allocation ended with63finished logical games and three partial executions. An explicitly recorded180s fixed-head completion ran only33 missing/partial starts; old partial receipts and their camera costs remain archived. Final native96:81/96first (80before),152/217later70.05% (134/201before),233hits (214before),16timeouts, no opponent misses. **Every practiced seed retained its previous return count**; the requested case went2→3. Six whole episodes were reused with exact-action certificates. Actual executed cost49,915 cameras +93initials, including all partial/repeated prefixes;4,598 logical cameras were reused, not newly observed. Overall wall531.776s across the stated phases.

Guarded headdf1f49fb… is useful TRAIN progress. Before stacking more guarded steps, selected an untouched32-start paired raw-policy screen versus02b028f at `pong-rally-guarded-fresh-20261027-r1`,17LEFT/15RIGHT, no fitting. This tests generalization rather than assuming exact retention on practiced trajectories transfers to unseen states.

## Guarded fresh screen and next curriculum

Root: `/workspace/m1b-experiments/pong-rally-guarded-fresh-20261027-r1`. Paired32 new starts,02b carry versusdf1 guarded head: first26→28/32, later34/57→40/66 (59.65%→60.61%), hits60→68. No opponent misses. Fresh paired uncertainty remains wide; this is modest transfer, not reliable unseen rallying or a strong later-rate improvement. Cost30,419 cameras +64initials,82.470s. Root selecteddf1 as the next continuing carry.

After preserving that fresh result, root explicitly admitted this32-start cohort to TRAIN. Its actualdf1 raw trajectories can join the96 current bank at zero new observation cost:128games,301earnedsegments,109firstreturns,192/283later. It must never again be called untouched qualification; future tests use new starts. Existing older large validation remains untouched.

Next proposed/authorized goal: sequentially learn about three individual reward-pruned repairs, checking each full witness and guarding all old earned prefixes plus already accepted new reward traces, then evaluate the resulting single model once on the full128bank. Rebase after each accepted update; defer infeasible/unfinished witnesses while retaining accepted progress. This avoids both simultaneous incompatible requests and a full-bank rollout after every tiny improvement. Sol is reviewing the concrete reusable curriculum implementation.

## Sequential curriculum underway

Root: `/workspace/m1b-experiments/pong-rally-guarded-curriculum-20261027-r1`. The first two sequential updates passed exact old-action guards and whole rewarded trajectory checks: an opening-RIGHT three-action repair (61.93s fit) followed by a later-LEFT one-action repair earning the sixth return (72.87s fit). The latter certificate covers all884 actions. The third later-RIGHT case retained14 differing actions after bounded discovery/pruning and did not fit within its90s allocation; it was deferred without losing the first two updates. Final128 native evaluation is running; no final promotion yet.

Parent measured the initial admitted bank's conditional return-depth curve from saved outcomes: first109/128 (85.2%), second76/109 (69.7%), third53/76 (69.7%), fourth31/53 (58.5%), fifth22/31 (71.0%), sixth10/14 (71.4%). Timeouts are censored, not misses. These20-second single-rally trials cannot establish full-match reliability; eventual qualification must include longer fresh rallies/multi-point matches. Artifact: `pong-rally-TRAIN-bank-20261027-r1/parent-return-depth.json`; zero new observations.

Next efficiency hypothesis: end the guarded solve as soon as an analytically feasible contraction exactly reproduces every requested action while preserving all guard choices, retaining the full-witness check. Optimal confidence margins are not required for acceptance. Benchmark on saved data after the frozen cycle before using this in a larger repair sequence.

## Sequential curriculum completed — accepted

Head `1f45726341691dcee3f5b573e70f3d45238f1897d63aace921d477f159c0eafc` selected at `pong-rally-guarded-curriculum-20261027-r1/selected.npy`. Two accepted repairs accumulated:128 TRAIN games improved301→315 total returns,109→112 opening returns,192/283→203/293 later returns (67.84%→69.28%),18→22 timeouts. No practiced seed lost a prior return. Target138732597 improved0→2 and7831632 improved5→6. Opponent returned374/374. Fifteen full games reused exact-action certificates;113 actually executed. Acquisition10,843 cameras +8 initials; final62,040 cameras +113 initials,169.32s native. Total elapsed616.38s since first acquisition includes implementation gap; counted phase sum464.41s. Third witness14 actions was not learned; seven tested chunk deletions failed, but other subsets were not exhausted. Neither necessity nor capacity failure is established.

Root approved next block: benchmark early feasible repair stopping using saved step1, refresh128-bank outcomes/earned memory to selected head without new observations, then seek6 accepted repairs within8 attempts before one full evaluation. Rotate opening-LEFT first; defer the just-failed seed for this block. Keep runtime/vision unchanged, no backprop or privileged targets; preserve every accepted repair and guard. This is TRAIN progress, not fresh reliability.

## Guard confidence and throughput investigation

The early-feasible-stop implementation passed exact guards/full-witness checks on saved curriculum step1 but took63.48s versus61.93s reference; no speedup was measured. The next6-accepted/max8-attempt block launched at `pong-rally-guarded-curriculum-20261028-r1` with this stopping rule, a refreshed128-game/315-segment bank, and otherwise unchanged guard margins. Opening-LEFT step1 found a one-action reward recovery but did not learn it in the bounded fit.

Parent independently tested whether retaining old confidence, beyond retaining choices, was unnecessarily restrictive. A separate saved-data solver snapshot capped required guard margins at `kappa*1e-4`, still no larger than half the old margin. On the original three-action curriculum step1 it reproduced the complete reward witness with zero old guard action changes in20.12s, versus61.93s reference. No new camera observations; diagnostic weights were not promoted. Artifact `pong-guard-margin-benchmark-20261028-r1`. This supports testing softer confidence requirements while keeping exact action retention, rather than treating numerical failures as architecture limits.

A workspace restart interrupted the larger block during step2 fitting. Completed step1 and step2's reward acquisition remain saved; root authorized a minimal resume with no repeated acquisition and explicit numerical restart overhead. The ongoing block retains original margins.

Independent possible generalization direction: [Invariant Transform Experience Replay](https://arxiv.org/abs/1909.10707) uses task symmetries to reuse feasible trajectories. Root inspected our visual transport: grid reflection and velocity-channel exchange are plausible, but frozen perception, forecast behavior and bin boundaries prevent assuming raw-feature mirroring is valid. Any later augmentation must validate actual mirrored camera/action trajectories; no synthetic successes or architecture changes have been adopted.

The newer opening-LEFT failure was not rescued by the smaller guard margin (51.19s) or a warm lower-ridge fit1e-8 (1.91s); both saved-data probes used0 cameras and were not promoted (`pong-guard-margin-benchmark-20261028-r2/r3`). Sol independently found an exact conflict: its requested action0 input exactly equals an old earned first-LEFT input for seed3351729059 requiring action1. This explains why that witness cannot coexist with the chosen old-action protection. It is not proof that the return cannot be learned: a different successful trajectory can avoid the conflicting command. Next acquisition should cheaply screen exact conflicts before spending a solver budget. No further probes on this witness are allocated.
