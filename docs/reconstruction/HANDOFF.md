## Active learning goal — b9 rejected, d7 retained

Latest fresh result: b9 TRAIN330vs323, but NEW32 fresh61vs84 returns and54.24%vs69.23% later rate. Root rejects b9. Current selected model remains d7d7d872 at `pong-rally-guarded-curriculum-20261028-r2/selected.npy`.

Same Sol agent `/root/pong_one_pass_research` now authorized up to30min goal-directed later-return improvement: confirm prior EXIT; preserve fresh evidence; admit32 **d7/carry** trajectories from `pong-rally-b9-fresh-screen-20261028-r1` to TRAIN (160cases/407earnedreturns); use new seeds for future qualification. Root admission authorization JSON is saved there. Agent may choose smaller/medium implementation and experiments without approval; no backprop/privileged inputs/major architecture change. Focus actual later returns, control unseen-state regression, use empty solver working sets and native-target acceptance, reuse experiences, avoid endless failing variants. Parent reviews and steers; user requested continued work until reliable learned returns, so do not finish after allocation.

Current candidate avenue: investigate why coefficient-space fitting causes unseen policy changes; feature-RMS/function-change or spatial regularization are parent hypotheses, not mandated implementation. Stable d7 fallback retained. Bridge remains another thread's task. Historical updates below superseded.

## Active evaluation — b9 outcome candidate

Current selected carry remains d7d7d872 (323 TRAIN returns). Parent's seven-game native probe of `b9f1b98a6ee1b352897d51736b6b28bb6b2848568d878803e19dc3cee4796bd4` found target8325019400→3,3519222253 4→5,2732613884 5→4; other4 retained. Net+3 with one known regression. Root chose to evaluate overall learning outcomes rather than discard useful improvement for a self-imposed per-seed zero-loss rule. d7 retained as safer checkpoint.

Same Sol agent owns active fixed128 check `/workspace/m1b-experiments/pong-rally-b9-native-check-20261028-r1`, private exec48326/PID5063, source520d11e (mainc4b63b1). Seven actual b9 games reused. Conditional32 fresh paired screen at `pong-rally-b9-fresh-screen-20261028-r1`; root gate is TOTAL gain + no FIRST count loss, tiny conditional later-rate dip is not a veto. Loaded runner has stricter later-rate gate; if it skips solely for that reason, resume agent to launch SAME declared fresh schedule manually. No fitting/admission until root reviews. Agent parks while processes run; use followup_task to wake.

Prepared next learner (held pending candidate selection): empty cross-witness working sets, smaller guard confidence cap, cached reward experiences with exact conflict checks, optional actual-native-target acceptance when full exploratory-action imitation fails. Source60993b2 integrated as7e6cd6d. Fresh d7 TRAIN bank `/workspace/m1b-experiments/pong-rally-TRAIN-bank-20261028-r2/admission.json`,128contexts/323earnedsegments. Keep d7's five-return native trace for2732613884 as recovery experience if b9 selected. Goal still unmet; continue orchestration, bridge separate thread.

## Current continuation — 2026-10-28

Selected carry `/workspace/m1b-experiments/pong-rally-guarded-curriculum-20261028-r2/selected.npy`, SHA `d7d7d872c1acde4b34748240c2ce9bf9d37af4c65ff2cd1f25d753f19d01fe61`.128 TRAIN:113 first,210/299 later,323 hits,24 timeouts; no per-seed losses vs315-hit prior head. Goal still unmet; user asks continue until reliable returns or stop.

Same Sol agent `/root/pong_one_pass_research` running zero-camera empty-working-set benchmark at `pong-guard-working-set-benchmark-20261028-r1`, private exec17524. It reconstructs failed replay step2 around currenthead. If successful, verify all refreshed323 earned segments before reusing candidate. Await result/root nextallocation. Agent implements/executes; root reviews directions/results. Bridge separate thread.

Guarded replay now screens exact positive/old-action input conflicts and uses smaller confidence cap `kappa*1e-4`. Previous larger block stopped after3failedfits, preserving4acquisitions; reuse those experiences. Current accepted head learned one further recovery, other fits likely slowed by inheriting thousands of inactive working constraints across different witnesses. Keep within-same-fit warm starts; test clearing cross-witness state before bigger changes.

Historical updates below are superseded.

## Active continuation — 2026-10-28

Current selected carry: `/workspace/m1b-experiments/pong-rally-guarded-curriculum-20261027-r1/selected.npy`, SHA `1f45726341691dcee3f5b573e70f3d45238f1897d63aace921d477f159c0eafc`.128 TRAIN:112 opening,203/293 later,315hits,22timeouts; no per-seed loss against df1 baseline. Two sequential guarded repairs accepted, third deferred. Full result in sustained-return-results. Goal still unmet; continue until consistent learned returns or user stops.

Reused Sol agent `/root/pong_one_pass_research` authorized to benchmark early feasible repair stopping on saved data, refresh current128 bank, then seek6accepted/max8attempted repairs with one final128 evaluation. Root explicitly approved this next allocation. Do not duplicate its running work. Bridge belongs to another thread. Prior status below is historical.

## Active sustained-return learning (2026-10-27)

User authorizes continuation until match-long returns are learnably consistent or stop. Bridge belongs to another thread. Follow [plan](../plans/2026-10-27-sustained-return-learning.md) and [results](2026-10-27-sustained-return-results.md). Reuse Sol XHigh `/root/pong_one_pass_research`; root owns hypotheses, selection and review.

CURRENT continuing TRAIN carry: `/workspace/m1b-experiments/pong-rally-guarded-repair-20261027-r1/selected.npy` (df1f49fb…), same4389-feature linearpolicy/currentvision. Guardedrepair retained ALL96prior returncounts and gained19hits:81first,152/217later(70%),233hits. Fresh32paired02b→df1:26→28first,34/57→40/66later,60→68hits; modesttransfer, reliabilitynotmet. OriginalOct25export and02bfallback immutable.

That32freshcohort is NOW explicitly admittedtoTRAIN afteritsfreshresult waspreserved; futurequalificationmustnewseeds. Combinedcurrentbank128cases/301earnedsegments/109first/192of283later, no newcameras toassemble. Active next: Solreview/implementationof~3sequentialguardedreward-prunedrepairs, fullwitnesscertificateandold/newearned-prefixguardsforeach, thenONEfull128nativeevaluation. Rebaseafteracceptedpatch;deferfailedwitnesses. NoBP/privilegedinputs/newruntimearchitecture. Exactepisode-reusehelperavailable; source/core/config/RNGunchangedrequired, actualcachedvsexecutedcostsseparate. Olderactive/pausedsectionsbelowhistorical.


## GitHub synchronization and local artifact bridge (2026-10-26)

The accumulated work is published on `origin/reconstruction/m1b-continuation`. The user already runs the downloaded viewer on Windows and wants future source updates through GitHub plus artifact delivery into the local repo. `scripts/run_pong_viewer.py` runs Git checkout code with separately stored checkpoints; Windows newline conversion is tolerated by viewer source-pin checks without accepting code changes. [Bridge setup](../viewer/artifact-bridge.md) provides a stdlib-only authenticated receiver on loopback, private Tailscale Serve, and a checksum-verified upload client. It writes only under the chosen checkpoint directory, never source or training state. The user has Tailscale; cloud VPN/HTTPS permissions and secret configuration are still pending. **No real cloud-to-laptop connection or upload has been verified.** Do not repeat the earlier mistake of presenting cloud localhost as a user-accessible URL. No new training is active.

## Playable frozen viewer — no active training (2026-10-26)

The requested presentation app is integrated in `1dedf90`: [viewer instructions](../viewer/frozen-pong.md). It serves five real checkpoints, full native bot games, human right-paddle controls and current vision overlays composed after camera capture. It has no training or checkpoint-write endpoints. The main-worktree server was started on port 8767; restart instructions are in the guide if the process has ended. Ten viewer/legacy UI tests passed, plus Chromium interaction checks on desktop and mobile. All 12 exported checkpoint files remained byte-identical after interaction. No additional learning or research evaluation was launched; browser games are demonstrations only.

## Saved interception milestone — no active training (2026-10-25)

The first-interception checkpoint passed its fixed larger fresh test:125/128LEFT,117/128RIGHT versus r4's118/113. Across all160heldout starts perphase it is154LEFT/142RIGHT, so RIGHTvariation remains explicit. Fullgames:28/32firstreturns and11multiple-return games forbothheads, no rallyimprovement. This is operational interception success, not fullM1B/continuous-learning/from-scratch replication. [Milestone, visual and limits](2026-10-25-interception-milestone.md). Export: `/workspace/m1b-experiments/pong-interception-checkpoint-20261025/README.md`, headSHA184dd67a..., successfulsource27ba385/mainb90eddf; qualificatione1c21b7/mainf84b566. All workers stopped, same Solagent available for reuse. Future continuation starts from the exported head; r4fallback and allfailed experiments are preserved. No BP or privileged learnerinputs; majorarchitecture changes stillneed userreview. Older active/paused statuses and budgets below are historical.

## Active continuation — first larger reliability check (2026-10-25)

Current frozen carry: `/workspace/m1b-experiments/pong-local-repair-20261025-r3/selected.npy`. Practice 211/229 LEFT and 210/233 RIGHT; latest fresh 28/32 and 29/32, +6 paired returns with no losses. Same Sol agent is staging/running `pong-interception-reliability-20261025-r1`: 128 fresh starts per phase in two cohorts versus original RIGHTtheta6, conditional ordinary-game check if both aggregates≥90%. No learning during validation, no retries. First-interception goal still not declared complete; continue until fulfilled or user stop. See practical plan/outcome log for exact provenance.

## Active continuation — repeated repair learning (2026-10-25)

Current carry: `/workspace/m1b-experiments/pong-local-repair-weighted-20261025-r1/selected.npy`. Interception is still incomplete. Local exploration discovered 13 new rewarded returns; 4× fitting integrates five and gives practice 139/165 LEFT, 143/169 RIGHT, fresh 28/32 and 26/32. A working-set QP now solves in 1.23s rather than timing out at 30s. Next selected pass: 398-context bank, new missed-context discovery plus reuse of eight old missed successful trajectories, same 4× fitting, exact hard-guard conflict filtering. Same Sol agent implements; root owns direction/review. See current practical plan. Continue until reliable efficient interception or user stop; no old time ceiling.

## Active continuation — localized reward discovery (2026-10-25)

Current carry: `/workspace/m1b-experiments/pong-spatial-residual-20261025-r1/selected.npy`. Practical interception remains incomplete: about 80% on recent fresh games, with no clear fresh gain from the last two fits. The next authorized pass discovers new own-rewarded returns through legal-state-local weight mutations, retains complete successful action traces, and integrates them with constrained preference fitting. Same Sol agent `/root/pong_one_pass_research`; see the current practical plan and outcome log. User requests continuation until reliable efficient interception or user stop; old time limits are superseded. No backprop or privileged action labels; major architecture changes still require review.

# Fly-Connectome M1B handoff — scaffold research

## Active October25 engineering continuation

User renewed ongoing work until reliable, efficient interception learning or stop. The old two-hour and two-experiment limits below are historical. Use the [practical interception plan](../plans/2026-10-25-practical-interception.md): concrete miss fixes, a cheaper equivalent rollout loop, consistent training practice, and fresh-game checks. Preserve real partial gains instead of requiring every old statistical gate. No backprop; major architecture changes still need user review. Same Sol XHigh agent reused.

The lean rollout path is integrated and passed exact retained-trajectory checks with a 1.617× single-process speedup. Five completed practical blocks have not yet established reliable interception. Broad population search was flat; directed reward-earned action fitting improved LEFT but interfered with RIGHT; a constraint-aware fit produced a useful partial checkpoint. Current carry: `/workspace/m1b-experiments/pong-constrained-residual-20261025-r1/selected.npy`, with 75/101 LEFT and 84/105 RIGHT on the 206 contexts now used for training (predecessor: 69/101 and 84/105). Its latest fresh comparison was only +1/64 overall, so reliability remains unresolved.

Active next step: mesh the earlier reward-trained DIRECT and RIGHT skills through their own successful trajectories, using the existing linear constrained solver. No new random population is selected. Same Sol agent reused. Current integrated source is `4b9298e`; experimental QP dependency is OSQP 1.1.3. Runtime-only encoder-copy branch `1b4dddf` is parked. [October25 results](2026-10-25-practical-interception-results.md) and [session state](/workspace/m1b-experiments/orchestration-session-20261025.json) contain the evidence and current execution state.

## Previous research direction

The user now authorizes starting the next research target: **reliable interception
and sustained rallying**, building from the proven return learner. Follow the
[continuous-learning scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md).
Record competing hypotheses and outside evidence; reuse Sol6.1XHigh for independent
challenge and execution while Astra owns research selection and interpretation.
Winning is descriptive, not the current skill gate. Time-varying rewards remain an
integration hypothesis. Major architecture changes still require user review.

On 2026-10-22 the user authorized ongoing orchestration until stopped. Continue
bounded research cycles, with independent challenge and reassessment between
allocations. The previous no-followup stopping point is superseded. Major
architecture review and efficient resource use remain required.
**PAUSED after the renewed continuation. No candidate or worker is active.**

[Combined continuation results and visual](2026-10-23-continuation-session-results.md). Both completed confirmation roots are now losslessly archived after both raw audits; restore journals before a new raw audit. All heads, source, protocols and metadata remain plain.
The selected batch is finished; both independent completion checks passed for both
allocations. No further training is scheduled, regardless of time remaining before
the 18:49:30 UTC ceiling. Future work needs a new prospective allocation; never
restart either immutable clock or reuse partial-population fitness.

The [weighted-proposal confirmation trial](../plans/2026-10-23-confirmed-proposal-interception.md)
and final [executed-child follow-up](../plans/2026-10-23-executed-child-interception.md)
each completed six cycles and accepted zero updates. Both retained exact RIGHTtheta6
and correctly skipped all DEV under their registered unchanged-policy rules. No
held-out learning, preservation, competence or promotion result was produced. All
fresh total success differences were below two even without phase protection;
strict phase guards alone do not explain the failures.

Weighted proposals used 159,824 training transitions (157,443 complete-cycle used;
2,381 partial unused), 1,755 initials and 981.600 seconds. Exact-child selection
used 159,824 training transitions (154,313 used; 5,511 unused), 1,789 initials and
949.207 seconds. Total continuation cost:319,648 training transitions,
311,756 used / 7,892 unused;3,544 initials;zero evaluation;1,930.807 allocation seconds
excluding preparation, verification, archival and orchestration. Own/parent checks
passed13.029/2.731 seconds for weighted proposals and12.331/2.749 seconds for
children, all with zero observations/replay. Raw reads are complete; parent owns
explicit scoped archival, with heads/T/metadata plain and interrupted roots excluded.

The weighted proposals moved 0.07945–0.10 function RMS and children 0.25; applied
movement was zero. Confirming an actually executed controller did not overcome the
poor fresh gains. This is not a representation-capacity refutation or a held-out null
for rejected heads. RIGHTtheta6 remains the experimental accumulated-skill
candidate; DIRECTtheta9 and qualified GLOBAL ARtheta112 remain separate.
RIGHT-only acquisition and a combined reliable-interception recipe are unreplicated.
No failed morning or afternoon head is promoted. Learned ancestry remains 558,832
training transitions (532,296 used; 26,536 unused); interrupted 14,879 transitions are
separately paid unused exposure.

A future recommendation, not a selected allocation, is balanced acquisition directly
from DIRECTtheta9 with explicit new RIGHT acquisition, LEFT preservation and
full-game global guards. It would be a separate research lineage and path test,
not a way to manufacture gain from a weaker baseline. Larger discovery banks,
phase-local perturbations and information/history changes remain alternatives;
none is implemented or authorized by this result. No threshold, noise, rate,
initialization or architecture sweep follows automatically.

The completed morning session below remains historical. Its checks and archival
passed; no morning candidate was promoted. Preserved RIGHTtheta6 has learned
ancestry 558,832 training transitions (532,296 used; 26,536 unused). Interrupted
14,879 transitions are separately paid unused research exposure.

The [minimum-phase objective test](../plans/2026-10-23-minphase-interception.md)
finished with nine nonzero updates but no combined first-return improvement:
LEFT48→45/64, RIGHT44→42/64; equal-weight −3.91 pp, 95% interval [−12.50,4.69].
Three of eight gates passed. A secondary ≥3-return signal, 2→9/64 versus START,
is unreplicated and does not establish rally learning. The endpoint is a separate
research artifact, not a promoted learner. Actual cost: 159,830 training transitions
(153,978 used; 5,852 unused), 89,902 evaluation transitions, 2,374 initial images,
1,598.446 seconds. Own and parent checks passed in 16.702 and 3.558 seconds with
zero new observations/replay. Raw reads and verified archival are complete.

Earlier this session, [mean-balanced practice](../plans/2026-10-23-balanced-practice.md)
gained LEFT skill and damaged RIGHT skill. The [scalar interpolation test](../plans/2026-10-23-interpolation-interception.md)
then selected coefficient zero on all 320 fresh training episodes and skipped held-out
games under its registered early-stop rule. Neither result qualifies a combined head.
Preserve RIGHTtheta6 as the main experimental candidate, DIRECTtheta9 as its direct-
skill reference, and qualified GLOBAL ARtheta112 separately. Heads, projections,
protocols and verification reports remain plain. The original interrupted run remains
immutable, with its paid partial observations separately charged and unused.

The two completed fresh TRAIN comparisons above addressed proposal-to-center
mismatch without producing accepted updates. They supersede the morning recommendation. Finite-bank uncertainty, initialization and information
limitations remain unresolved; the results do not prove incompatible representation.
No new architecture change is selected. Parent owns the final visual,
progress map and session closure; this handoff records execution and verification.

**RIGHT-first curriculum completed: strong acquisition; joint preservation unfinished.**
[Checked protocol/result](../plans/2026-10-22-opponent-return-interception.md).
RIGHT19→44/64 (+39.06pp,95%CI[26.56,51.56]); FULL first31GLOBAL→54/64
(+35.94pp,[23.44,48.44]). Four of five required gates pass; LEFT54DIRECT→50/64
has−6.25pp CI[−17.19,4.69], failing preservation without proving damage.
No combined/global promotion or replication claim. Six complete/nonzero updates,
159,833train (155,738used/4,095unused),89,386eval,1,759initials,1467.965s.
Both checks PASS (Sol13.692s,Astra3.219s,zero new observations/replay); freeze released.
RIGHT r1theta6 remains an experimental accumulated-skill candidate, not qualified global.

**Historical pause on2026-10-22: “find a good pausing point and stop.”**
Ongoing orchestration is cancelled. No candidate processes remain. Balanced fresh
practice was interrupted promptly by SIGINT to exec9708's coordinator; active
workers settled, EXIT130.161completed episodes paid14,879training transitions+
161initials; zero complete populations, zero updates, zero evaluation. These are
unused partial-population observations, not a null/failed learning result. Four
settled jobs'291transitions are included from durable journals despite absent
coordinator cost receipts. Source/driver/protocol and optimizer/head remain unchanged.

The last fully checked learning result is RIGHT r1theta6 above; its copy at
`pong-balanced-interception-20261022-r1/theta_at0.npy` is the last durable head.
Paid candidate lineage573,711train=532,296used+41,415unused; no new learned update.
Root`pong-balanced-interception-20261022-r1`, originalstart1792697965537299559ns;
final coordinator output83.9197s after start, later shutdown confirmation135.1914s
(includes bookkeeping; exact exit instant was not sampled). See pause-receipt.json.
All journals/checkpoints and immutable frozen protocol are preserved; no summary or
eligible evidence was fabricated. No observations/replay were added by stop checks.

**Continuation was requested on2026-10-23; do not restart this old driver,
reset its immutable clock, or treat the partial population as completed.** Any future
learning requires a new prospective allocation/design/budget; preserve the existing
partial artifacts. The [balanced selection](../plans/2026-10-22-balanced-practice-selection.md)
remains an untested proposal beyond partial exposure, not a replicated RIGHT recipe
or qualified combined scaffold. No global promotion or rally completion is claimed.

RIGHT cold archive completed before this allocation:1759DB5.2481GB→.6890GB,
56.697s; sidecars3.491s EXIT0 first. Every original byte reconstruct-verified,
zero observations/replay. Heads/T/metadata remain plain.

The existing frozenvision/linearclass demonstrably supports high direct interception;
prior localactor nulls did not establish representationceiling. Originalpopulation
changes objectivealignment/exploration/dose together, so no purestepsizecausalclaim.
Earlier directserveactor primaryfailed26→29/64 but fullretentionpassed; source90d7d73.

Earlier approved nonlinear comparison also failed learning/retention (38/64 carried,
40/64linear,37/64residual); sourcefb55f7d. Latent-score alternative rejected before
fresh training after retained dispersion18–19x higher despite correct toy identity.

Latest [visual baseline comparison](2026-10-22-visual-baseline-research.md): both
checks pass, source integrated343e881. First returns41/64 carried versus44/64 in
both arms: learning/advantage fail, retention inconclusive. Neither endpoint promoted;
retain temporal ARtheta112. Prediction error improved10.4% but update dispersion
increased. Durable plan and checked result are complete.

**Latest result:** [temporal exploration](2026-10-22-temporal-exploration-research.md)
completed112 joint batches: first returns31/64 start,32/64 IID,33/64 correlated.
Primary lift+3.125pp CI[-3.125,+9.375]: learning and coherence advantage fail.
Correlated all-return retention passes; IID retention inconclusive.79,641 training
(254 unused),48,063 evaluation transitions,32.21min. Both checks pass, freeze
released, source integrated8c0fed5. Continue orchestration; no scaffold completion.

**Latest completed 2026-10-22:** [paired-start actor](2026-10-22-action-credit-research.md).
First returns26/64→28/64,+3.125pp CI[0,+7.8125], insufficient incremental learning.
All-return retention passes. Only16/210same-start pairs changed outcome despite
202action-divergent pairs;15nonzero updates/105batches.39,862train (258unused),
27,531eval,17.85min. Both checks pass; source freeze released. Actor source
integrated5627a07. Exact-byte hash streaming integratedf703b8a,32tests pass;
retained-state hash microbenchmark1.50x faster, not an end-to-end claim.
The first actor also failed incremental first-return learning (35/64 both);
no scaffold-completion claim. Temporal exploration, visual precision and credit
quality remain competing explanations. Major architecture changes need review.

Previous batch below is completed history:
Current research artifacts: `/workspace/m1b-experiments/pong-scaffold-research-20261021/`.
The bounded first-return refinement is complete: **no detectable improvement**.
Both policies returned34/64 first balls; lift0pp,95%CI[-4.69,+4.69]. All-return
lift+0.39pp,CI[-5.83,+6.64]; retention inconclusive. Four complete populations,
79,835new training transitions and33,907evaluation transitions,19.5min execution.
Reporting-only recovery fixed a duplicate-field summary failure with no new
observations; parent evidence check passes. See
[refinement results](2026-10-21-interception-refinement-results.md).
No followup allocated. Reassess search credit versus visual/motor-state sufficiency
before another training allocation; do not automatically sweep perturbation sizes.
The previous one-pass evaluation is complete; the older executed-child training
allocation remains interrupted.

## Historical instructions and status (superseded by the direction above)

### Previous one-pass instruction (completed)

The user now requests **one** hypothesis/research/experiment/interpretation pass,
then stop. Astra owns independent hypothesis generation, outside-source synthesis,
experiment selection and interpretation; Sol 6.1 XHigh independently challenges
the hypotheses and implements the selected experiment. Do not continue the old
automatic allocation loop. Current pass artifacts:
`/workspace/m1b-experiments/pong-one-pass-20261021/`.

The earlier executed-child run is interrupted, not live: seven committed cycles,
two accepted updates, no final evaluation/summary. Its original clock must not be
reset. This pass proposes a separate evaluation of its last committed policy on
the already reserved, untouched development games; this cannot retroactively
qualify the interrupted study. Historical status below is retained as history.

**This one pass is now complete; stop here.** The separate fresh evaluation gave
endpoint17/64 wins versus native-start14/64 and original13/64. Primary lift4.69pp,
95%CI[−4.69,+15.63], failing the registered practical improvement screen. No new
training was run;39,383 evaluation transitions+192initials completed in389.31s.
See [one-pass results](2026-10-21-one-pass-orchestration-results.md). No automatic
follow-up allocation, old-clock reset, or claim of M1B completion is warranted.

### Earlier continuous authority (superseded)

On **2026-10-21** the user explicitly authorized continuous research and execution.
The historical no-further-allocation / stop-after-negative language below is
**superseded**. Continue bounded, prospectively registered allocations, independent
audits and evidence-based adaptation through the orchestrator. Do not end the
research loop merely because one allocation is negative. No backprop or privileged
policy inputs; genuinely major architecture alterations still require user review.

**Currently live:** [native-win executed-child learning](2026-10-21-validated-win-children-protocol.md)
at fixed .35, execution source **6df746f** (integrated **e46e743**) in isolated
`/workspace/Fly-Connectome-m1b-validated-child`. Artifact
`/workspace/m1b-experiments/pong-validated-win-children-20261021-r1`; original start
**1792585847619927483**, exec **9657**, coordinator **119077**, workers
**119079/119082/119083/119084**. Last audited native theta5 start; 2 directions,
12 discovery serves and 24 fresh validation serves, exact child accepted only at
net ≥2 wins. All observations count toward160k/30min training,20min evaluation,
50min overall; max24/min6 completed cycles, accepts may be0. Last incumbent only.
22 focused zero-camera checks and actual source/input identity pass; independent
stage check passes. Keep execution source frozen through both outcome audits;
never reset its clock or expand final streams. All negative/partial results stay.

**Latest completed:** [epsilon .02 pendulum continuation](2026-10-21-pendulum-low-exploration-results.md),
source **c36ebdd** (integrated as **dc40c48**). Both audits pass. On 64 fresh common
starts, .02 mean 2.502 s/4 caps versus reused .2 2.758 s/6 caps; paired mean
−0.255 s CI [−0.654,0.141], cap −3.125 pp [−12.50,6.25]. Both competence screens
fail; no epsilon benefit/equivalence or best-midpoint selection. A first returns
36→36/64 for .02 on explicitly reused development serves; no formal noninferiority.
20k new training +69,691 evaluation transitions, 875 initials, 20,171 learning
images in 1,738.553 s. Own705-pin/eight-zero-replay verification passed32.302s;
parent all-physics/TD/reuse/statistics passed18.535s. The initial external verifier
attribute error is preserved; candidates were unchanged. Exec25486/PID113443 and
all workers exited0; source released. No final streams. The subsequent native-win
validated-child allocation is live above. The separate frozen-survival B actor proposal is
external and unallocated. Continue iteration; M1B remains incomplete.

**Two independent training seeds now pass the unchanged hit-return DEV screen.**
The [first curriculum](2026-10-21-first-return-curriculum-results.md) gave initial
14/64 → phase0 29/64 → final 41/64 first returns, final 63/114 all returns.
The [whole-procedure replica](2026-10-21-first-return-replication-results.md), source
**886d05d**, gave **6/64 → 37/64 → 39/64**, final **64/122**. Its primary lifts
**35.94 pp [15.63,46.88] / 26.58 pp [13.81,33.65]** pass; final-minus-owninitial
also improves both metrics. Additional replica phase1 benefit is inconclusive;
do not imply both stages' incremental gains were independently reproduced.
Blanking becomes constant down (original seed: no-op), showing whole-input
sensitivity under OOD input, not ongoing tracking or forecast benefit.

Replica artifact `/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/`
retains the [two-seed chart](/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/replication-summary.png),
all source/protocol/recipe pins, readouts, RNG and raw journals. Six first-return
plus four full-rally updates used 188,389/199,771 cameras; 11,382 partial rows
unused, next global12. Evaluation95,918; total43.20min. Parent audit221.52s verified
all295,689 raw rows/ten updates/phase carry/statistics; own zero-work recovery and
all2,213 pins passed10.58s. Unfinished-rally hit summaries were clarified in the
auditor, not candidate code; raw partial hits remained durable and unused. Extra
partial recovery passed0.314s. Coordinator PID6544 / exec56235 exited0; all candidate
workers and audits exited. **Source is unlocked.** No final streams were expanded.

**Feedback test complete and independently audited:** [results](2026-10-21-camera-feedback-results.md),
source415f066, artifact `/workspace/m1b-experiments/pong-camera-feedback-20261021-r1/`.
Both final policies on32 new shared serves: live first10/32 and15/32, held6/32 and11/32,
missing9/32 and8/32. Equal-policy first lifts12.50pp versus both interventions;
paired32-serve CIs[4.69,20.31]/[3.13,21.88] pass the prespecified aggregate criterion.
Seed1 missing effect is individually inconclusive. Later visual updates benefit
this fixed policy set; no forecast-specific claim or optimal opening-only control
impossibility follows OOD masking. New live rates are lower than previous cohorts;
retain that variability.192jobs/29,724cameras/272.00s, no training. Own six-mode
zero-work recovery/allpins passed2.93s; parent allphysics/onset/statistics audit
passed13.60s. Exec32120/PID14161 and workers exited0; **source unlocked**.

**Forecast pair complete, independently audited:** [results](2026-10-21-forecast-comparison-results.md),
launch source39fe1f9, artifact `/workspace/m1b-experiments/pong-forecast-comparison-20261021-r1`.
FULL (third unchanged seed)18→32→35/64 first returns,55/113 all; state-only
18→30→38/64,49/110. Both improve over their own initial policies but fail the
unchanged strict20pp/both-CI-positive strongest-control screen. FULL all-return
point lift19.9369342pp receives no rounding tolerance. FULL−STATE first−4.69pp
CI[−12.50,1.56],all+4.13ppCI[−3.57,11.60]: ablation inconclusive, not equivalence.
Full/state supported ranks183/103; all state-only mapped forecast columns exact0.
Total train199,702/199,688; newly executed eval97,513/47,786, plus47,496 reused
FULL control rows attributed zero new STATE work. Coordinator times45.64/40.72min.
Own zero-work/all-pins checks passed13.46/13.54s; parent audits passed140.02/119.65s,
all544,689 distinct candidate rows. Both coordinators/workers exited0. Source is
unlocked; final roots remain unexpanded. Hot/cold journal restoration manifests
preserve byte-exact raw evidence.

**Persistent-vision study complete and audited:** [results](2026-10-21-online-vision-drift-results.md),
launch source **cf163cd**, artifact `/workspace/m1b-experiments/pong-online-vision-drift-20261021-r1`.
Both fixed qualified policies completed2k exposure and2k prequential transition doses;
64 shared frozen BEFORE/AFTER probes each. First returns39→36 and40→41/64;
equal-policy first/all effects−1.56pp CI[−6.25,2.34] and−1.75pp CI[−7.54,4.05].
No clear benefit/harm and no noninferiority claim. Actual74,696 transitions +297
initial images;8,041 learning images,0 actor updates,1,209.59s. AFTER inference
median/p95 58.98/80.36ms versus BEFORE44.54/66.89ms;20ms target unmet. Own recovery/
all-pins audit passed25.42s; parent allphysics/counters/statistics passed61.44s.
Exec1956/PID78086 and workers exited0. Source unlocked; no final streams.

**Second-task reward gain with plastic vision, strict competence still unmet:**
[audited results](2026-10-21-pendulum-acquisition-results.md). Predetermined seed1
EXPOSURE-AFTER(2k) versus fresh generic prior,20k native-survival SARSA cameras each.
Frozen B means2.734/2.812s versus strongest simple1.343s; lifts+1.390s
CI[1.088,1.698] and+1.469s [1.187,1.763]. Only8/64 and9/64 reach5s, so both
fail the unchanged mean>=3s/50% cap screen. Shared32-state curves continue improving.
No clear experienced-vision advantage. Old A readout retained without retraining:
experienced first36→36/64 (same outcomes), fresh39→42; all-return effects inconclusive.
These reuse old online64 development serves and do not prove formal noninferiority.
Artifact `/workspace/m1b-experiments/pendulum-online-acquisition-20261021-r1`, launch
source **2055109** (integrated unchanged into main). Complete115,941 transitions,
1,230 initials,40,526 learning images in31.21min plus2.62s final sealing. Own706-pin/
nine-zero-work recovery verification passed24.80s; independent all-physics/mapping/
statistics audit passed29.76s. Exec1380/PID79135 and all workers/audits exited0;
source unlocked. The separate128+1-image profile matched every old record/state/chain;
fingerprinting dominates, not encoder copying. A subsequent byte-equivalent
[bulk guard optimization](2026-10-21-hash-guard-runtime-results.md), source75dce69,
reduced one warmed saved-prefix loop by23.8%; this is not a universal speed claim.
All512 benchmark record/state/chain rows remained exact. No final streams used.

**Frozen.35 score transfer complete, both screens failed:** [results](2026-10-21-pong-score-transfer-results.md).
Both original policies/priors,64 fresh paired serves, nine cohorts. Wins14/16
versus best control12; own-hit wins5/7 versus3. Lifts3.125pp CI[-6.25,7.8125]
and6.25pp[-3.125,12.5] miss the registered10pp/positive-CI gate. Nine natural wins
are identical across cohorts, so total/own-hit lifts are the same paired evidence.
No retraining/opponent sweep, no score-reward acquisition claim. All576 jobs/
75,899 new cameras completed539.247s. Own576-pin/nine-zero-work verification
passed5.984s; parent all-physics/statistics audit passed33.061s. Source10acd5e,
artifact `/workspace/m1b-experiments/pong-score-transfer-20261021-r1`; exec12221 /
PID93633 and all workers exited0; source unlocked. Postlaunch full-width metadata
check found no actual seed collision; future guards include high uint32 IDs.

**B ON/OFF continuation complete and audited:** [results](2026-10-21-pendulum-continuation-results.md),
launch source **51293d2**, integrated unchanged as **3c09df9**. Shared experienced
20k head/memory, new episodes/traces reset, +20k native-survival updates per arm.
Fresh 64 B survival: start 2.552 s/7 caps; ON 2.460 s/1 cap; OFF 2.440 s/5 caps.
Incremental mean CIs both cross zero; ON cap decrease is −9.375 pp
[−18.75,−1.56]. ON−OFF mean/cap do not establish a plasticity effect or equivalence.
Both remain above controls but fail strict five-second competence. Fixed 32 curves
are retained without midpoint selection. A first returns ON 36→34/64; OFF 64 trials
are byte/content/config-verified reuse, not new work. No formal retention margin.
Exactly 106,540 new transitions/1,126 initials/20,205 visual-learning images,
28.14 min within 20/30/50 caps. Own 706-pin/eight-zero-work recovery check passed
30.61 s; parent all-physics/TD/maps/stats/reuse audit passed 19.59 s. All candidate
and audit processes exited0; source unlocked. Artifact:
`/workspace/m1b-experiments/pendulum-onoff-continuation-20261021-r1`.

**Native-score learning complete, audited screen failed:**
[results](2026-10-21-native-score-search-results.md), launch source **4f974bb**,
integrated unchanged as **35cd0e9 + f6447b0**. Fixed .35/win-only search from the
predetermined first qualified return head: endpoint 18/64 wins (8 own-hit) versus
start 16/64 (6 own-hit), lift 3.125 pp CI [−6.25,12.50]; no established incremental
winning skill. All cohorts share ten natural wins, so total/own-hit contrasts are
one paired accounting result. Five complete populations; 199,853 training rows =
165,212 used fitness + 20,588 unused incomplete-population fitness + 14,053 anchors.
Last-complete theta5 retained; g05 skipped, next index 6. Measured training 28.93 min,
overall 39.89 min, no failures. Total 285,351 new transitions/1,755 initials;
own pins/optimizer/seven zero-work recoveries passed 34.55 s; independent physics,
manual optimizer and statistics passed 120.57 s. All workers exited 0; source unlocked.
Artifacts: `/workspace/m1b-experiments/pong-native-score-search-20261021-r1`.

**Current allocation:** reviewed epsilon .02 native-survival continuation from the predetermined
E20k pendulum head/memory, reusing audited .2 ON training/checkpoints as its control.
Only epsilon changes; 20k new treatment transitions, fixed 5k/10k/final exports,
fresh 64-state matched B evaluation, unchanged competence and paired mean/cap contrasts,
plus A retention with explicit old development-reference reuse. Static source:
`/workspace/Fly-Connectome-m1b-pendulum-epsilon`. All 16 zero-camera checks and actual
owned-input/reuse validation passed; parent clock-free stage check passed. Fresh
64 states/actions are pinned, with only the old environment/exploration roots
intentionally reused. Limits are 20 min training/30 min evaluation/50 overall.
Keep old ON/OFF on_train and 64 A journals plain for reuse verification. Subsequent
validated-child win learning is a separate reviewed design, not an allocated run.
Do not select old midpoints or claim class failure.

This is not complete M1B: score-winning/final evidence, online distinct-task
acquisition/retention and the B2 scope review remain. The [completion checklist](2026-10-21-continuous-m1b-plan.md)
now marks demonstrated two-seed return acquisition, fixed-policy feedback evidence
and the completed inconclusive forecast comparison explicitly.

Cold storage: the completed `pong-score-policy-search-continuation-20261016-r1`
DBs are now byte-verified `.db.zst` archives; metadata/readouts/source stay plain.
Its `RESTORE-JOURNALS.md`, hash manifest and `restore-cold-journals.py` provide exact
reconstruction before historical raw-journal audits. No live journals were removed.

**Completed bounded continuation:** [protocol](2026-10-21-direct-hit-continuation-protocol.md),
source **aea715b**, artifact
`/workspace/m1b-experiments/pong-direct-hit-continuation-20261021-r1/`.
Coordinator PID 3990 / exec session 55045 exited0; consult the
original manifest clock and durable journals before resuming after any restart.
Starts exact theta9, skips unused g09, first global draw 10. Same hit-only rule,
frozen vision/T/cap/.65 opponent. Max 16 new updates /160k TOTAL fitness+anchors /
35 training minutes; fresh64 nine-cohort evaluation, 20 evaluation /55 overall
minutes. Pre-update center/theta9 anchors are pinned every even global draw and
executed after the learning update; never fitness or checkpoint selection.
Original theta0 plus five simples remain primary; theta9 and blank are secondary.
Extra-learning evidence requires positive lower bounds against theta9 on both
return metrics, independently of primary qualification. No final streams.
All 23 focused checks and actual staged mapping/sidecar/resume proofs passed;
source/protocol/archive/seed exclusions are pinned. See `launch.json` and
`prelaunch-proof.json`. Complete pending fitness is reconciled before a restart's
budget check, persistent failures disqualify, and every original deadline remains.
The retained source/protocol is immutable; both independent and own verification
passed before source unlock. Continue after this bounded outcome
under the existing user authorization; no automatic stop after a negative.

**Frozen confirmation complete:** [results](2026-10-21-frozen-hit-confirmation-results.md),
source **5c9e219**, artifact
`/workspace/m1b-experiments/pong-frozen-hit-confirmation-20261021-r1/`.
That allocation's candidate workers exited. Theta9 first returns 34/64 and all 48/106 versus
original theta0 27/64 and 34/95. Primary lifts +10.94/+9.49 pp have intervals
crossing zero and fail the unchanged screen. Secondary full-minus-blank lifts
+31.25/+25.03 pp have positive intervals: whole-input dependence, not established
reward acquisition, forecast benefit or ongoing tracking. All 83,814 physics rows,
512 jobs, frozen weights, seed recipes, geometry and bootstraps passed independent
audit (69.56 s); actual zero-work recovery/all 512 pins passed (2.65 s).
The zero-camera startup sidecar repair retained the original clock/serves/weights.
Nine distinct focused checks and both immutable source archives are retained.

The reviewed continuation is active above. No final holdout or seed replication
yet; continuous user authorization remains in force.

**Latest allocation complete:** [direct hit-return search results](2026-10-21-direct-hit-policy-search-results.md),
source **2e20707**, artifact
`/workspace/m1b-experiments/pong-direct-hit-policy-search-20261021-r1/`.
Nine updates; 159,873 training cameras, 145,561 used and 14,312 unused partial-g09.
Frozen endpoint first returns 16/32 and all 21/51, versus own initial 8/32 and 8/40,
strongest simple down 9/32 and 10/42. Positive point estimates **fail the unchanged
screen**: first lift 21.875 pp (CI −3.125 to 31.25), all lift 17.367 pp (CI 1.772 to 25.359).
No acquisition acceptance or final holdout follows. All candidate workers exited;
parent independent audit passed all 192,340 physics rows, nine manual updates,
pending-draw accounting and exact bootstrap results in 152.82 s. Actual zero-work
recovery and all 1,171 complete completion pins passed before shared-source changes.

An environment restart interrupted the run. Parent resumed the same directory,
clock and configuration; 399 committed training cameras lacked settled task-cost
records, and 99 previously committed steps were exactly replayed. Preserve this
accounting and unknown interrupted compute tails. Source archive and23 focused
passing tests remain retained. The reviewed fresh confirmation is complete above;
it adds no training or final holdout. Continue under existing user authorization;
do not silently extend the completed run or request renewed user permission for
ordinary experiments.

**The visual continuity repair and its fixed SARSA screen are complete.** Read
[results](2026-10-21-continuity-repair-results.md): source 570222e, 24k training,
32 fresh serves per policy. Corrected SARSA first returns 11/32 and all 15/45 versus
constant up 11/32 and 14/45; no acquisition, paired intervals cross zero. Parent
verified 50,467 physics/reward rows, 24k TD updates and all paired statistics;
zero-work actual recovery and all 193 completion pins passed before shared-source
changes. Full artifact: `/workspace/m1b-experiments/pong-continuity-repaired-sarsa-20261021-r1/`.
The subsequent direct hit-return search is complete above.
Use the approved imports-only GC initializer once per fresh spawned search worker;
never freeze inside repeated session/chunk calls. No codec optimization was adopted.

Saved-action
replay established that a beam slot can alternate selected lineages while one
public ID fabricates velocity and forecast differences across distant objects.
The approved repair revises routing IDs on a selected-parent endpoint/time break,
clears slot context and retires old forecast credit, preserving the beam and normal
occlusion. Read the [repair protocol](2026-10-21-continuity-repair-protocol.md).
Historical learning comparisons remain nonpaired. The repair preserves current
coverage and retained forecast errors while abstaining on discontinuous histories;
it does not solve association itself. All experiments from that screen have exited.

No final holdout is authorized for adaptive tuning. The original endpoint's
post-draw RNG records skipped global index9 and next index10. A future separately
registered continuation must explicitly declare whether it adopts the retained
partial population or starts after that skipped draw; no partial update occurred
in the completed allocation.

## Previous completed camera-rate exploration comparison

The user revised the next step toward genuine reward learning, rather than an
online teacher diagnostic. The approved sustained-exploration study is complete.
No experiments are running.
**No further candidate allocation, replication or final holdout is authorized
automatically. No genuine Pong acquisition has been demonstrated.** The former
teacher recommendation below is historical, not the next default action.

Source **77310e7**, branch `reconstruction/m1b-continuation`. Both from-zero arms
used seed 736187003, hit-only scalar reward, frozen DCT/cap256, .65 opponent,
alpha .001/lambda .995/epsilon .2, hold1 and exactly 24k camera-rate updates each.
Independent redraws exploration each camera; sustained draws mode per ten-frame
block, holds only exploratory random actions, and recomputes greedy each camera.
Actual exploratory occupancy was 20.08% versus 19.76%, with median exploratory
action runs of 1 versus 10 cameras. Training produced 46 versus 60 contact rewards.

Fresh greedy evaluation on 24 new shared serves: independent first returns 6/24,
overall 8/32; sustained 3/24 and 4/28. Strongest control random-h1 achieved 10/24
and 12/35. Neither passed. Sustained-minus-independent paired intervals crossed
zero (first −12.5 pp, CI −37.5 to +12.5; overall −10.71, CI −31.14 to +11.47).
No natural wins occurred; opponent evaluation returns were 144/144. The experiment
improved knowledge of training persistence, not demonstrated competence. One seed,
sparse rewards and behavior-mode aliasing do not establish asymptotic impossibility.

Read [latest results](2026-10-16-sustained-exploration-results.md) and
[protocol](2026-10-16-sustained-exploration-protocol.md). Full manifest/source/
journals/endpoints/metrics/costs are under
`/workspace/m1b-experiments/pong-sustained-exploration-20261016-r1/`.
All jobs completed: 48,000 training + 25,649 evaluation cameras, zero recovery
replay, 1,345.97 seconds shared wall. The ledger now has 16 allocations:
602,819 training + 145,482 evaluation transitions, with diagnostic work separate.
Parent independently verified all 73,649 physical/reward rows, 48,000 TD updates,
24 paired first trajectories, full exploration continuity and exact paired
statistics in 57.12 seconds. Evidence:
`/workspace/m1b-experiments/sustained-exploration-parent-audit-20261016/summary.json`.

If the user chooses further work, prefer a new bounded same-class direct hit-return
policy-optimization protocol over more tiny exploration sweeps; prior search
failures and sample cost remain material. No launch is automatic. Preserve no
backprop, policy-input oracle exclusion, untouched final streams and all negative
results. Small/medium experiments remain within existing user authorization;
genuinely major architecture changes require user approval.

## Previous reliable-opponent return-learning screen

The user's next authorized investigation is complete. **No experiments are
running, no genuine Pong acquisition has been demonstrated, and final holdout
streams remain unused.** Do not automatically launch the historical staged 90k
DCT study, another seed, or a longer continuation.

Latest fixed study: opponent speed .65, hit-only reward (+1 own contact; zero
game-score and survival contribution), same frozen DCT/cap256/SARSA alpha .001 /
epsilon .2 / camera lambda .995, one matched training seed, hold1 versus hold10.
Both completed exactly 12,000 cameras. Training yielded only 22 and 29 hit rewards;
hold1 made 12,000 updates, hold10 1,228. The latter also changes persistence and
exploration coherence, so this is not a pure update-frequency comparison.

On 24 new fresh-phase ten-second serves per policy, frozen hold1 returned 3/27
incoming balls and 3/24 first balls; hold10 returned 13/36 and 9/24. Strongest
simple controls achieved 10/33 overall (constant up) and 7/24 first returns
(up/no-op). Hold10's +5.81 and +8.33 percentage-point lifts had paired 95% intervals
crossing zero; neither arm passed the registered return-learning screen. There
were no wins or natural wins. The opponent returned 105/105 evaluation balls,
including 49/49 after own contacts. First-incoming trajectories matched exactly
across all seven policies. This is a negative finite-budget mechanism screen,
not an impossibility result or standard Pong competence claim.

Read [latest results](2026-10-16-reliable-return-learning-results.md) and
[protocol](2026-10-16-reliable-return-learning-protocol.md). Source commit
**0ea413a**, branch `reconstruction/m1b-continuation`; full source archive,
manifest, journals, theta/W/cap, counters and costs are under
`/workspace/m1b-experiments/pong-reliable-returns-20261016-r1/`.
Parent independently verified all 46,838 physical/reward rows, 13,228 TD updates,
all 24 paired first trajectories and exact bootstrap statistics in 34.90 seconds:
`/workspace/m1b-experiments/reliable-return-parent-audit-20261016/summary.json`.

Total allocation: 24,000 training + 22,838 evaluation cameras, zero recovery replay,
757.32 seconds shared wall. Separate .45/.65 opponent feasibility and .65 vision
probes are diagnostic compute in the external cumulative ledger. The latter found
99.80–99.90% observed/predicted ball coverage within three pixels on 2,048 frames;
it does not certify all features or exclude rare faults. New training raw norms
were sometimes high, but norm magnitude alone does not identify their cause.
The ledger now contains 15 allocations: 554,819 training + 119,833 evaluation
transitions, with calibration and diagnostic reexecution recorded separately.

Recommended next distinction, **not allocated**: a quarantined online immediate
privileged-action-feedback test on the learner's own closed-loop states. Prior
cached mapping accuracy did not test that state distribution or adequate Bellman
Q values. Teacher feedback must remain diagnostic only, never policy input,
candidate initialization or acquisition evidence. Agree a new concrete bounded
allocation with the orchestrator under existing user authorization before work.
Small/medium experiments do not need renewed user permission; genuinely major
architecture changes still do. No backprop.

## Previous completed reward and vision investigation

Ryan resumed and explicitly requested informative reward and vision tests. The
staged90k/three-seed score-only DCT run was NOT launched; its old launch command
below is historical, not the next automatic step. No experiments are running.
No genuine Pong acquisition or untouched-final evaluation has occurred.

Completed NEW fixed reward diagnostic: same DCT/cap256/alpha.001/lambda.995/
epsilon.2/hold1/.25, one matched seed,12k cameras EACH. Fresh raw-score24DEV:
score8wins/0own-hit/2contacts; +1own-hit bonus9/1/7; +.1physical-second bonus9/1/3.
All had8natural wins and0timeouts. No-op was10wins/2own/9contacts; random9/1/8,
up10/2/7. No shaping arm exceeded simple controls or14/6; do NOT call the extra
contacts acquisition. Training rewards were explicitly separated from raw score.
36k train+7456eval cameras,659.59s shared wall,0replay. All43,456actions/physics/
reward rows and36k TD equations independently verified exactly. Cached readouts
remain state-varying, not proof of actual visual necessity or adequate linear Q.

Parent's separate frozen-vision audit:4096distinct+4096reprocessed cameras,
49.10+45.66s, frozen hashes unchanged. After startup, fresh ball matches~89%,
observed-or-predicted coverage within3px99.85–99.90%;8-camera ball forecast errors
1.79–1.94px beat persistence5.42–5.46 and encoder-CV2.88–2.95. Single-pixel
detections are intermittent; own-paddle y90%interval coverage is only42.5–61%.
These two trajectories do not refute earlier rare public-ID/velocity corruption.
Cached privileged immediate-action feedback learned83–86%excluded-rally mapping
across3seeds, but never rare no-op and never closed-loop reward-only competence.
Its282,900updates/31.61s/0newvision and quarantined coefficients are separate.

Read [combined report](2026-10-16-reward-and-vision-results.md) and
[registered protocol](2026-10-16-reward-credit-diagnostic-protocol.md). Source
commit3e79af5, local branch reconstruction/m1b-continuation. Full source/manifest/
configs/journals/endpoint theta+W+cap/parent audit are preserved at
`/workspace/m1b-experiments/pong-reward-diagnostic-20261016-r1/`.
Vision evidence: `/workspace/m1b-experiments/vision-integrity-20261016-r1/RESULTS.md`.
Latest docs commit follows3e79af5; read git log. Ledger now has14registered
allocations,530,819training+96,995evaluation transitions, named diagnostics separate.

Recommended next finite study, NOT yet allocated: same DCT hold1vs10 to test
stronger coherent motor consequences with per-camera vision/journaling preserved,
proper SMDP gamma/lambda.995^10 and explicit update-count confound. Physics-only
legal-PD feasibility:13/24wins(5own) at both cadences, but hold10 lost8rallies
versus0 and reduced contacts82→42. It is feasible, not an improvement guarantee.
Online immediate-action-feedback oracle diagnostic is a fallback, not a candidate
reward-only learner. Do not launch further training/holdouts or alter architecture
without a new concrete allocation and orchestrator review under existing user
authorization. Small/medium experiments do not require renewed user permission;
genuinely major architecture changes still require user approval. Keep no-backprop
and policy-input oracle exclusion; shaping was scalar and explicitly diagnostic.

## Historical night pause (superseded; retained for provenance)

Ryan requested a good stopping point. **No experiments are running. Do not launch
the prepared DCT study or start its actual manifest clock until continuation is
requested and the final coordinator review below is complete.** No genuine Pong
acquisition has been demonstrated; all untouched final seed streams remain unused.

Latest completed curriculum:13/24wins(8natural+5own-hit),11loss/0timeout/15hits,
versus original initial10/24(2own-hit), FAILED unchanged14total/6own-hit gate.
Five stationary and seven .25 target updates used82,462 of89,712training cameras;
7250partial cameras never updated weights. Fresh eval4201cameras,1294.47seconds
shared wall, zero replay. Independent768-rollout reconstruction matched all
terminal-score updates, directions/candidates/RNG, phase timestamps/skipped draws
and actual frozen endpoint. Preserve the negative:
`/workspace/m1b-experiments/pong-stationary-curriculum-20261016-r1/{summary.json,optimizer-lineage-audit.json}`.

Main branch `reconstruction/m1b-continuation`: curriculum result/docs commit
**85dbab4**, validated DCT implementation commit **ad81c975cba60a92ade08955c58c6167f406b568**.
Its isolated preparation commit is9e79d50 on `reconstruction/m1b-dct-sarsa` at
`/workspace/Fly-Connectome-dct-sarsa`. The new handoff commit is the latest local
commit after ad81c97; read `git log -1` or the staged artifact's `base-commit.txt`.
No push/PR occurred. Main source is integrated and locally committed; the isolated
checkout is retained for provenance, not a separate live experiment.

Prepared ONE frozen DCT-SARSA condition, all three fixed seeds
172900127/736187003/1597334677 concurrently:30k cameras each/90k aggregate,
alpha.001/lambda.995/gamma-per-second.95/epsilon.2/hold1/zero-Q/.25 opponent,
exact prior214×4389score-free T and post-projection cap256. Fixed30shared training
minutes/35overall, endpoint-only24DEV per completed seed,14/6 EACH; promote only
if all three pass. No replacement, best-seed selection, wall extension, further
policy-search allocation or final evaluation. Protocol and coordinator:
[protocol](2026-10-16-dct-sarsa-pong-protocol.md),
`scripts/experimental/run_dct_sarsa_pong.py`.

Validation is complete:17unique narrow projection/worker/gate/default-regression
tests passed, with repeats recorded separately. This includes pending-trace/RNG
crash recovery, immutable/NumPy-alias mutation rejection, same-boundary export
gaps, actual mapped fresh evaluator handoff, and completion-pin zero-new-work
recovery. Real-T preflight640diagnostic camera executions+23existing replay took
27.17s; raw mean128-step3.740s versus projected4.705s, so30k naive18.4min and
historical-scaled25.5min (early-state timing, no guarantee). Transformed1024-row
score-free norm² median14.87/max233.63,0cap activations. Full-T/nonzero-theta
mapped greedy actions matched1024/1024; normalized score discrepancy≤2.61e−16.
Four controls'11,551action/camera/physics/reset events matched exactly in10.72s.
Evidence: `/workspace/m1b-experiments/dct-sarsa-preflight-20261016/` and its
`report.json`, `full-matrix-mapping-check.json`, `baseline-projection-proof.json`.

Staged candidate directory: `/workspace/m1b-experiments/pong-dct-sarsa-20261016-r1/`.
It contains T/fit/descriptor, proofs/checks, prior negatives/ledger and the complete
tracked source archive. **It has no manifest, journal or started training clock.**
`source-snapshot.sha256` and `base-commit.txt` identify the final archived source.
The separately named `dct-sarsa-preflight-20261016/prepare-smoke/manifest.json` is
a disposable zero-observation validation fixture; NEVER use its clock/state for
the candidate allocation. Cumulative13registered allocations:494,819training/
learning+89,539evaluation cameras; named diagnostics/calibration separate in
`/workspace/m1b-experiments/cumulative-research-ledger-20261016.json`.

On continuation: verify clean tree/no live workers and source archive, then have
the parent review the final coordinator completion-pin fix (report+all semantic
journal tables+theta/W/cap hashes, exact30k rows before eligibility; unpinned
complete JSON requires actual zero-new-work recovery). Tests cover stale reports,
incomplete rows and tampering; no further benchmark/optimization is needed.
After that review and the user's continuation, launch from main with the existing
runtime, one Torch/BLAS thread per worker and explicit PYTHONPATH:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/workspace/Fly-Connectome-m1b/src:/workspace/Fly-Connectome-m1b/scripts/experimental \
/workspace/Fly-Connectome/.venv/bin/python /workspace/Fly-Connectome-m1b/scripts/experimental/run_dct_sarsa_pong.py \
  --directory /workspace/m1b-experiments/pong-dct-sarsa-20261016-r1
```

That actual invocation creates the durable clock. Preserve its source freeze and
every result/partial; no learning work is pending in a running process overnight.

## Current authority and state (2026-10-16)

The historical pause below is **superseded** by Ryan's explicit continuation and
authorization for experiments and smaller/medium changes. Do not treat the old
zero-count stop state as current. Major architecture changes still need review;
no backprop, policy oracle state or silent reward shaping is authorized.

Working worktree: `/workspace/Fly-Connectome-m1b`, on local branch
`reconstruction/m1b-continuation`, based on `b36d8709be2d09d7f11ec1afc3740ff78d64ce86`.
Accumulated intended source/tests/docs are preserved in a local commit before
the continuation source archive. Do not discard edits; no push or PR was made. Runtime is the other checkout's
`/workspace/Fly-Connectome/.venv/bin/python`, Python3.12.14/Torch2.14.1+cpu/
NumPy2.5.4, with `PYTHONPATH=/workspace/Fly-Connectome-m1b/src`; use one Torch/BLAS
thread per independent worker. The environment grants four CPU cores/16GiB.

Start with the [current result index](2026-10-15-acquisition-and-runtime-results.md).
The registered toy pilot and multiple bounded diagnostics/experiments completed.
Immediate toy acquisition passed; original delayed toy conditioning/exposure and
all Pong acquisition screens so far failed. A later exact cached-template probe
shows RMS/alpha.001 learns both delayed templates by1000episodes in all seeds;
this does not relabel earlier allocations or establish visual generalization.
A subsequent fixed centering probe was negative and is not promoted. Original .45-opponent Pong has strong finite
evidence of a return-to-score ceiling; further .45 training is paused, preserving
the original task. Explicit .25-opponent Pong is winnable with legal actions, but
the first raw 6000-transition screen merely matched no-op at5/12 and stopped.
No genuine Pong acquisition or final holdout use has occurred.

Latest completed allocation: the fixed [stationary→.25 curriculum](2026-10-16-stationary-curriculum-protocol.md),
artifact `/workspace/m1b-experiments/pong-stationary-curriculum-20261016-r1/`,
finished cleanly at13/24wins(8natural+5own-hit),11loss/0timeout/15hits. It failed
the unchanged14total/6own-hit screen; original theta0 was10/24(2own-hit).
Five stationary updates and seven target updates used82,462 of89,712training
cameras;7250partial-population cameras never updated the optimizer. Fresh eval
used4201cameras; shared wall1294.47seconds, zero replay. Independent cached
reconstruction of768full rollouts, all updates/RNG/phase boundaries and actual
frozen readout passed. This remains an unreplicated development point estimate,
not acquisition. Coordinator94805 has exited; its source archive/local commit
fddda83 remain preserved and its source freeze has ended.

Optional DCT-SARSA preparation is isolated at `/workspace/Fly-Connectome-dct-sarsa`,
branch `reconstruction/m1b-dct-sarsa`, based on fddda83. It adds an exact fixed
projection at the learner boundary, embedded immutable T persistence, and mapped
raw-weight evaluation exports. It is NOT an allocated training study. Narrow
tests and a bounded actual-T cost measurement must precede any prospective run;
no new SARSA training allocation or untouched-final evaluation is authorized yet.

The [paired30k exposure/sharing study](2026-10-15-pong-exposure-sharing-protocol.md)
failed numerically before either endpoint: raw committed20,773 before nonfinite
weights; sharing was explicitly resource-stopped at23,400 after finite divergence.
No trained endpoint evaluation occurred. Exact replay captured public-ID
hypothesis switches producing large finite velocity inputs before trace/weight
explosion. Preserve this failure and its post hoc resource-stop amendment.

The [fixed radial-cap study](2026-10-16-radial-cap-pong-protocol.md) completed
30k stably, but its fresh24-rally endpoint failed:9wins(8natural+1own-hit),15losses,
0truncations, four contacts, versus strongest simple baseline10wins(2own-hit).
It used30,000learning+2534fresh evaluation transitions in1215.19seconds, with no
replay or replication allocation. Artifacts:
`/workspace/m1b-experiments/pong-raw-cap-20261016-r1/`. This establishes numerical
robustness at the allocation, not genuine Pong acquisition.

The separately registered [capped block-RMS comparison](2026-10-16-capped-block-rms-pong-protocol.md)
also failed its fixed30k endpoint:8/24wins, all natural,16losses,0truncation,two
contacts, using30,000learning+2162fresh evaluation transitions in1221.49seconds.
Both capped arms were finite; neither qualified or received replication.
Frozen epsilon.2 raw and bias-only controls subsequently both won8/24 with no
own-hit wins (4289camera transitions,95.28seconds shared wall, no updates).

The completed [paired alpha/hold study](2026-10-16-alpha-hold-pong-protocol.md) is
retained at `/workspace/m1b-experiments/pong-alpha-hold-20261016-r1/`: raw+cap256,
alpha.0001/hold1 versus alpha.001/hold4, same firstseed/.25physics,30k camera
transitions each, fixed24DEV endpoint. All cameras still update vision and commit
durably; held actions update/sample only at macro boundaries, with exact physical
discounts and lambda.995^4. Default hold1 records/weights/source-normalized complete
state matched the previous source. Hold changes were prepared in the isolated
`/workspace/Fly-Connectome-hold4` worktree and integrated only after both previous
coordinators ended and old full source archives were verified. Old journals must
use their original source archives. Check process/artifact status before resuming;
both endpoint evaluations are complete: lower alpha9/24(1own-hit) and hold4
11/24(3own-hit), both0truncations, below14total/6own-hit. No arm qualified. The
source freeze has ended after all archives/coordinators completed. A separate
worker-only warm-session benchmark is isolated in `/workspace/Fly-Connectome-warm`;
do not silently overwrite the archived driver. The opt-in warm-worker path is
now integrated after16focused tests; its1000camera matched continuation reduced
36.49to32.16seconds(11.85%, one-pair/load caveat), with exact full states/records/
weights/traces and unchanged per-camera durability/recovery checks. No replication/final holdout
is automatically allocated. Nonfinite guards reject invalid
updates before journal commit; discard failed live sessions and recover the last
valid journal state.

A NEW quarantined closed-loop witness completed under
`/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-closed-loop-capacity-20261016/PROTOCOL.md`:
prior fixed fit recipe,12DEVrallies against legalPD teacher,180second shared wall,
no coefficients exported or candidate use. Camera-only readout8/12wins(5own-hit),
4loss/0timeout/19hits; teacher8/12(5own-hit),0loss/4timeouts/44hits. It completed
in105.85seconds, providing a positive existing-class capacity witness, NEVER
reward-learning acquisition or teacher equivalence. Full records/source/fit recipe
and coefficient hashes are retained. The subsequent score-only policy-search
allocations described below are completed negative studies; no further allocation
is automatic.

The completed [score-only smooth search](2026-10-16-score-only-policy-search-protocol.md)
is preserved at `/workspace/m1b-experiments/pong-score-policy-search-20261016-r1/`.
Seven updates used56,089 of59,903training cameras;3814partial-generation7 cameras
were excluded. Fixed endpoint12/24(4own-hit) versus original initial10/24(2own-hit)
FAILED14/6 gate. No acquisition, replication or final evaluation. Read-only audit
reconstructed all448scored rollouts, arrays, updates and RNG exactly. A separate
restricted-class oracle witness won6/12(3own-hit),4loss/2timeouts; usable capacity,
not reward learning. Oracle coefficients remain quarantined and unexported.

The completed [fixed continuation](2026-10-16-score-only-policy-search-continuation-protocol.md)
is authorized at `/workspace/m1b-experiments/pong-score-policy-search-continuation-20261016-r1/`.
Its coordinator exited cleanly; source freeze ended. Endpoint10/24(2own-hit)
FAILED14/6gate, matching original initial and below theta7.14newupdates/21total,
119,831NEWtrain+3486eval;336unusedpartial cameras,1642.78seconds. Cached audit
exactly reconstructed896newrollouts, updates, RNG and endpoint. FunctionRMS.40079,
noise/policy ratio.624 and25%actiondisagreement rule out an obvious exploration
collapse. No replication/final use. Its procedure starts theta7/postdraw partialg7
RNG, skips that partial fitness, and uses global
indices8..23 for at most16additional updates (23totalupdates). Same frozen rank183
transform/366contrasts, noise.25/top4/step.10/hold1/.25physics. Fixed120k NEW cameras,
34mintrain/40overall, four workers, full per-camera journals/warm chunks; no partial
updates, no intermediate DEV or best-checkpoint selection. ORIGINAL theta0 plus
simple controls define14total/6own-hit gate; theta7's earlier score is descriptive.
All final seeds remain unused. Replication, if later authorized, reproduces the
WHOLE staged procedure from zero. No further same-rule .25 continuation is allocated.
The separate stationary→.25 curriculum also completed and failed, as recorded at
the top of this handoff; preserve every prior source archive.
The only new core-source change is exact-byte bulk graph hashing. Its source bridge
retains earlier baseline projection proofs; old journals use archived source.
Gates:37digest/search/evaluator tests,14continuation/durable/warm tests, plus
fresh-process16new+1replay proof, all exact records/full graphs. A local commit and
complete new source archive preserve code; outside-repo experiment data are excluded.

The separately isolated `ORACLE-DIAGNOSTIC-readout-capacity-20261016` audit found
86% excluded-rally/92% independent-calibration agreement with a legal PD teacher
using a generic linear readout, versus60%/58% majority baselines. Solver reached
its fixed iteration limit. No diagnostic coefficients were saved or supplied to
candidate policies; this is representability evidence, not learned control.

All artifacts live under `/workspace/m1b-experiments/`, including complete source
snapshots, not just hashes. Source changes require replay compatibility checks;
old hash-format journals require their original archived source. New binary digest
is version-pinned and fail-closed. Fresh Pong evaluation is separate from the
legacy trajectory-clone diagnostic. Default prior long banks are empty, so the
four-step forecast fallback is the actual context configuration. Reserved final
seeds308713321/1203078761/1983461927 remain untouched.

Latest cap/recovery/driver focused gate:48passed/15.58s, then10passed/3.92s for
composed block-scaling/cap/frozen-base/durable recovery before its launch.
Hold preparation then passed51tests/25.16s plus13main/coordinator tests/9.06s;
hold1 source-normalized complete state, all records and weights matched exactly.
The fresh-process hold proof used80new+1replay cameras; direct-vs-worker400-camera
proof matched all records, weights and complete state. Reused controls passed
19,903 camera/action/physics projection checks before launch. Active source archive
sha256 is `b8ae1ce0aaec2de877a647a47726fac8096175687972521ec3693888f073fb9e`.
Captured-sequence replay
reproduced both uncapped failure windows exactly and showed finite bounded capped
updates. Reused controls passed11,551 exact cross-source physics/action comparisons.
Earlier400-transition custom-checkpoint proof matched complete state, records and
weights, including final export recovery. Full timing remains in SQLite despite
compact progress reports. The known
baseline native exact-float failure below was not relaxed. No push/merge occurred.

Potential later execution optimization: the direct-step profile attributed about
.267/2.019seconds (13%) to encoder deep copy, including .203seconds across4529
Tensor deep copies. A tensor-clone/tree-copy implementation might reduce this
while preserving full ownership/alias contracts. It is a measured opportunity,
not implemented; do not edit the live study's source to pursue it.

## Historical paused handoff (preserved for provenance)

Ryan requested a safe stopping point so he can provide his own orchestration.
This supersedes earlier permission to automatically launch the toy pilot.
Do not launch training/evaluation from this handoff without new direction.

## Exact checkout and checkpoint

- Working repository: `/workspace/Fly-Connectome-m1b`
- Branch: `reconstruction/m1b-foundation`
- Code checkpoint: `b8449d5b6103f8ae531575222ac793d982ad2799`
- This handoff is committed in a following documentation-only commit. `git rev-parse
  HEAD` identifies that final handoff commit; source is unchanged from the code checkpoint.
- Original checkout: `/workspace/Fly-Connectome`, clean on `work` at
  `b52f06e9a4cf1664d8429e7098a6371eafb77bc7`.
- No public push or merge occurred. The working branch exists locally; Library
  review packets are evidence/source excerpts, not a guaranteed complete Git backup.

## Stop state and actual experiments

No registered toy training, competence evaluation, or Pong competence experiment
started. **Pilot counts: 0 learning + 0 evaluation = 0/4560 new transitions.**
There is no pilot manifest, pilot database, partial pilot allocation or pilot result.
Inspection of `/proc` found no running pytest, recovery-proof or pilot processes.
No background task/scheduler was installed. A transient exec transport disconnect
recovered on a read-only retry; it did not launch a job or leave a partial script.

Controlled unit/recovery diagnostics DID update synthetic policy state on separate
seeds. These are correctness checks, not competence evidence. Selected proof counts:

- Stage one: 48 vs 24+24 frozen-policy continuation, 96 transitions in successful
  proof, 193 including preceding diagnostics. It proved no learner recovery.
- Stage two: 32 vs 16+16 learning continuation, 64 new transitions, 12.37 seconds;
  nonzero weights, traces/q_old/pending action and RNG/perception/cadence matched.
- Storage proof: 16 vs 9 + crash + replay 2 + 7, 32 new + 2 replay transitions,
  7.84 seconds in final run. A prior same-sized diagnostic also passed.
- Toy-specific proof: 18 vs 10 + crash + replay 3 + 8 on diagnostic seed 99031,
  36 new + 3 replay transitions, 7.80 seconds. The interruption was inside delayed
  episode 2 at delay step 2. Side permutation/index, latched choice, pending action,
  all RNGs and the complete state/audit matched exactly after recovery.

These counts describe named proofs, not a sum of all repeatedly executed unit
fixtures. Proof JSON and test logs live under `docs/reconstruction/` and raw runtime
artifacts under `/workspace/m1b-verification/`.

## Implemented and validated

- True-online Sarsa(lambda), Dutch correction and q_old, physical-time discount
  indexing, pre-update next-action sampling/reuse, terminal/truncation distinction.
  Independent online forward-view tests compare each update prefix with overlapping
  nonbinary features, variable durations and action blocks.
- Pong boundary can suppress auto-reset, stops immediately at a score, and captures
  the genuine final image. Existing default Pong auto-reset behavior remains intact.
- Frozen nonzero Q evaluation. Fresh toy evaluation accepts only copied policy
  weights and registered task settings, reloads pinned visual prior, and creates
  fresh camera/perception/encoder histories and environment RNG. It does not copy
  the training trajectory or call learner updates. Diagnostic tests verify exact
  initialization, repeated same-seed results after unrelated ending scenes, and
  unchanged training/global RNG state, weights and traces.
- Immediate and eight-transition delayed visible-choice task implementation,
  first-action latching, ignored later commands, persistent target/consequence,
  zero intermediate rewards, and true .02-second cadence. Balanced evaluation
  schedules and identical schedules between lambda arms are tested.
- SQLite WAL + FULL durability, atomic sample/checkpoint updates, sequence/hash
  checks, one immutable frozen bank, periodic active snapshots (100 transitions or
  10 seconds, plus stops), fail-closed committed-tail replay, poisoned failed writers.
  Replay-v2 explicitly dispatches known Pong/toy session types and checks source,
  model/configuration and runtime identity. Toy code/settings are included in checks.
- Timing records separate active process prefixes from restart gaps. Crash gaps
  include unobserved execution after the last heartbeat; they are not pure downtime.

Pinned model: `src/fly_connectome/vision/data/default.pt`, 680428 bytes,
SHA256 `a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173`.
Load configuration: legacy prior upgraded by `load_default` to consensus-v3 64x64.
Learned vision remains frozen. No backprop was introduced.

## Validation and remaining failure

Latest full preflight: **790 passed, 4 skipped, 1 failed in 54.60 seconds**.
Focused evaluator/toy/durability: **21 passed in 8.20 seconds**.
After moving the unrun supervisor into the experimental directory, its manifest
allocation test was rerun: **1 passed in 0.93 seconds**. No competence run occurred.

The sole full-suite failure is the independently reproduced baseline
`test_native_neural_step_matches_reference[False]`, a voltage difference of at most
`1.1920928955078125e-07` at tick 1 under an exact-equality assertion. The assertion
was not relaxed. Two missing Linux -fPIC flags were fixed earlier. The baseline
rounding issue is distinct from reconstructed learner/evaluator correctness.

## Unvalidated work kept separately

`scripts/experimental/run_registered_toy_pilot.py` is explicitly UNVALIDATED as a
supervisor: its 36-job manifest and exact allocation are unit-tested, but the
worker/supervisor path has never run end to end. It was parked rather than launched.
Ryan should replace or review it before using it. It must not be treated as a
validated production scheduler. All other pending code edits were committed;
no uncommitted source changes are intended to remain.

## Registration and pending hypotheses

The authoritative proposal is `docs/reconstruction/2026-10-15-toy-preregistration.md`,
including the final review amendment. The previous authorization is now paused.

Budget if explicitly reauthorized: 2400 learning + 2160 evaluation = 4560 new
transitions across 36 jobs. Two tasks, lambda .8 and 0, three paired seeds. Fixed
alpha=.0001, gamma_per_second=.95, epsilon=.2; zero initial readout weights, fixed
features, no tuning/extensions/success-stopping. Learning seeds: immediate
2101–2103, delayed 2201–2203. Fresh evaluation: 3101–3103 / 3201–3203; random baseline
RNG: 4101–4103 / 4201–4203. At most 200 transitions and 180 seconds per chunk, saving
reserve and timeout backstop, aggregate 60 minutes including replay/startup/I/O.
Initial/reset camera observations must be counted separately from transitions.

This measures acquisition of TWO FIXED VISUAL TEMPLATES, not visual generalization.
Report left/right counts separately; repeated templates are not independent
visual-generalization trials. Screens are 16/20 immediate and 14/20 delayed per
seed. Immediate arms must match complete training trajectories AND final weights.
The lambda comparison is exploratory: persistent consequence features permit
lambda=0 bootstrapping, and hidden countdown prevents a full-Markov-value claim.
Pong learning remains unproven. Forecast availability is not evidence of benefit.
No rebuilt toy acquisition result exists yet; performance at the small fixed alpha
and budget is an open question, not a reason to silently tune.

## Runtime and exact safe commands

CPU environment: `/workspace/venvs/fly-m1b-cpu/bin/python`, Python 3.12.14,
NumPy 2.0.1, Torch 2.5.0+cpu, SciPy 1.18.1, PyArrow 26.0.0, pytest 9.1.1,
matplotlib 3.11.2. Debian 13.6 x86_64, 5 logical CPUs, about 17 GiB RAM.
The editable package points at `/workspace/Fly-Connectome-m1b`. Dependency freeze:
`docs/reconstruction/cloud-requirements.txt`. GPU capability is not established;
this environment and the proved paths use CPU only.

Inspection and tests (these do not start registered competence experiments):

```bash
cd /workspace/Fly-Connectome-m1b
git status --short --branch
git rev-parse HEAD
/workspace/venvs/fly-m1b-cpu/bin/python -m pytest -q tests/test_fresh_evaluation.py tests/test_toy_session.py tests/test_m1b_durable.py
/workspace/venvs/fly-m1b-cpu/bin/python -m pytest -q
```

The second command retains the known baseline failure. Recovery proofs are
controlled learning diagnostics; run only deliberately, with a NEW output path:

```bash
/workspace/venvs/fly-m1b-cpu/bin/python scripts/prove_toy_recovery.py --proof /workspace/m1b-verification/NEW-toy-proof
```

For an existing compatible durable run, the new orchestrator should open
`DurableRun(exact_db_path)` without supplying a fresh session. Inspect
`run.session.spec`, `run.session.transitions`, `run.replayed`, and actual committed
sample counts before advancing. `run.step()` commits one new transition;
`run.stop()` writes a stop checkpoint and closes the connection. After an error,
never reuse the advanced in-memory writer: reopen the database and replay its
committed tail. Never count replay as new learning; charge it to compute/time.
Use one writer per DB. Do not overwrite an existing run with a fresh session.
Do not bypass source/model/runtime mismatch errors or resume old schemas by guess.

There is no live pilot to resume today. Historical diagnostic databases may be
incompatible with the latest source/schema and are retained as evidence; return to
their exact source checkpoint for reproduction instead of bypassing validation.
Trusted same-platform pickle remains required; Windows or arbitrary runtime
portability is unproved. Local filesystem survival across replacement cloud tasks
is not guaranteed. There is no automatic background continuation or scheduler.
FULL durability was tested against process exits, not machine power loss.

## Historical claims versus current evidence; roles/model assumptions

The parent supplied historical M1B toy successes, Windows 96 vs 48+48 recovery,
and historical 925/928 test counts. Those commits/weights were not imported.
They guided requirements but are not evidence for this new reconstruction.
Current evidence is only the checked-in tests and named Linux diagnostics above.

Ryan selected Sol6.1 ExtraHigh and requested the existing two-thread arrangement
remain until he restructures orchestration. No model, reasoning or thread setting
was changed here. Actual executing model ID/reasoning is not independently exposed
by the tools, so the user selection is an assumption, not a verified runtime fact.
This cloud thread implemented code; the parent coordinated; Astra reviewed via
Library because its filesystem is separate. Ryan is now taking over orchestration.
