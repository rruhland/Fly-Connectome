# M1B research handoff — 2026-10-29

## Current position — learned feedback qualified; reliability still incomplete

The milestone is **not met**, but the compact feedback learner has now produced
a substantial fresh-game improvement. Preserve `40069` as the historical baseline;
carry the new relational endpoint for further scaffold development.
The user has authorized continuing without further architecture approval until
reliable learned match-long returns are achieved or they stop work. Preserve the
no-backpropagation, camera-input and no-expert-label constraints.

The latest symbolic archive candidate improved TRAIN returns from 83 to 86, but
failed the frozen fresh comparison: **74 to 68 returns**, with first returns
unchanged at 26 and later returns falling from 48/69 to 42/64. No promotion.
Stop the symbolic grammar family; see the symbolic archive and fresh outcome notes.

The compact relational feedback search completed in
`/workspace/m1b-experiments/pong-relational-search-20261029-r1` under spec
`aa9b357ca0c2f354f1f078bcf41d25614024bd7cc25942d257dcd9da73c557c7`.
It learns anonymous object-pair binding, axis/polarity and five feedback
coefficients from actual own-hit rewards. Existing vision and the baseline opening
are retained. This is a stronger controller prior with a smaller search space,
not a new optimization algorithm. Fixed nested 4/8/16-game training cohorts avoid
the previous rotating-fitness problem; only complete 16-game fitness can replace
the incumbent. Total allocation: 900 seconds and 320,000 transition cameras,
including a final 32-game comparison reserve. Actual cost was 632.38 seconds,
231,307 transition cameras and 448 initial observations.

The selected controller learned choices `(3, 0, 1, 0, -1)`; all five numeric
coefficients remained zero. Its learned mechanism is position feedback between a
reward-selected anonymous visual pair. TRAIN32 improved **83 to 130 returns**,
with later returns **54/76 to 101/110**. Frozen fresh cohort 63313 improved
**90 to 131 returns**, with first returns unchanged at 30/32 and later returns
**60/84 to 101/113 (71.4% to 89.4%)**. Games reaching five returns rose from 8 to
18; the opponent missed no returns. All candidate decisions replayed exactly.
Sources/assets stayed frozen; the 32 fresh starts were independently checked
against all 86,815 declared exclusions. This is real improvement, not yet the
95% reliability target or independently replicated learning.

Endpoint `models/final-policy.npz` has SHA
`026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
Next: compare the same learned controller from the first supported observation
against its current baseline-until-first-hit gate. Separately, a new persistent-ID
adapter will test using the existing vision system's carried entities during
brief observation gaps. No additional fits or coefficient sweep are selected.
Both changes require new runtime identities and fresh qualification; 63313 must
not be reused as an untouched evaluation cohort.

The new fitness runner uses an exact encoder-copy optimization. A paired native
proof reproduced every action, feature, encoded observation and runtime-state hash.
One timing pair suggests modest savings, not a general speedup percentage. Original
pinned runners are untouched. Helper commit `3e9081d`; search commits `7238313`
and `e144648`. The old unlaunched spec is preserved in the run directory.

Reused Sol 6.1 XHigh agents: `value_execution` owns execution and opening-gate comparison;
`learning_challenge` independently reviews the mechanism and next direction;
`runtime_efficiency` completed the exact-copy proof; `symbolic_control_challenge`
completed the negative fresh result. Parent owns synthesis, review and direction.
Older entries below are historical and must not override this current position.

## Active mandate

Continue orchestration until ball returns are learned consistently throughout a
match, or the user stops work. The goal is NOT met. No backpropagation, privileged
policy inputs or expert action labels. Latest user instruction APPROVES the tree
pilot and temporarily waives further architecture approval until this milestone
is complete or the user stops work. Parent review/bounded experiments continue. Bridge/viewer connectivity is
another thread's work. Earlier two-hour session limits are superseded.

Use parent direction/review and reused Sol6.1XHigh execution/independent challenge.
Previous workers unavailable at this turn (agent list only root); replacement
Sol6.1XHigh `/root/value_execution` owns approved tree implementation in MAIN.
Independent `/root/alternative_mechanisms` researches distinct next hypotheses.
Prior worker: `/root/rally_integration`, checkout
`/workspace/Fly-Connectome-m1b-pong-fast`. Reuse it via followup_task when idle.
Old `/root/pong_one_pass_research` became unresponsive and was parked.
Main checkout `/workspace/Fly-Connectome-m1b`, branch
`reconstruction/m1b-continuation`; Python `/workspace/Fly-Connectome/.venv/bin/python`.
Use `PYTHONPATH=src:scripts/experimental`, one BLAS/OpenMP thread and four native workers.

## Latest active update — October29

User has waived further architecture approval through this milestone; do not ask
again. No backpropagation, privileged inputs or expert action labels. Goal remains
95% first/later returns on fresh games with200+ later opportunities and independent
training replication. Current40069baseline remains primary;761secondary is unpromoted.

Completed pilots: fixed-treeFQI23→1; local option memory found2actual rescues but
23→19 deployed; past-reward context correction23→23. Original Ogma temporal/local
actor23→10, rejected. Its first changes all occurred on empty selected current
visual planes, while the baseline still had other camera context.

Guarded SAME Ogma weights also REJECTED on32games:82→74total,29→24first,
18→15second,53/76→50/67later. Source19c8bdf; recovery85f2b5b.
Screen8+1did not transfer. All16,441completed candidate rows replay exact;
64initial empty-prefix rows matchbaseline. Original180s interruption left17complete
receipts+166-row orphan. Separate120s recovery completed15missing in46.586s,
6167cameras+15initials. Saved work lowerbound16,607cameras+33initials PLUSunknown
interruptedwork. No fresh63185, no more temporalinput/Tvariants. Artifacts under
`pong-ogma-prior-20261028-r1/empty-current-correction/recovery-20261029-r1`.

Pairedforest9f282c6 REJECTED: all48sites acquired86.056s,28,139cameras+48initials;
16rescuepairs/12sites,19losthitpairs/15sites,109ties. Singlefit0.269s/exact2096row
sklearnparity. Actual8games23→9total,7firstunchanged,4→1second,16/21→2/9later;
114optionstarts/1656heldcams/1319changedactions.2590evalrows exact. Combined
30,729newcams+56initials; no32/fresh63201, no parametersweep. Source/forestfrozen.
Artifact `pong-paired-option-forest-20261029-r1`. Root requested<=3minoffline
mechanismreview fromvalue_execution, no newnative. Preserve compactN+1suffixdata.

Forestmechanismreview:18/21all-zero training sites stillpositive;37selectedholds
include12actualrescues+25ties.107/114deployedstartsrepeatwithin160cams(median16).
Root acceptedSol'schallenge: no fixedweights tailcorrection, stopforestfamily.

Active graph: `/root/learning_challenge` firstactualofflinefit converged13.56s,
1012iterations, residual9.966e-9/errorbound5.526e-6.27471eligibleobservations,
118hits,52naturalends;8censoredfinalsomitted.65capacityoverflows unsupported,
nohit/end evidence lost.20,080/22,233posthitrows baseline+alternative supported;
674/27301unknownsuccessorboundaries;13,343savedchoiceschanged (notperformance).
Graph4067f033ec767fa072ab4e9342d0d95b9f27b030cbcda536e16261fd7bdccb75 at
`pong-kernel-transition-graph-20261029-r1-recovery/graph.npz`.
Original60ssetupfailedbeforeanylearning(importpath,overflow); explicitseparate60s
recovery preservedoldreceipts/specs. Sevenfocusedtests+64queryserializationchecks.

Frozen graphnative69ea13a REJECTED23→10total,7firstsame,4→2second,16→3later.
2717cameras+8initials,10.913s;allchronologicalrows exact,1020changedactions,
1611/2020posthitrows supported. No32/fresh63217/radius/supportvariants.
Artifact `pong-kernel-graph-native-20261029-r1/summary.json`.

AdaptiveSMDPsource3d156b9 complete: physicalelapsedgroupinginvariant reward,
MCobserved-returnwarmstart+26FQIupdates168.313s, finalca99b349... selectedbyclock.
Notconverged; exact44,699training+2048serializedqueryparity. Heldmacros stopon
actualhit/naturalend/16; baselineoption1.17/32actualoptionleafpresenceguard,
independencediagnosticonly. Allsettingsfixedbeforefit; seeplan/resultsledger.

Initialnative201f517genericnewharness:23→12returns,7firstsame,4→2second,16→5later,
3150cams+8initials/18.591s. Allrowsauditexact. Onegain1→2andone5hitretained,
threebiglosses. No32/fresh63233/promotion. Artifact `pong-smdp-native-20261029-r1`.

Closed-loop C1 improvedMATCHED32 C0→C1:44→50total,29firstsame,7→13second,
15/42→21/49later. Baseline40069still82total/53later/18second. No63249fresh or
promotion; thisisTRAINadaptationonly. Comparison77.661s,21664newcams+56initials,
C0eightcachedreused; allaudits/frozenchecks passed. Artifact
`pong-smdp-cycle-comparison-20261029-r1/summary.json`.
Collectionall64complete71.734s,22876cams+64initials,86hits; exactmacroexploration.
C1refitdata65409usable/139starts;MC+16FQI166.942s(same180budget,C0had26updates).
Finale58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd;C0unchanged.
Exact65409+2048parity; notconverged. Olddataprefixes byteidentical. Seeledger/plan.

Second feedback cycle is complete and rejected; switching to learned dynamics/planning.
Collector2 used frozen C1 with unchanged exploration/options on 64 new TRAIN starts
(32 per phase), excluding all 142 prior/admitted starts. All games completed in
62.941s: 19,998 transition cameras +64 initials, 73 hits. These collection hit totals
are not paired performance because collection1 and collection2 use different starts.
All 56 new comparison traces (21,664 already-paid cameras) plus all 64 collector2
traces were appended; cached C0 eight were not duplicated. Prior array prefixes
and start IDs are byte-identical. Combined 100,286 usable windows across 206 starts.

C2 fitted in 168.054s, MC plus 12 completed FQI updates; endpoint
`785d35d48814633f253073f2e6f7804cb98b4293e99dadc632375ba2f4c7b43d`.
Exact query optimization preserves targets/support/scores; all 100,286 training
inputs and 2,048 serialized queries passed parity. More data still costs more:
C1 completed 16 updates, C2 12 in the same 180s limit. Neither is converged.
No settings, reward, exploration or action-duration changes. Sources c31de46.
Artifact: `pong-smdp-feedback2-refit-20261029-r1`; data spec d590d036b673bb099efe9c309cec799aceaa81ce97274d9109177718ae203a56.

Comparison2 (source a86d89c) used cached C1 games; only C2 played. Spec
92ae13da91aaf38de4a6d58e20ec7a85212492ee4f678ed99a93c7560c4f1481.
C2 failed the first-eight gate: C1 16→13 total, first7 unchanged, second5→2,
later9/15→6/13. Baseline still23total/16later. All3,270 rows audited exactly;
3,270 new transition cameras +8 initials in16.508s. Source/assets frozen.
No remaining24 or fresh63265 games; no promotion. Stop data-only SMDP cycles.

Learned-dynamics pilot completed and was rejected. Source18f09cb: ridge visual
changes +8 residual trees,16 event trees, H8/W8 primitive replanning, previous16
actual commands, frozenC1option3 tail. Prepared114,229 unique transitions from452
traces/206starts in25.04s; retained visibility/cardinality changes and actualprefixes.
One fit39.131s,1.47MB new model; model
171b9acabd3c4abc4b446b9f8c83a373bd36816aedfc8f96cebda80a075a189d.
Compiled exports2048rows/head and32 serialized beam plans exact; mean9.145ms/query.
Actual8 baseline23→11total, first7same, later16/21→4/11, second4same, max6→2.
2958transitioncameras+8initials in21.307s. AllrawQ/historyrows checked;402planner
queries replayed includingonecompletegame. Sources/assets frozen. No32/fresh63281.
Artifact `pong-delta-native-20261029-r1/summary.json`. Stop model/planner variants.

Symbolic search completed at literal zero: 749.693s,169402transitioncameras+
400initials,399completegames+1partial;94completechallengerfour-game evaluations,
11fullgenerations+partial12th. OwnTRAIN32bank baseline83total/29first/54of76later;
NOT the earlier82-returnbank. Localwinners failednewrotations andzerowasrestored.
Literalzero endpoint reusedexactcachedbaseline, noadditionalqualificationgames.
No fresh63297/no promotion. Spec fdc6b3b81e6398dca97f0c861c68fc2794fc5658c5e86803a16ff45a636ea949.
Resultnote `2026-10-29-symbolic-policy-outcome.md`, commitbac94e5.

CURRENT parallel tasks:
1. Posthoc frozenarchivequalification: root found exactlytwo previousincumbents
withnet+3hits across8actualmatchedfitnessseeds each, so lastrotationzero isnotproof
allarchivedpoliciesbad. a43dc09a175c...18vs15; f85f6d53fcda...16vs13.
`/root/symbolic_control_challenge` ownsapprovednewwrapper/source/spec
168d489cc9ff5a2e56f94fb5c9ea58ffedcc4a56e33464c62ebfc112da195acc,
artifact `pong-symbolic-archive-comparison-20261029-r1`. 300s/64ktransitioncap,
64newfullrawgamesinterleaved2policies/seed, sameown32bank, nofit/parameterchanges.
Budget300correctedBEFOREnativefrom180 usingmeasured226cams/s toavoidcensoring
best64×1000gameoutcomes. No fresh; parentallocatesifonequalifies. Rootreviewdone,
launchauthorized. This isseparateposthocselection, notchangedoriginalsearch.
2. Compactrelationalfeedbackservo implementation: `/root/learning_challenge` core,
`/root/value_execution` search. Five discretechoices n/i/j/axis/sign; five numeric
ci,cj,b,deadzone,hysteresis. Anonymousrelativeposition+velocityfeedback, rolesonly
selectedthroughactualreturns. Explicitgenericservo prior, notdiscoveredgeneral
control ornewoptimizer. Usefixednested4→8→16TRAINfitness withsameaccumulatedseeds;
stoprotating4singlechampion objective. Implementationonly, nolearningauthorizedyet.
3. `/root/runtime_efficiency` ONEactualzero-baselineprofilegame authorized≤1000
transitions+1initial at `pong-runtime-profile-20261029-r1`. Profilebeforeproposing
EXACTspeedups; noexistingpinnedsource edits. CurrentCPUquota4cores/16GBRAM.

One0.58s savedtraceworldplannerreview found stable-maskone-stepmotionreasonably
predicted, but8-step tailerror exceededchosenrootadvantages. Sol independently
advised NO-GO event-onlyH8variant: finaleventpredictiondoesnotproveearlyuseful
actionranking. Rootacceptedstop; preservelearnedmodelsforfuturefeatures/proposals,
notmoregreedyplannerparameter variants. Notesinlearned-delta-pilot.md(b2fcc33).
Userwaiverpersists; noarchitectureapprovalquestions. Continueuntilmilestoneorstop.
Bridge belongs toanotherthread.

Disk headroom: removed147MB completed unit-test temporaries, then hardlinked125
SHA-identical frozen transform copies, recovering1.267GB. All research paths/bytes,
checkpoints and receipts preserved. Manifest `storage-deduplication-20261029.json`
in experimentroot. Copy shared transforms before future in-place modifications.
Do not duplicate large metrics/checkpoints. Bridge belongs to another thread.

## Latest result / next action

Cloud connection recovered. Final broad-block fresh32 rejected bfb: 90→69
returns versus40069, first30 unchanged, later60/82→39/66, zero gains/eight losses.
Only six missing evaluation games were recovered in a separate13.575s extension;
original deadline expired, saved58 unchanged, unknown interrupted work preserved.
No promotion. TRAIN256's662→686 gain did not transfer. Final audit passed35,048fresh action/Q rows;14games changed actions,8lost
returns, with6losses first diverging after an earned hit. No identical hard-prefix
replay block. Reused rally_integration is implementing one bounded causal-burst
plus joint-output pass; see [plan](../plans/2026-10-28-causal-joint-return-learning.md).
Causal pass completed80.737s with9269cameras+11initials, no restartneeded.
One rewardedaction learned in5.33s; target2→3, but representative32total60→57,
first24same,second16same,later36/56→33/55. Candidate85b35d rejected; nofull/fresh.
TAG63057unused/reserved. Sourcec9295e1/main789648b;35focusedchecks passed.
User mandate continues. Full-coordinate actor completed786.97s,190265cameras+
384initials. FULL256662→566,first233→214,second161→144,later429/615→352/535.
Reject4a30; nofresh63073. Training-start greedy returns also151→141. Keep40069.

Smooth198-coordinate actor completed435.28s,114966cameras+250initials.
Screen128287→237returns,first106→91,second67→61,later181/262→146/222.
Both64strata lost25returns; rejectdc33740, keep40069. No fresh63089; no process
active. AllchecksPASS, no unknownwork. Actorfamily stopped aspredeclared.

Continuous residual pilot completed with a weak mixed signal, NOT a reliable
learner.14completed updates; calibration recovery kept the original clock and
reused all2048saved observations without replay. TRAIN32 total63→64,first24→22,
later39/59→42/59. Parent approved SAME frozenendpoint on32predeclared freshpairs
within original600s: total78→81,first30→28,second18→20,later48/72→53/76.
Paired mean-return difference .09375,95%bootstrap[-.625,.78125]. Preserve76126332
as a secondary research candidate, keep40069baseline; no automatic promotion.
Artifacts `pong-rally-continuous-actor-20261028-r1`; source5626846/df65c94/f5abc5d.
Parent firstbatchaudit2039rows exact Q/probability/eligibility/credit,0cameras.

Fresh Sol6.1XHigh `/root/fresh_learning_directions` independently researched
value learning, learned dynamics/planning, and temporal predictive hierarchies.
LSPI existing-runtime test completed4.613s then failed actual8games23→2returns.
Reject7195fd; no sweep. Artifact `pong-value-feasibility-20261028-r1`.
Root authorized new existing-policy epsilon.2 acquisition to address action
coverage BEFORE another valuelearner:32fixedTRAINstarts,16/phase,<=20kcameras,
120s ceiling; completed39.099s/11871cameras+32initials,49hits,23natural ends
and9quota partials. Complete N+1observations/rewards/timings, no model changes. Artifact
`pong-value-collection-20261028-r1`, spec
127f1c6c24fa5278938f5fe8b51d6eadbe8bf00ee4a9b1471987279b17e27c4b.

Next major proposal: [fixed-partition value learning](../plans/2026-10-28-fixed-partition-value-review.md).
User approved; implementation underway. User explicitly wants distinct parallel hypotheses,
outside evidence, and advancing/combining promising methods instead of exhausting
feature/actor variants. Further architecture approval temporarily waived; ordinary acquisition
and research remain authorized. No backprop, privileged inputs or expertlabels.
Goal remains unmet. Bridge belongs another thread.


## Current provisional model and actual evidence

Current practical research baseline is joint40069, not a fresh-qualified claim
of better later-return skill:
`/workspace/m1b-experiments/pong-rally-joint-output-20261028-r1/rbf-model.npz`
Bundle SHA `40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438`;
W SHA `479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f`.
Frozen vision/map/transport, existing raw4389 plus fixed512pairwise outputs; both
output weight arrays learned without backprop. Feature assets unchanged.

* TRAIN192:493→523returns,174→175first,113→123games≥2,319/455→348/482later.
* Pooled existing64fresh (descriptive after interim decision): eb145→joint151
  returns,57→60first,39→42games≥2,later88/136→91/142. No convincing later-rate
  improvement. First/second coverage motivates a practical baseline, not success.
* Batch4c checkpoint retained as own-reward repertoire: TRAIN546, but fresh73
  versus joint86 on latest32, so not selected as the general-player replacement.
* Preserve eb at `pong-rally-pairwise-residual-20261028-r1/rbf-model.npz`, SHA
  `eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152`.
* Preserve 3ac at `pong-rally-earned-trace-merge-20261028-r1/proposal.npy`, SHA
  `3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e`.
* Preserve d7 at `pong-rally-guarded-curriculum-20261028-r2/selected.npy`, SHA
  `d7d7d872c1acde4b34748240c2ce9bf9d37af4c65ff2cd1f25d753f19d01fe61`.

All relative artifact paths are under `/workspace/m1b-experiments`. Current model
is a reset/branch-assisted own-hit-reward scaffold, not the final continuously
learning general agent. The original viewer export is older than these models.

## Active — joint existing-output learning

Root `pong-rally-joint-output-20261028-r1`, main7e53729. Learn both EXISTING
linear and fixed512 output weights, with frozen vision/features, no backprop.
Plan: [joint output learning](../plans/2026-10-28-joint-output-learning.md).
One60s fit verified74,623complete-earned-prefix rows in18.53s. Sourcefc's rewarded
201-row two-hit target4195433914 replaces its old one-hit prefix; all other earned
prefixes retained. Actual target1→4; representative32 87→93; full192493→523,
174→175first,113→123games≥2,later319/455→348/482. No perseed return losses.
Both output paths contribute; no necessity ablation performed.
New W SHA479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f;
bundle SHA40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438.
Fresh32 TAG62993 interrupted at63games/30,923cameras. Parent recovered only the
missing carry game within ORIGINAL900s:5hits/1,000cameras/12.62s. All64receipts
complete:73→65returns,29→30first,21→19games≥2,later44/68→35/62.
No demonstrated transfer gain. All saved action/Q rows audited; all processes
exited. Native completion865.718s within original900s (fresh300substage exceeded
during interruption); documented≥105,587cameras+191initials, unsaved interrupted
work unknown. Joint40069 now practical research baseline as described above; oldeb/r5 preserved.

Completed diverse3-target batch `pong-rally-joint-batch-20261028-r1`:
agentdf9c09e/main3a5600a,27focused tests. One33.26sverifiedfit/80,839rows,
actualtargets1/1/2→3/6/6; full192523→546total,175→176first,123→129second,
later348/482→370/501. All192retained;11gains/no losses. But newfresh32:
eb72/28first/18second/44of68later;40069 86/30/23/56of80;candidate73/28/21/45of71.
No demonstrated fresh improvement. Batch4c0b preserved as TRAIN artifact, not
promoted. Parent/worker selected broader new TRAIN experience next. Implement minimal
manifest collector:64newstarts32/phase, fixed40069and4c,128games/600s,
no fitting yet. Sourcea3c2aee/main41d1ce6 reviewed;29focusedtests passed.
Collector launch authorized at `pong-rally-new-TRAIN-repertoire-20261028-r1`,
manifest at sibling `.manifest.json`, ROOT1943906127/TAG63025. Collector completed209.534s/600,63,910cameras+128initials,128complete,
all saved action/Q/source/raw checks PASS. New64:40069 139/58first/38second,
4c140/58/38;14better4c,16worse, same6firstmisses. Existingfresh remains out.
Plan: [broader own-reward experience](../plans/2026-10-28-broader-earned-experience.md).
Parent approved admission of64auditedscaffold40069receipts into uniformTRAIN256:
662returns/233first/161second/429of615later;100,069earnedrows/233cases.
Implement bounded3-updateblock,≤3targets/update,90sfit each,one1200sclock,
broaderGsameformulaonce.12compatiblepool; first1289501189L1→2,58405525R1→2,
3654232341L2→3;100,586initialhardrows. Sourcebeb99fc/maina8a0a84 reviewed,32focusedtests PASS; launched
`pong-rally-joint-learning-block-20261028-r1`, session98406. Single1200sclock,
implementation900sseparate; spec5759fd6f36eb4c536a6fcae7c9c4a09ec41433aca8130ada76e37fb83f32cfd3.
FreshTAG63041 reservedpre-fit, uniformnewbankr7. No activecodechanges.
Exact completeprefix certificates can preserve known-return LOWERBOUNDS between
fits without fullsuffixreplay; mixedsourceprovenance required, currenttargetactual
outcomes must be measured before replacement, full/freshqualification atblockend.
Use fixed8correctnesssentinel incl104fivehit, finalfull256/fresh32vs40069once.
Stopupdatesonfailedfit; preserve lastverifiedcandidate. No pruning/overrides now.
All3updates accepted by218.70s: fits20.62/27.59/28.55s verified100586/101837/103318rows.
Actualtargetgroups4→10,4→12,7→10; controls28→28/28/29. MemoryLOWERBOUNDS
662→668→676→680,233first/165second/447later; notfullnativecounts.
Finalmodelbfb0d2824c7eb943434f31ff5772bb1ceee48add957220bd0374f6c83e84d1f7,
Wf90abda40c83aa4a9fa9df25efdbf04db66df6f13c03519cb857f828def048a6.
Full256complete:662→686returns,233first,161→167second,429/615→453/636later;
13gains/0losses, all earned minima retained. Cloud outage left58/64freshgames
saved and no activeprocess. Recovery authorized ONLY missing frozen games
under separate180s allocation if originaldeadline expired; do not resetclock
or restarttraining. Reused rally_integration recovering; no activecodechanges.
No more blind small oldreplayfits.
No fresh62993or63009admission. Both r5eb and r6TRAIN-scaffold banks preserved.
Batch589.36s/900,160,146cameras+271initials, no interruption/unknown work.
Bundle4c0b6374f3e35fad8260d5bd07161f780383de99a8c536b67437169d12200811;
W3f2e4958de960c32d0ba30b61d25449e1d60d1344fe9c5bf669b12d08f1419aa.
All runs exited. Plan/results: [joint batch learning](../plans/2026-10-28-joint-batch-learning.md).
Goal remains reliable match-long returns, not merely passing a fitted constraint.

Current TRAIN bank `pong-rally-TRAIN-bank-20261028-r5/admission.json` has192games:
493returns/174first/113games≥2/319later of455. Earlier32fresh candidate receipts
were explicitly admitted after evaluation; subsequent fresh sets remain evaluation
only. Preserve old banks/checkpoints and reserved final streams.

Recent stopped routes (details in ledger):
* Residual-only historical replay improved TRAIN493→499 but fresh80→78 and
  forgot original104562292five-hit behavior5→2; unpromoted ef890acc.
* Eight projected random requests all failed bounded projection verification;
  zero native games, not evidence that their actual rewards failed.
* The SAME eight raw requests yielded two strong8game screens, but chosenfc
  failed full192493→404. Existing runnerup1+ failed32games92→59. Search family
  stopped; useful individual rewarded fc trajectories were retained as experience.
* Joint learning combines existing output paths to integrate that experience
  while retaining complete earned memory; no new architecture was introduced.

Detailed results: [ledger](2026-10-27-sustained-return-results.md).
Bridge remains another thread's work. Main through2f9ddc6 pushed; a8a0a84 and currentresults pending nextpush.
Disk has about1.5GB free; reuse the192MB metric.

## Completed block and failed routes

Four new burst recoveries were found on current-policy misses, each+1actual next
return (`pong-rally-new-bursts-20261028-r1`). Integration remains the difficulty.
Hard complete-sequence preservation did not verify within bounded fits. Full
empirical score covariance made fitting fast but harmed actual retained behavior:
target-only sentinel61→58; three selected guards improved sentinel61→68 yet
full409→391. Protecting every known successful opening gave full409→403,
143→144first but266→259later. All rejected. Do not resume blind guard growth.

User approved ONE fixed64-RBF residual comparison, now completed:
`pong-rally-local-rbf-20261028-r1`.2,048TRAIN samples/160episodes; exact zero-output
parity on92,162saved rows. Effective centered rank1.208, broad activations. An initial
heuristic screen stopped after4.81s; parent and Sol explicitly relaxed only that
heuristic and ran one fit on the exact saved basis/data/objective under the ORIGINAL
600s deadline. Fit6.18s: no verified candidate, solver-reported primal infeasibility
(approximate certificate, not exact impossibility proof). ZERO native cameras,
no new model, no bandwidth retry. No further RBF sweep is authorized.
Source agent7bb1847/191bde6 integrated main1baf03a/2ad26e7; eleven focused tests pass.
Implementation416.4s recorded separately. Outcome docs pushed in main12a18b1.

One passive vision diagnostic suggested short-history velocity estimates reduce
noise in a selected miss; another miss had accurate inferred ball tracking.
Neither establishes global vision correctness or improved control. Smoothing can
lag rebounds and change the head's input distribution. These motivate the active
bounded native pilot, not a declaration that vision is solved.

## Durable references

* [General scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md)
* [Sustained-return plan](../plans/2026-10-27-sustained-return-learning.md)
* [Detailed results ledger](2026-10-27-sustained-return-results.md)
* [Completed RBF proposal and amendment](../plans/2026-10-28-local-nonlinear-residual-review.md)
* [Preserved full historical handoff](archive/HANDOFF-through-2026-10-28.md)

Report actual native outcomes separately from rewarded exploration, fitted actions
and selected checkpoints. Preserve prior checkpoints and charge all camera work.
Use exact deployed-policy episode certificates where valid; avoid re-running
unchanged games. Reserved final streams remain reserved. Do not declare M1B
complete merely because a Pong scaffold improves.
