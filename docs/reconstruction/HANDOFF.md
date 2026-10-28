# M1B research handoff — 2026-10-28

## Active mandate

Continue orchestration until ball returns are learned consistently throughout a
match, or the user stops work. The goal is NOT met. No backpropagation, privileged
policy inputs or expert action labels. Major architecture changes require user
review; smaller/medium experiments are authorized. Bridge/viewer connectivity is
another thread's work. Earlier two-hour session limits are superseded.

Use parent direction/review and reused Sol6.1XHigh execution/independent challenge.
Active worker: `/root/rally_integration`, checkout
`/workspace/Fly-Connectome-m1b-pong-fast`. Reuse it via followup_task when idle.
Old `/root/pong_one_pass_research` became unresponsive and was parked.
Main checkout `/workspace/Fly-Connectome-m1b`, branch
`reconstruction/m1b-continuation`; Python `/workspace/Fly-Connectome/.venv/bin/python`.
Use `PYTHONPATH=src:scripts/experimental`, one BLAS/OpenMP thread and four native workers.

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

Next: prepare concrete generic continuous entity/pair-feature residual review;
major input/architecture change requires USER APPROVAL before implementation.
Reused rally_integration drafting proposal ONLY; parent will review and push it.
No more actor/temperature/rank/guard sweeps authorized. Usergoal stillNOTmet.


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
