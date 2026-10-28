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

## Current provisional model and actual evidence

Parent selected the fixed pairwise candidate as PROVISIONAL working lineage:
`/workspace/m1b-experiments/pong-rally-pairwise-residual-20261028-r1/rbf-model.npz`
Bundle SHA `eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152`.
Despite legacy filename, this is centered degree-two TensorSketch512, frozen
vision/original4389input transport/scaler, frozen3ac basehead, learned3×512outputs.
No backpropagation, no feature recalibration. Only output weights may learn.

* TRAIN160:3ac409→pairwise412total,143→146first,266/384→266/380later.
  All known openings retained; two-returngames103→95, five-returngames30→35.
* Independent fresh32:73→81total,27→28first,46/69→53/75later. Later-rate
  +4pp CI[-11.85,+18.75]pp; mean+.25returns/game CI[-.44,.97]. Uncertain.
  Two-returngames21→18, five-returngames4→8. Reliable match-long goal NOT met.
* Preserve fallback3ac at `pong-rally-earned-trace-merge-20261028-r1/proposal.npy`,
  SHA `3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e`.
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
work unknown. No promotion yet; current eb/r5 retained until parent review.

Completed diverse3-target batch `pong-rally-joint-batch-20261028-r1`:
agentdf9c09e/main3a5600a,27focused tests. One33.26sverifiedfit/80,839rows,
actualtargets1/1/2→3/6/6; full192523→546total,175→176first,123→129second,
later348/482→370/501. All192retained;11gains/no losses. But newfresh32:
eb72/28first/18second/44of68later;40069 86/30/23/56of80;candidate73/28/21/45of71.
No demonstrated fresh improvement. Batch4c0b preserved as TRAIN artifact, not
promoted. Parent/worker selected broader new TRAIN experience next. Implement minimal
manifest collector:64newstarts32/phase, fixed40069and4c,128games/600s,
no fitting yet; code review before launch. Existingfresh remains evaluation-only.
Plan: [broader own-reward experience](../plans/2026-10-28-broader-earned-experience.md).
After collection, review diverse better prefixes and a bounded joint-learning block.
Exact completeprefix certificates can preserve known-return LOWERBOUNDS between
fits without fullsuffixreplay; mixedsourceprovenance required, currenttargetactual
outcomes must be measured before replacement, full/freshqualification atblockend.
This block is not implemented or launched yet. No more blind small oldreplayfits.
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
Bridge remains another thread's work. Main through a73015e pushed.
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
