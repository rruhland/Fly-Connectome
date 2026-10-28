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

## Active — next target is reliable second returns

Approved architecture test completed556.11s within original600s,92,106cameras+
156initials,204simplementation separately. Mixedsentinel61→62total butlater-2
was explicitly advanced by parent/Sol to full160 under SAME originalclock; no refit.
Separate frozen fresh qualification completed123.43s,33,287cameras+64initials.
All model/action/Q/provenance checks passed; all those processes exited.

Parent now continues ordinary learning within the SAME approved512map/basehead.
No seed/dimension/basis/architecture sweep. Reused Sol is reviewing a pass focused
on current second-return misses, protecting existing first-two-return behavior
rather than allowing longer-rally gains to hide early losses. Parent/Sol agreed on a history-first pass: existing rewarded3ac second-return
prefixes, cheap conflict screening, one delta-output fit, current first-two-prefix
retention. One900second pass; no discovery/parameter/guard sweep.
Plan: [second-return replay](../plans/2026-10-28-second-return-replay.md).
Implementation is active; no new fit/native execution yet. Selected historical
recovery161900621 has214rows through its second hit and no exact conflict with
36,560current guards; shorter than other compatible18599159(303rows).
221148912 conflicts;138732597 has no historical second success. Generic nonlinear carry/source-Q support is required;
do not treat pairwise source Q values as the frozen linear base's values.

Parent explicitly authorized admitting only32CANDIDATE fresh receipts intoTRAIN
after evaluation; authorization is under `pong-rally-pairwise-frozen-fresh-20261028-r1/parent-admission-authorization.json`.
Build new192episode bank from pairwisefull160+candidatefresh32:493returns,
174first,319later/455opportunities. No feature recalibration; new future evaluation
seeds required. The oldr4bank and old checkpoints remain immutable.

Detailed results in [ledger](2026-10-27-sustained-return-results.md) and
[pairwise proposal/results](../plans/2026-10-28-pairwise-residual-review.md).
Bridge remains another thread's work.

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
