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

## Current selected model and actual evidence

Retained provisional head:
`/workspace/m1b-experiments/pong-rally-earned-trace-merge-20261028-r1/proposal.npy`
SHA256 `3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e`.
Current bank: `pong-rally-TRAIN-bank-20261028-r4/admission.json` under the same
experiment root.160existing TRAIN seeds; no recent fresh admission.

* TRAIN160:409total returns,143first,266/384later (69.27%),26timeouts.
* Paired new32 vs preserved d7:50→63total,27→27first,23/49→36/61later
  (46.94→59.02%). Confidence intervals include no improvement. Promising and
  uncertain; far from reliable match-long returns.
* Preserved d7 checkpoint:
  `pong-rally-guarded-curriculum-20261028-r2/selected.npy`, SHA
  `d7d7d872c1acde4b34748240c2ce9bf9d37af4c65ff2cd1f25d753f19d01fe61`.
* Current scaffold uses reset/branch-assisted own-hit rewards; this is not yet a
  demonstrated continuously learning general online RL agent.

Frozen vision, existing scaler/cap and3×4389 linear greedy K1 head remain deployed.
Original viewer/interception export is older and is NOT the current research head.

## Current — review next architecture after two rejected follow-ups

Fixed4-observation LS velocity pilot rejected:16TRAIN66→52total,16→16first,
50/58→36/46later.10,819cameras+16initials,44.37s. No filter sweep/learning.

Burst-credit test learned target4→5 with94ordinary continuation choices free.
Sentinel61→64, but full160409→394total,143→144first,266/384→250/368later.
Rejected; no fresh/promotion.54,235cameras+75initials,223.82s.95full games reused.
Original3ac7f87f/default transport remain selected. All runs exited.

Parent and Sol recommend one fixed centered degree-two TensorSketch512 residual
comparison. Concrete proposal: [pairwise residual review](../plans/2026-10-28-pairwise-residual-review.md).
Independent review completed; proposal includes asset hashing, equal-episode
calibration, unchanged base Q/cap and one absolute deadline.
No implementation authorized; user review is required. This is not covered by
the completed64-RBF approval. Current-turn total65,054actual transition
cameras+91initials across velocity/credit tests; RBF test used zero cameras.

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
