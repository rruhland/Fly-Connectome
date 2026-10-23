# Resumed M1B orchestration: confirmation did not produce a stronger learner

No new learning gain was established. Both bounded experiments completed six search-and-confirmation cycles, accepted zero updates and retained RIGHTtheta6 byte-exactly. The registered unchanged-policy rule skipped heldout evaluation. This is a failed learning procedure under the allocated budgets, not heldout evidence about every rejected policy and not proof that the existing representation cannot combine the skills. M1B remains incomplete.

![Two confirmation experiments and their outcomes](/workspace/m1b-experiments/pong-mechanism-review-20261022/continuation-20261023-results.png)

## What ran

| Experiment | Proposed change | Complete / accepted cycles | Fresh confirmation LEFT/RIGHT net hits, each out of24 | New learner |
|---|---|---|---|---|
| Weighted search proposal + fresh confirmation | Existing mean-balanced ES update, function RMS .07945–.10 | 6 / 0 | (+2,-1), (0,0), (+2,-9), (-1,+2), (+2,-2), (+1,-2) | None |
| Best executed child + fresh confirmation | Exact best discovery child, function RMS .25 | 6 / 0 | (+1,0), (-1,-5), (+3,-2), (+1,0), (-4,0), (+2,-5) | None |

LEFT means a direct incoming opening; RIGHT means an opening initially traveling toward the opponent before returning. These labels group the explicit training curriculum and never enter the policy. The goal remains reliable returning, not winning on the first hit.

Each cycle used eight mirrored directions, sixteen candidate policies sharing six starts per phase, then incumbent/proposal comparisons on24new starts per phase. Accept only if neither phase loses observed hits and total gain is at least two. Both experiment families failed even the total-gain requirement: removing phase protection alone would still accept none. We did not lower the threshold afterward.

The executed-child follow-up tests a policy actually played during discovery, but also increases deployment displacement from at most.10 to.25. It therefore changes selection and step size together; it is not a clean causal isolation of proposal-to-center mismatch. Its failure weakens that mismatch as a sufficient explanation. All proposals had substantial movement; applied movement was zero.

For the executed children, selected discovery scores averaged60/72=83.33%, versus198/288=68.75% on fresh TRAIN games; the incumbent scored208/288=72.22% on those same fresh games. Each attempt selected a different child. These are descriptive selection/generalization observations, not a single-policy learning estimate or proof that sampling noise alone caused failure. Small discovery banks, path dependence, direction quality and rare useful directions remain live explanations.

## Evidence, cost and preserved state

| Cost | Weighted proposal | Executed child | Total |
|---|---:|---:|---:|
| TRAIN transitions |159824|159824|319648|
| Used in complete cycles |157443|154313|311756|
| Unused partial-cycle transitions |2381|5511|7892|
| Initial images |1755|1789|3544|
| Heldout transitions |0|0|0|
| Allocation wall seconds |981.600|949.207|1930.807|

Allocation runtime totals32.18minutes; session preparation, research, checks and archival are additional wall time. Both runs ended under their original immutable caps with coordinator EXIT0 and no worker failures. No cap resets, backpropagation, new architecture, privileged policy input, visual learning or final-stream expansion. RIGHTtheta6 retains its558832-transition ancestry; the319648new research transitions did not become inherited updates.

Four independent completion audits passed: Sol13.029s and Astra2.731s for weighted proposals; Sol12.331s and Astra2.749s for executed children. They reconstructed candidate selection/update arithmetic, exact maps, RNG advancement on rejection, all decisions, native records, accounting and unchanged endpoints. Audits add no observations or simulation/vision replay. Production source remains68d15a7. Focused fixtures passed before both launches; no whole-suite reruns or extra experimental allocation followed.

Roots: `/workspace/m1b-experiments/pong-confirmed-proposal-interception-20261023-r1` and `/workspace/m1b-experiments/pong-executed-child-interception-20261023-r1`. Parent proofs and figures are in `pong-mechanism-review-20261022`. Journals are losslessly archived only after both raw audits; checkpoints, summaries, source and protocols remain plain. Restore journals before any later raw audit. Interrupted historical allocations remain untouched.

## Orchestration and next research question

Astra owned the research question, outside-source reading, alternative comparison, fixed decision rules and independent reconstruction. The same established Sol6.1XHigh agent challenged the mechanism, found the earlier negative native-win confirmation precedent, proposed conditional alternatives, implemented both harnesses, ran focused fixtures and performed separate raw audits. Sol was parked during routine training and resumed for review. No token or monetary share is inferred from these roles.

Outside sources informed questions rather than guarantees: Mahsereci/Hennig on noisy comparisons; earlier ARS work on combined search proposals; Sener/Koltun on common directions for multiple objectives. Their backpropagation or exact-gradient machinery was not adopted. See the [first selection](../plans/2026-10-23-fresh-confirmation-selection.md) and [final follow-up decision](../plans/2026-10-23-executed-child-selection.md) for assumptions and primary-source links.

Stop this batch after its primary trial and one justified follow-up. No further acceptance-rule, threshold, noise or rate variant is selected. The next recommended research target is balanced acquisition from the earlier proven DIRECTtheta9 scaffold: can both phases be learned together before specializing on RIGHT? It must demonstrate fresh opponent-return acquisition while retaining direct returns and ordinary-game performance, with explicit absolute comparisons to preserved RIGHTtheta6 and the global reference. A weaker starting score must not become a free success criterion. This would be a separate research lineage; the accumulated RIGHT branch stays preserved. Freeze its complete protocol before a future launch. A larger discovery bank is an alternative hypothesis test, not an automatic sampling sweep. A major representation/history change still needs user review and stronger justification.

## Clean closure

Both coordinators exited0 and the closure verification found no active candidate workers. Final child archival completed after both audits:1,789journals,5,252,042,752→703,741,793bytes in51.592s; sidecars exited0 first in1.095s. Every original byte stream was verified before unlinking, with0observations/replays. Both completed-futility archive exceptions are restricted to exact checked roots; experiment status is unchanged. Session stopped at a research checkpoint after approximately57minutes, within the two-hour ceiling, rather than launching a third adjacent experiment.
