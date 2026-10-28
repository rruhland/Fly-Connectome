# Causal action credit with joint output learning

## Decision

The broad replay block improved practiced games from 662 to 686 returns but
reduced fresh performance from 90 to 69, with no fresh game improving. Stop
complete-trajectory replay updates. Retain 40069 as the research baseline.

Parent proposed combining verified action credit with the working joint output
solver and outcome-based later-skill retention. Reused Sol independently selected
paired-policy divergence as the cheaper acquisition route, challenged a standalone
margin sweep, and specified a short donor-policy burst. Parent corrected two
important pitfalls before implementation: evaluate subsequent returns through the
same episode horizon, and do not simultaneously require an old target first-hit
action and its contradictory exploratory replacement.

This uses the same frozen vision, raw4389 + fixed512 features and runtime policy.
Only existing output weights learn; no backpropagation or expert action labels.
It is an own-reward, reset-assisted scaffold, not proof of continuous online RL.

## One bounded pass

Start from 40069. Select at most four previously collected TRAIN starts where 4c
returned more balls, balanced by serve phase. Freeze selection before outcomes.
At the first shared-state action divergence, execute ten camera decisions using
4c, then resume 40069. Compare own returns after that snapshot through the original
episode horizon or terminal loss. An already-earned hit is not new reward; a
first-hit-only stopping rule would conceal later improvements. Donor superiority
in a separate full trajectory is not itself evidence that this intervention helps.

For successful interventions, permit at most four predetermined chunk-deletion
trials. Retain a deletion only after actual reward verification. Explicitly record
exploratory actions and their difference from the unchanged baseline chooser.
Do not pretend a forced action was the source policy's greedy action. If there is
no verified improvement, stop acquisition without fitting.

Fit at most one verified repair per phase, once, within 90 seconds. Use the existing
joint score metric without duplicating its large file. Hard constraints cover
other games' earned first-return prefixes, target roll-in to intervention, and
verified remaining burst choices. If a target changes before its first return,
replace its conflicting old first-prefix contract; check its first return in
actual deployment. Exact duplicate action conflicts must be checked before fitting.
Later historical choices are preserved softly by the score metric. A single fixed
existing kappa margin is used for reward-verified exploratory target choices;
old guard margins remain tiny. No margin or duration sweep.

Actual deployed target returns must improve. The original five-hit skill on
104562292 must retain at least five hits in a real game, without requiring all its
old later paddle actions. Then run a representative 32-game TRAIN screen, full256
if useful, and an unused paired fresh comparison only if the full result merits
it. Judge actual first/later/total return outcomes, disclose individual tradeoffs,
and do not automatically reject a different successful trajectory.

Implementation allocation: at most ten minutes. Review concrete selection,
provenance, tests, and manifest before launch. Execution allocation: one 900-second
clock after review, covering acquisition, pruning, fit and qualification. Save
partial results; no silent retries or clock reset. Prior fresh sets remain excluded
from training. Disk is limited; avoid metric copies and unnecessary snapshots.

## Why this is worth one pass

Earlier linear causal-credit and soft-retention variants had mixed or negative
results; this is not a new credit algorithm. The joint output solver now integrates
requests that the earlier linear learner could not retain, but full action locks
transferred poorly. This pass combines that working solver with fewer experimentally
justified requests and allows later behavior to change. It may still fail because
local rewards do not guarantee broader improvement. A failed pass should change
direction, not trigger an unbounded series of margin or pruning variants.

Parent also checked Peters and Schaal's reward-weighted regression work:
https://is.mpg.de/ei/publications/peters_picml_2007 . Its use of reward-informed
regression motivates separating rewarding behavior from mandatory trajectory
copying. Its operational-space setting does not establish effectiveness for this
discrete Pong policy; no algorithm or performance claim is imported.
