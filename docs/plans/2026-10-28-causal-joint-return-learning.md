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


## Fixed acquisition selection and interpretation

Before intervention outcomes, select two starts per phase, prioritizing a shared
action divergence after an earned hit, then recorded donor advantage, shorter
roll-in and seed. Left:3238414060 (transition147, one prior hit,3→5 in saved
policies),1482281654 (transition27, no prior hit,2→6). Right:3016544790
(transition138, one prior hit,2→5),1601498681 (transition323, two prior hits,4→5).
The old full-memory conflict on3016544790 is rechecked against the new first-only
contract; it is not assumed to apply forever.

A successful burst establishes a useful joint pattern, not the individual causal
value of each action or inferiority of an unqueried third action. The fit requests
selected successful burst actions as greedy behavior to integrate. Actual deployed
rollouts determine whether that request improves the policy. One exact null replay
checks baseline/context/RNG equivalence before accepting intervention evidence.

The104 five-hit check is a practical safeguard, not a user-specified invariant.
If it conflicts with a materially improved representative screen, retain results
for parent review rather than silently discard a potentially useful policy.


Implementation needed a separately authorized four-minute completion extension
for the compact runner and tests after the initial ten-minute allocation. Actual
time must be reported; the execution clock has not started. Parent reviewed the
core solver change, intervention/replay module and runner before tests. No metric
copy, new cache framework, or architecture change was added.


## Launched

Agentc9295e1 integrated as main789648b. Thirty-five focused tests, compilation and
diff checks passed. Parent reviewed all four changed files and concrete manifest
before launch. Root `pong-rally-causal-burst-20261028-r1`, sibling manifest SHA
`c01b37f2d52457abc1ae33429ff341069bdd090f672d1842f351aecb00e62280`.
Execution gets one900s clock; implementation840s estimated separately. Existing
kappa0.03785729178850489 is fixed for requested burst differences; ordinary guards
use kappa×1e−4 capped by half source gap. FreshTAG63057 reserved before acquisition.
No active-code changes during execution.


## Completed causal-burst outcome — rejected

The pass completed in80.737s, before the brief workspace restart; no recovery or
repeated execution was needed. It used9,269transition cameras+11initials. Source,
models, metric, bank and specification stayed unchanged. No full-bank or fresh
qualification was launched after the negative screen.

Acquisition used4,607cameras+4initials. Three interventions failed to improve
returns (3→2,2→2,4→4); seed1482281654 improved2→3. Pruning retained only one
changed action at transition27. The exact null replay passed. This was an
experimentally successful action pattern, not a fabricated expert label.

One joint fit took5.332s and certified21,152rows, with232other first-return
contracts, the target roll-in and one explicit exploratory request. Actual target
play retained the2→3gain, but104562292 fell5→3. The representative32screen
finished60→57returns,24first unchanged,16games≥2 unchanged, later36/56→33/55.
There were two gains of one return and losses of two and three returns. The
candidate is rejected for broader performance, not merely its104tradeoff.

Candidate bundle85b35d652d64b7e62597985ac40adb43b44df46acd986a3b267cadfe464bf686,
W e5cab6a844c124119c05ea50a27d4de467444d4d57c9d7cf00fa757910f92d95.
Keep40069. Fast fitting and local deployed improvement are demonstrated; reliable
sustained-return learning is not. Do not automatically expand one-witness repair
or margin variants. Parent/reusedSol are reconsidering on-policy learning versus
representation continuity, taking earlier actor, smoothing and credit failures
into account. FreshTAG63057 remains unused and reserved.
