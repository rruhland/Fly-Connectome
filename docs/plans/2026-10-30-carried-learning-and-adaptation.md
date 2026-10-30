# Carried online learning and adaptation

User authorizes orchestration until carried learning and continued adaptation are
demonstrated or they request a stop. No backpropagation, privileged controller
inputs, expert action labels or simulator lookahead. Own-hit reward remains an
explicit scaffold. Major architecture approvals are waived for this milestone.

## Target and honest scope

Build one chronological learner whose experience changes its choices during
actual play and survives game resets. Establish acquisition, retained performance,
and recovery from unannounced actuator reversal, then reversal back. No phase
flag, schedule, seed or actuator mapping is passed to the learner. This is a
finite-controller online-selection scaffold over inherited frozen vision, not
learning a general policy from raw pixels. The previously validated controller
is not installed as the initial incumbent or given special priority.

Parent owns hypotheses, acceptance and interpretation. Sol XHigh execution owns
new learner/runner code; independent Sol XHigh challenge checks failure modes,
references and acceptance. No live old agents existed at start; two new workers
were created. Avoid parallel edits to pinned historical code.

## First bounded comparison

Two mechanisms: capped recent reward evidence with a persistent incumbent and
loss-triggered challenger exploration; versus discounted evidence with a UCB-like
exploration bonus. Same16 structures, seeded neutral ordering. Whole-game
commitment; update evidence on each actual own-hit and on public negative terminal
score. Positive terminal score and time caps do not invent misses. Existing visual
tracking resets per game; the learner and its random state do not.

Pilot: two sequential streams,48 games per phase,A→B→A,20-second caps,
maximum144,000 cameras and600 wall seconds per stream,two workers. Freeze start
schedules and hyperparameters before running. Report every exploratory game.
No outcome-dependent extension of an incomplete pilot. Prepare control support
but run controls after identifying a promising route, rather than spending full
budgets on demonstrably poor mechanisms.

A frozen control must retain an acquired A controller and disable updates before
B, so it tests whether adaptation is necessary. A reset-every-game control clears
learned evidence but continues RNG, avoiding a constant first-arm artifact.

## Qualification, before declaring completion

Freeze a promising implementation and use independent new learning streams.
Proposed final assessment: at least3 initializations, fixed phase boundaries and
last32-game windows with at least100 later opportunities per phase; >=95% first
and later return rates INCLUDING exploration in every final window. Report all
acquisition/adaptation losses, physical time, wall time and recovery latency, not
only successful tails. Compare against no-update and reset controls; frozen
performance must deteriorate after intervention to make adaptation meaningful.
Persist/reload learned state and verify exact continuation; demonstrate that game
resets preserve it. Check reporting opportunities independently of reward updates:
a camera cap can occur after a missed paddle but before terminal score.

If control reversal passes, explicitly call it motor-remapping adaptation. Do not
claim arbitrary environmental adaptation. Consider a distinct actuator-bias or
visual perturbation only if necessary to probe broader capability; do not exhaust
failed parameter routes. Next decisions depend on live curves, not endpoint-only
validation.

## Outside evidence

Read primary SMPyBandits DiscountedUCB implementation:
https://github.com/SMPyBandits/SMPyBandits/blob/master/SMPyBandits/Policies/DiscountedUCB.py
It discounts arm evidence and combines mean reward with uncertainty bonuses,
referencing Garivier and Moulines, ALT2011. This motivates a comparison; theoretical
bandit guarantees are not asserted for our temporally correlated, variable-length
Pong outcomes. Continual uncertainty-driven exploration may be too costly for95%
online reliability. Web search connector failed; direct primary project source
was accessible. Independent challenge is researching additional sources.
