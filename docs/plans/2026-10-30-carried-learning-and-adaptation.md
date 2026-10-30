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
maximum144,000 cameras and1,200 wall seconds per stream,two workers.
The wall limit was increased before launch using measured native throughput;
600 seconds could censor successful streams. Game/camera budgets are unchanged. Freeze start
schedules and hyperparameters before running. Report every exploratory game.
No outcome-dependent extension of an incomplete pilot. Prepare control support
but run controls after identifying a promising route, rather than spending full
budgets on demonstrably poor mechanisms.

A frozen control must retain an acquired A controller and disable updates before
B, so it tests whether adaptation is necessary. A reset-every-game control clears
learned evidence but continues RNG, avoiding a constant first-arm artifact.

## Qualification, before declaring completion

Freeze a promising implementation and use independent new learning streams.
Final confirmation will use two new independent live initializations, fixed phase boundaries and
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

## Prospective confirmation efficiency decision

After initial pilot acquisition and reversal recovery, parent and independent
reviewer agreed on two fresh qualification streams, with one matched reset/frozen
control pair on replica0. Pilot remains development evidence, not a third untouched
qualification stream. Controls use only the same fixed final32 starts per relevant
phase: reset-every-game has no learning history, frozen has fixed acquired weights,
and both reset transient vision each game, so simulating unused prefixes adds no
necessary controlled evidence. Report the scope as one controlled replication and
two new learning streams. Live phase length64or80 is chosen from completed pilot
recovery before qualification starts; all final windows/thresholds remain fixed.
Budget uses worst-case cameras/measured84cameras persecond perworker plus20%margin,
set before execution. No outcome-dependent extensions.

## Infrastructure interruption and selection decision

Cloud restart interrupted the fixed r2 pilot at142/144 capped-evidence games and
137/144 discounted-evidence games. No process survived; all279 completed receipts,
checkpoints and event tapes passed independent audit, with no orphan tapes.
178,913 transition cameras and279 initial observations are saved. Charge up to
one additional unsaved1000-camera game and one initial per stream; do not treat
missing work as zero. Neither pilot is complete or qualified. No missing pilot
starts will be replayed or its clock extended.

Choose the unchanged discounted-UCB-like learner for fresh confirmation. Both
methods acquired and adapted, but the capped method launched a second unnecessary
16-arm sweep after ordinary B losses; its B endpoint is poor. Discounted learner
held reliable B behavior and recovered A again before the interruption. The
independent reporter confirmed two consecutive phase-anchored25-opportunity
windows at>=95% in each observed phase (confirmation atA28/B36/A-return33),
which is descriptive development evidence, not the unfinished fixed-tail pass.

Prospective confirmation: two new initializations,64games per phase, unchanged
learner/parameters, fixed last32 first/later targets and100later minimum. One paired
reset/frozen control pair on replica0. In addition to>=20percentage-point control
lifts, the frozen acquired A controller must itself retain>=95% first/later with
100later in A-return. This is a new independent confirmation, not a pilot retry.
The earlier 'wait for completed pilot' selection condition cannot be fulfilled
after the external interruption; selection is explicitly based on the audited
completed prefix, with confirmation untouched.

## Closed outcome

Fresh confirmation passed all planned criteria and independent audits. All544
games completed; two live streams retained state across A→B→A, with1,063/1,071
returns in the six fixed last32 windows. Whole live streams including exploration
returned1,719/1,815. Paired reset/frozen controls established knowledge-retention
and adaptation effects on replica0. No next native experiment is selected.
See [complete report](../reconstruction/2026-10-30-carried-learning-and-adaptation-results.md).
This closes the structural-selection/motor-remapping scaffold, not fullM1B or
general continuous policy learning.
