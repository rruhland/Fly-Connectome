# Full-rally reward learning with the existing visual policy

## Why this direction

Complete-trajectory integration improved practiced starts but reduced fresh returns
90→69. Selective causal repair learned a genuinely useful action in5.33s but its
broader32game screen fell60→57. More isolated repairs are not selected.

Parent proposed direct on-policy reward learning with the current competent
40069 policy and its fixed nonlinear features. Reused Sol independently agreed
this should precede requesting a new representation, while emphasizing high
credit variance and incidental early actions. Earlier October22 actor trials used
first-hit objectives and older linear features; their null results remain relevant.
The current objective, representation and warm start differ, but do not guarantee
success. Current vision retains continuous entity positions while policy features
largely bin them; this is an alternative hypothesis, not a proven capacity limit.

No architecture change: keep vision, transport, raw4389 and fixed512 map assets.
Learn only existing output weights with explicit analytic categorical eligibility.
No autodiff, backpropagation through vision/features/physics, or expert labels.
This is online learning in complete-episode batches, not yet a single uninterrupted
general-purpose learning agent.

## Fixed bounded pass

Start from40069. Each frozen batch has two TRAIN starts, one per serve phase, with
two independent categorical action attempts per start. Each attempt runs the same
20-second-or-loss horizon as existing native evaluation. Every own return earns1.
Maximum32batches,64,000transition cameras and600seconds training, inside one900s
execution clock that also covers qualification. Count partial attempts; never
update from an incomplete batch. No outcome-based budget extension.

For each action, accumulate the exact score eligibility in joint output coordinates.
At each own hit, add the eligibility accumulated so far to reward credit. Subtract
an independent same-start partner's total return times terminal eligibility as a
baseline. Average across the four attempts. This reward-to-go estimator avoids
crediting later actions for already-earned hits; it has the same expected full-
rally gradient before step limiting. Equal-total-reward pairs can still produce
updates, unlike the simpler terminal-return-difference estimator.

Precondition using the existing fixed joint score metric G, without copying its
large file. Scale the direction to at most1% of INITIAL centered-logit RMS on a
fixed2048-row TRAIN cache, additionally limiting mean categorical total variation
to.01 there. Any ray bisection uses only policy outputs, never reward tuning.
Calibrate temperature once to5% expected nongreedy probability on the initial
cache, then freeze it. Report actual action changes, probability movement,
function-space movement, reward differences, clipping, and cumulative displacement.
A null with negligible policy change must not be called a representation ceiling.

Save checkpoints for inspection, but choose the FINAL completed endpoint in
advance. No best-of-TRAIN checkpoint selection. Evaluate a fixed32game greedy
screen, then fullTRAIN256 only if total/later returns improve with first/second
retention; an unused32start paired fresh comparison versus40069 follows only if
full performance and remaining time justify it. Report actual regressions.
Old fresh sets remain excluded from training; no automatic promotion.

## Implementation and efficiency

Implementation target10minutes, separate from execution. Reuse existing evaluator,
fixed readout, conditioner and qualification helpers. Avoid large per-frame graph
journals and repeated full parameter hashing; verify frozen learned contents per
episode. Preserve enough compact action/probability/reward/eligibility evidence to
check the update. Tests cover analytic score derivatives, multiple-hit credit,
independent partner baseline, coordinate conversion, caps and zero-gradient cases.
Parent reviews concrete code, calibration/specification and focused test results
before launch. No training is active merely because this plan is written.

## Sources and limits

This is an adaptation of the analytic score-function method already examined in
[the October22 actor work](../reconstruction/2026-10-22-action-credit-research.md),
including Williams1992 and the existing paired-start reasoning. Sol proposed the
reward-to-go version and the additional probability-movement cap; parent checked
that equal-return pairs may still update. No new outside result is claimed.
