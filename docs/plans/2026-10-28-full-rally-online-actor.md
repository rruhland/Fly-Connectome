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


## Launched after review

Agent20b0a078 integrated as main59ac9a1;15focused tests, compilation and diff
checks passed. Parent reviewed the runtime actor, driver, tests and pinned spec,
including actual-deployed coordinate/cap assertions and attainable temperature.
Root `pong-rally-full-actor-20261028-r1`, specSHA
`c8ea9b7fdb421ece01caa986f57d37ce66d2803b32f91bd47ae60d6b523b4572`.
FreshTAG63073 reserved. Source frozen during execution. Implementation budget/
estimate600s is separate; one900s experiment clock includes preparation.
Training samples64existing TRAIN seed contexts with new on-policy action paths;
these are not64new seed contexts. Parent will independently reconstruct first-
batch eligibility and reward-to-go credit from saved actual data, without new games.


Initial calibration: temperature0.008161197682294006, centered-Q RMS0.800332956301049,
exact expected nongreedy rate5%. Parent independently reconstructed firstbatch
Q/probabilities, terminal eligibility and reward-to-go credit from all2,039saved
actual rows. Matrix reconstruction agreed within4.6e−12; zero new observations,
0.070s computation. Artifact `parent-first-batch-credit-audit.json`.
At eight batches/14,570cameras/88s, all eight updates were nonzero; reference-cache
cumulative RMS movement0.01761 (2.2% initial),119/2048greedy decisions changed.
This demonstrates effective policy movement, not improved game performance.


## Final endpoint and explicit evaluation-gate amendment

Training completed all32batches in302.67s:58,837cameras+128initials, no partial
batch. All32updates nonzero;39/64pairs differed in total reward,56/64had nonzero
reward-to-go credit. Cumulative reference RMS movement0.0348719 (4.36%initial),
165/2048greedy choices changed. Final endpoint bundle
`4a30c52437bbed7abfaa34abae3b82329663ab071362ac386d3ecde441c20b42`;
W `44371955920938801c82bd17d9434a3d3a0342e59a5d738676f21db8961e0491`.

The fixed32screen completed63→65returns,24→25first,16→15games≥2,
later39/58→40/62;12gains/11losses. It is mixed, not a reliable learning claim.
The driver exited normally at357.54s/74,001cameras+160initials because its second-
return safeguard failed. Source/assets/spec unchanged; no missing work.

Sol recommended evaluating the SAME frozen final endpoint on full256 before
another fit or representation change. Parent explicitly amended the TRAIN screen
gate: the mixed small sample is insufficient to decide practical usefulness.
The full check uses the ORIGINAL900s deadline, with no refit, checkpoint selection,
new training, or reset clock. Preserve the initial screen result. Report first,
second, total and later counts/rates plus individual tradeoffs. Parent reviews full
results before any fresh evaluation. FreshTAG63073 remains untouched.


## Completed full evaluation — rejected

The unchanged actor endpoint completed full256 at786.97s within the original900s:
662→566totalreturns,233→214first,161→144games≥2,later429/615→352/535
(69.76%→65.79%). No fresh evaluation. Total execution190,265cameras+384initials;
all128trainingreceipts and256evaluationreceipts complete, sources/assets frozen,
no missing or unknown work. Final audit passed.

On the64starts actually used during training, greedy returns fell151→141 and
first56→48 (second34→35). Other192starts fell511→425, first177→166,
second127→109. Failure therefore includes training-start deployment, not only
unfamiliar starts. Retain40069; reliable sustained-return learning remains unmet.

Parent's saved-gradient diagnostic used no new observations or alternate fitted
models:25/64pairs had equal total reward;17still produced reward-to-go gradients.
They account13.66%of summed pair energy in the inverse-G metric. Among39discordant
pairs, median alignment with terminal paired credit was.627 and six opposed.
These describe gradients across different policies/states; they are not variance
estimates or proof that terminal credit would perform better. Equal-reward noise
alone is not a sufficient explanation. No credit/temperature sweep is selected.

Next selected mechanism is a lower-dimensional, smooth correction within the
EXISTING output weights, not a new input or architecture. See the smooth-correction
plan. If that also regresses broadly, stop this actor family and prepare a concrete
continuous-representation proposal for user review rather than more variants.
