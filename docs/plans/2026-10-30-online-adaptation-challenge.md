# Independent challenge: carried online adaptation

Read-only research and acceptance review; no native games or implementation
changes. Companion to `2026-10-30-carried-learning-and-adaptation.md`.

## What the experiment can establish

The acquired endpoint is anonymous-pair position feedback with all five numeric
coefficients zero. A paddle-speed change can leave that same policy effective;
successful play then shows robustness, not adaptation. Unannounced A→B→A action
polarity reversal is a useful, identifiable first intervention. Passing establishes
reward-driven selection and motor remapping over inherited frozen vision and a
small expert family. It does not establish general policy learning or full M1B.

Whole-game commitment avoids assigning a hit to an expert installed just before
contact. Evidence, incumbent, exploration position, sufficient statistics and RNG
must persist across ordinary game resets and both interventions. Public visual
IDs, pending actions and within-game tracking may reset normally. No phase,
mapping, seed or external game index enters the learner. Checkpoint continuation
must reproduce the next choices and every subsequent update exactly.

## Acceptance and accounting challenges

- Predeclare all phase lengths and final 32-game windows. Every initialization
  must meet >=95% first and later returns in each final A/B/A window, including
  exploratory games, with >=100 later opportunities per window. A shortfall in
  opportunities is an unmet criterion, not permission to extend the stream.
- Include all starts, early losses, interrupted work and initial observations.
  Report first/later counts, complete and capped games, physical seconds,
  cameras, wall time, resets and own hits per physical second. A high contact
  rate alone can hide failures that repeatedly restart easy opening trajectories.
- Compare actual live late performance against no-update/uniform and
  reset-every-game controls on prospectively matched exogenous start schedules.
  Preserve control RNG across resets. Never reroll an initial expert that happens
  to be strong. Report online improvement separately from frozen endpoint skill.
- The acquired A expert with updates disabled must deteriorate materially in B
  (suggested >=20 percentage points), and the live learner must recover materially
  above it. A→B→A tests recovery twice and retention. Reversion recovery alone
  does not distinguish stored-memory reuse from relearning; compare its latency
  and retained evidence with a fresh learner if claiming faster recall.
- Recovery latency is descriptive: first predeclared 25-opportunity window at
  >=95%, followed by another such window. Report games, cameras and physical
  seconds since change; censor unsuccessful recovery at phase end. Continue to
  the fixed endpoint regardless. Do not choose the best rolling tail.
- Contacts in one game and games coupled by one learner are dependent. Report
  each independent stream; 100 contacts are not 100 independent replications.

`PongCamera.reward` is terminal score; `hits` is the separate own-hit signal.
Natural negative score is public feedback if explicitly admitted by the plan.
Positive score and artificial truncation must not invent failure labels.
`ReturnCounter` detects a missed own plane before the eventual negative goal:
a cap in that gap must count as a reporting failure, without exposing the plane
event to learning. Counter-derived opportunity/first/later labels are reporting
only. Preserve one terminal observation and account for every reset observation.

Short-memory 16-arm discounted UCB may continually revisit bad experts; a forced
exploration floor can make 95% live reliability impossible. For example, epsilon
0.1 uniform exploration with one good expert spends about9.4% of arm selections
on bad experts, before variable-duration effects. A persistent incumbent with
sparse challengers is a sensible comparison. Do not report exploitation-only
rates as the live learner's rate. Discount evidence on a declared common clock;
per-game discount and per-physical-second discount are different with early losses.

## Sources and distinct alternatives

1. [SMPyBandits DiscountedUCB](https://github.com/SMPyBandits/SMPyBandits/blob/master/SMPyBandits/Policies/DiscountedUCB.py)
   and [SlidingWindowUCB](https://github.com/SMPyBandits/SMPyBandits/blob/master/SMPyBandits/Policies/SlidingWindowUCB.py)
   implement recent-evidence bandits and cite Garivier–Moulines, ALT2011.
   Discount all arms on global updates; a sliding window covers recent global
   draws, not each arm's last samples. Pong correlation and variable exposure
   prevent simply importing their regret guarantees.
2. [SMPyBandits Exp3.S](https://github.com/SMPyBandits/SMPyBandits/blob/master/SMPyBandits/Policies/Exp3S.py)
   implements reward importance weighting plus shared mass for changing best
   arms, citing Auer et al.,2002. It avoids permanently extinguishing an expert,
   but exploration can be expensive. This is another selector, not a materially
   different control-learning mechanism.
3. [River ADWIN](https://github.com/online-ml/river/blob/main/river/drift/adwin.py)
   implements adaptive-window change detection, citing Bifet–Gavalda,2007.
   It could trigger challenger bursts from observed reward changes while keeping
   learned experts. Its default 32-sample checking clock is unsuitable for a
   tiny number of game-level samples without explicit adaptation. A detector
   receives observed reward, not simulator opportunity labels or phase flags.
4. A materially different fallback is anonymous visual motor-response learning:
   use lagged action-conditioned covariance or forgetting-factor least squares
   to identify which visual entity responds to commands and its response sign.
   Update from actual past public visual displacements and executed actions,
   including lagged actions to handle activation inertia; ignore occluded,
   wall-clipped or unsupported samples using public vision only. Retain the
   learned response across games and update after reversal. Feed its estimated
   sign into the existing relational feedback rule, without candidate-state
   rollout. This learns a sensorimotor association without expert labels, but
   assumes a controllable entity, local motion consistency and the relational
   servo prior. Action/target correlation and clipping can create spurious
   associations; independent seeded command dither and held-out prediction of
   observed past-response samples are essential negative checks. This is a
   proposal, not a source-backed claim of demonstrated Pong success.
5. A different reward-only route is local perturbation with eligibility traces
   on the compact feedback weights. [Miconi's implementation](https://github.com/ThomasMiconi/BiologicallyPlausibleLearningRNN)
   accompanies eLife2017 delayed sparse-reward learning, including a faster
   node-perturbation option. The [Brian2 Izhikevich2007 example](https://github.com/brian-team/brian2/blob/master/examples/frompapers/Izhikevich_2007.py)
   exposes separate decaying eligibility and dopamine signals with their product
   changing a synapse. Apply only a few output parameters initially, use a
   reward baseline, actual elapsed-time traces and bounded updates. Prior failed
   large actors make another large network a poor first experiment; these
   sources motivate a mechanism, not another unconstrained actor sweep.

First stop criterion: if neither fixed bandit configuration shows clear live
acquisition and B recovery in its bounded pilot, preserve the traces and switch
mechanism rather than sweep discounts/exploration constants. If a route is
promising, qualify the same pinned implementation on new independent streams
and controls before any completion claim.

## Initial implementation audit (no native games)

Reviewed `pong_carried_structure_learner.py` and `run_pong_carried_learning.py`.
The learner receives only own hits, public terminal score/status and elapsed
time; its game choice receives no arguments. Phase/seed/mapping stay on the
environment/reporting side. Whole-game ownership and per-game JSON restore are
implemented. Independent synthetic checks passed exact future continuation
including surprise-sweep RNG, midgame restore, and deterministic arm relabeling.
Learned sufficient statistics persist through all144 synthetic A/B/A games.

A perfect synthetic one-good-arm test (five hits and cap for the good arm;
zero hits and negative termination for others) gave final32 good-arm selections
32/30/30 for capped evidence and32/32/29 for discounted evidence over48-game
phases. This illustrates why the pilot's fixed last16 is a diagnostic, not final
qualification: the two-loss trigger plus full16 sweep can extend into a last32
window. Freeze longer qualification phases before collecting those data.

Prepared control streams currently have distinct start cohorts, so they support
unpaired comparisons. Prospectively shared exogenous starts are needed before
calling a later control comparison matched. Counterfactual starts are not
queried by the live learner; separately run reporting controls are compatible
with that boundary.
