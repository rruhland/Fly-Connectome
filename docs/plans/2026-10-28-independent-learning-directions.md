# Independent directions after sustained-return failures

Research note, 2026-10-28. No code, training, model fitting, native cameras, or
checkpoint promotion performed for this note. Only this document was added.
The already approved continuous visual residual pilot belongs to the existing
integration worker; these are separate hypotheses, not amendments to that pilot.
Major controller, temporal-action, or representation changes still require user
review. Research permission is not execution permission for those changes.

## Recommendation

Rank the next distinct mechanisms as follows:

| Rank | Candidate | What actually learns | Smallest useful test | Approval boundary |
|---|---|---|---|---|
| 1 | Fitted value control, preferably fixed random partitions initially | Expected own-hit return for observed action transitions, including failures | Reuse TRAIN transition tuples; learn a value policy once; actual full-game screen and conditional fresh pairs | A tree/value controller is a major change even with unchanged legal inputs |
| 2 | Learned action-conditioned model plus short shooting control | Next legal visual representation, scalar own-hit reward, termination; actions chosen by predicted returns | One small ensemble learned from TRAIN; deploy replanning on actual games | New predictive/planning modules are major; no hand-coded intercept cost |
| 3 | Sparse temporal predictive hierarchy, OgmaNeo2 family | Layer-local next-state predictions and bottom-layer reward control | One pinned no-BP implementation, small temporal hierarchy, own-reward acquisition and actual native endpoint | Replacement recurrent hierarchy is major; source/version and learning budget need review |

The goal remains repeated actual returns throughout normal games. No prediction
loss, fitted action agreement, partition occupancy, or recovered scripted branch
can substitute for that measurement.

## Why these survive the existing failures

The current 40069 scaffold's TRAIN256 result is 662 returns, 233 first returns,
429/615 later returns. Complete successful-prefix fits increased TRAIN while
losing fresh returns; causal single-action repair helped its target but regressed
its screen; both full-coordinate and smooth actor updates regressed broad native
play. These are strong reasons to stop those particular loops. They do not prove
that representation is the unique bottleneck or that reward learning is exhausted.

Fitted value learning differs from imitation: failed transitions constrain bad
actions; future reward can propagate through states without asserting that every
action on one rewarded trajectory was correct. Model-based control differs from
actor fitting: learning is supervised by every observed state transition, while
action selection searches a learned model. A predictive hierarchy differs from
adding another static random feature map: state is learned/retained across time
and prediction feedback affects the controller.

None is guaranteed to fix the problem. Visual aliasing, insufficient counterfactual
action coverage, reward sparsity, horizon length, and policy-induced distribution
shift can defeat all three. An actual learning experiment should discriminate
these mechanisms rather than turn into another capacity-diagnostic sequence.

## 1. Fitted value control: strongest independent next candidate

Primary source: Ernst, Geurts, Wehenkel, *Tree-Based Batch Mode Reinforcement
Learning*, JMLR 2005:
https://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
The paper learns Q from observed `(state, action, reward, next_state)` tuples by
repeated regression on Bellman targets. It distinguishes totally randomized trees,
which can supply a convergent averaging approximation under its assumptions,
from ExtraTrees, whose better empirical approximation does not carry the same
general convergence guarantee. Its Markov-state assumptions are not established
for our legal compressed visual observations.

Reproducible references include the explicit ExtraTrees FQI example at
https://pierrelucbacon.com/rlbook/fitted-q-iteration/ and the separate LSPI
implementation https://github.com/stober/lspi. These are implementation references,
not evidence of Pong performance in this environment. Neural fitted Q iteration
usually trains an MLP with backpropagation and is not the proposed method.

**Cheapest practical learning proposal:** build one fixed set of action-conditioned
random tree partitions over the EXISTING frozen legal visual features; learn leaf
values through repeated Bellman averaging. Fix partition RNG, topology controls,
discount and iteration count before examining outcomes. Cache leaf assignments
for both executed `(x,a)` and all three next-action queries. Then a Bellman sweep
updates leaf means rather than rebuilding an entire forest. This is the
totally-randomized/averager route, not a claim that reward-trained ExtraTrees has
been implemented. All three action queries must use the same state representation
and declared action encoding. Reward is the existing +1 own-hit signal; terminal
misses have zero bootstrap. Preserve timeout semantics rather than treating a
censored continuation as an observed natural terminal.

Start with an episode-stratified fixed TRAIN subset containing BOTH earned prefixes
AND failed suffixes, plus all available reward-earned action alternatives. Do not
sample only rewarded frames or assign reward to every frame of a successful rally.
If only baseline actions are represented in a region, report the unsupported
alternative queries; do not invent negative rewards for unseen actions. A small
fixed exploratory TRAIN collection is part of the useful test if existing action
coverage is inadequate. Any behavior-coverage fallback or action-support rule must
be declared once, not tuned against screens.

The initial test should keep primitive camera-rate actions. In the worst case,
one Bellman sweep moves reward back only one camera; choose enough fixed sweeps
to cover the observed 40–80-camera correction scale. A 5–20-sweep negative would
not test that hypothesis. Caching a few dozen trees for roughly 20–50k transitions
is the reason this can be cheaper than 80–160 complete forest refits. Runtime and
memory still need a bounded implementation estimate; no measured speedup is
claimed. An initial reviewable envelope is one 900s experiment: at most 20k new
TRAIN cameras, one fixed fit, 32 full-game TRAIN screen starts and, only on useful
completed outcomes, 32 paired fresh starts. This is a proposed ceiling, not an
authorized allocation or promise all stages fit. An incomplete fit or evaluation
is partial, not a negative result for completed learning.

An alternative is explicit 10/20-camera options with semi-Markov targets:
`sum observed discounted rewards + gamma^duration * max next-option Q`.
That makes credit propagation cheaper but changes temporal actuation and requires
separate major-change review. Existing hold10 SARSA already failed a small
from-zero acquisition screen; this is relevant negative evidence, though it did
not test batch tree value learning with the much richer current replay. Do not
naively take a behavior trajectory's 20-step return and attach an optimal max-Q
bootstrap: without correction that is not an off-policy optimal-action target.

Old online linear/DCT SARSA failures therefore reduce confidence in an easy value
solution, but they do not test this estimator, nonlinear partition class, data
reuse, or action-coverage regime. Conversely, replacing SARSA with FQI while
retaining an aliased observation may simply learn a better fit to an impossible
conditional average. Judge actual native learning, not Bellman error alone.

## 2. Action-conditioned learned model and planning

Primary methodological source: Chatzilygeroudis et al., *Black-Box Data-efficient
Policy Search for Robotics* (2017), https://arxiv.org/abs/1703.07261 ; official
implementation https://github.com/resibots/blackdrops. Its useful distinction is
learning uncertain dynamics and optimizing policy as a black box, rather than
requiring differentiable policy optimization. It is robotics evidence, not a
camera-only Pong reproduction. PILCO/MC-PILCO and neural PETS-style systems are
not automatically whole-algorithm no-BP precedents merely because GP inference or
their shooting planner needs no backpropagation.

Learn a small action-conditioned transition ensemble by ridge/RLS, trees, or GP
regression from legal frozen visual features and executed actions. Learn reward
and terminal probability from the existing scalar own-hit/terminal events. Supply
no ball/paddle identity, intercept coordinate, simulator mechanics, analytic bounce
law, geometry-shaped cost, or external expert target. An action-conditioned
prediction of how vision changes is a learned target from observations, not an
expert action label.

Use short random shooting or planning-CEM sequences, score predicted own hits plus
a learned terminal value, execute only the first action and replan. Planning-CEM
is a search subroutine over a learned transition model; this does not repeat the
already exhausted native parameter-search family. Keep planning horizon and
candidate budget fixed. With sparse reward, a very short horizon without a learned
value tail may see all-zero scores and is not a meaningful control test. Long
open-loop rollouts risk model exploitation and compounding prediction error.

The cheapest useful version reuses current TRAIN transitions, adds one bounded
action-diverse collection if needed, fits once, and immediately tests the deployed
planner on full native games. Predictive accuracy is a debugging check, not a
separate research campaign or success gate. A bounded 900–1200s first pilot with
20–30k new training cameras and the same 32-screen/conditional-32-pair structure
is an estimate requiring code-specific timing. The parent is independently
developing this direction; do not duplicate implementation here.

## 3. Temporal predictive hierarchy: credible outside project, uncertain budget

Primary author report:
https://ogma.ai/2020/03/beating-atari-pong-on-a-raspberry-pi-without-backpropagation/
The author explicitly distinguishes the older routing method, which still used
backpropagation, from the March2020 system, which removed it. Higher layers predict
the layer below one timestep ahead; reinforcement learning occurs at the bottom
using temporal feedback. The Pong demo used two temporal layers plus an image
encoder, with 4×4×32 sparse representations, on a Raspberry Pi4. This is a relevant
camera-based, low-compute architectural lead, not independent confirmation of
sample efficiency or our sustained-return criterion: the report gives no complete
training-frame budget or comparable paired outcome statistics.

Source: https://github.com/ogmacorp/OgmaNeo2 ; current preferred successor:
https://github.com/ogmacorp/AOgmaNeo . Do not silently treat the current successor
as the exact 2020 demonstrated algorithm. Pin a version and verify that the actual
reward path has no routing BP or teacher labels. No foreign code was executed in
this research. A useful first experiment would learn from the existing legal
camera stream/reward interface using one small pinned hierarchy and a fixed
acquisition budget, then test normal native games. It is a larger implementation
and architecture commitment than FQI and should be ranked behind it; no claim
that a ten-minute test would reproduce the demo is justified.

## Other outside evidence and rejected shortcuts

Capone and Paolucci 2024, *Towards biologically plausible model-based reinforcement
learning in recurrent spiking networks by dreaming new experiences*:
https://www.nature.com/articles/s41598-024-65631-y ; official code
https://github.com/cristianocapone/biodreaming . The paper reports both RAM and pixel
conditions and supplies action-conditioned predictive/dreaming evidence. But the
official `PG_pong_dreaming.py` specifically instantiates
`Pong-ramDeterministic-v0`, calls `import_ram`, sets 2000 iterations per repetition,
up to100 active decisions plus warmups, and runs ten repetitions. It is not a
drop-in camera-only small-budget demonstration. Source contains local analytic
updates and NumPy/Adam machinery; Adam itself does not imply BP. A port still
needs its complete reward/model path checked. The RAM path and built-in semantic
Pong helpers must not become our legal inputs or labels.

Plain ARS/CEM/native parameter noise is **not fresh** here. October21–25 population
search, exact-child confirmation, action-dose review and October28 coherent raw
perturbations already exercise that broad mechanism. Renaming it CEM would not
justify another allocation. A tiny controller learned from hand-crafted ball and
paddle coordinates also violates the intended comparison even if it trains fast.

A fixed reservoir can be trained without BP if its output uses RLS/TD algebra,
but many reservoir RL papers still train the readout with ordinary backpropagation.
Moreover current vision already tracks velocity; memory is not proven absent or
the dominant limitation. A reservoir-only feature expansion is a lower-priority
candidate than action-conditioned prediction or value learning, not a fourth
equally supported recommendation.

## Combinations that have a purpose

1. **Current residual representation + fitted value learning:** if the approved
   representation pilot completes, its frozen features can feed a separately
   reviewed value learner. This tests a different learning rule without starting
   another representation search; positive representation evidence is useful but
   not a prerequisite for trying fitted value control on current features.
2. **Learned model + fitted value tail:** use real transitions for value learning,
   and a short uncertain model rollout only for action selection. The value tail
   gives sparse rewards influence beyond the planning horizon. Keep real and
   imagined transitions separately labeled; synthetic rewards are predictions.
3. **Learned model + local own-reward branch acquisition:** use model predictions
   to propose a small number of promising branches, verify their rewards in the
   native game, then include both successful and failed transitions in value
   learning. This exploits the existing recovery machinery without declaring its
   forced actions individually optimal or returning to whole-trajectory cloning.

Do not bundle all of these into the first pilot. A shared legal transition cache,
fixed evaluation protocol and distinct learned endpoints let mechanisms be tried
independently and combined after actual gains. Failure of the current residual
actor alone should not automatically select another actor/map sweep; the next
strong independent choice is fitted value control.
