# Joint learning of the existing linear and pairwise outputs

## Decision and motivation

Keep the provisional eb policy and all source checkpoints immutable. The projected
random proposals did not verify within their solver budgets. Direct evaluation of
the same proposals produced attractive eight-game scores but substantial losses
on broader training games. Both qualified proposals are rejected; stop that search
family rather than tune its noise scale.

The rejected fc policy nevertheless improved 54 individual training games. Several
have one return under both 3ac and eb and six under fc. These are usable examples
of the learner's own rewarded exploration, not expert action labels. They do not
make the rejected policy a better whole-game controller.

Parent and reused Sol selected a different learning scope: jointly adjust the
EXISTING linear output weights and the approved fixed 512-feature output weights.
The current runtime already supports both arrays. Vision, observations, feature
map, normalization, action timing and architecture remain fixed. This is an
optimizer change under the standing experimentation mandate; no backpropagation.
The saved 3ac file stays immutable; a candidate gets a new linear-weight file and
bundle, with hashes recorded separately from the carried policy.

The hypothesis is that the sparse linear coordinates can compensate for broad
changes caused by the fixed pairwise features. Prior linear-only and residual-only
failures do not prove the combined output space cannot incorporate useful behavior.
This is a hypothesis to test, not a claim that capacity is the diagnosed cause.

## Preparation before fitting

Sol independently chooses a demonstrated later-return target from existing native
receipts, its useful reward horizon, and an efficient joint-coordinate solver.
Check exact input/action conflicts before fitting. Replace a target's old prefix
with its improved rewarded prefix where necessary; do not demand both divergent
trajectories. Retain the acquired five-return behavior on 104562292 and justify
whether the remaining retention scope is early prefixes or complete successes.

Parent shortlist, without new cameras: 4195433914 and 3386520763 have eb=1,
3ac=1 and fc=6 returns, with second returns at prefix lengths 201 and 210.
Other candidates include 3982055578 (1/1/2), 1212947804 (1/1/6), and 2897272218
(1/1/2). Novelty is only relative to those named checkpoints, not all past work.

Use fixed input scales for optimizer conditioning and distinguish source-policy Q
validation from current-reference Q in update constraints. Avoid constructing a
dense 74,000 by 9,802 inequality matrix; use sparse or working-set dual operations.
Persist the chosen target, metric, retention scope and budget before fitting.

## Execution outline, pending concrete implementation review

One bounded fit, approximately 60 seconds, with no target/guard/ridge retry loop.
Require exact deployed action verification, then actual target execution. Follow
with a representative 32-game training comparison, the full 192-game bank if
useful, and unused fresh games only after broader improvement. Eight games alone
proved too weak for candidate selection in the preceding tests.

Report actual returns and first/later retention separately from fitting success.
Preserve all rejected candidates and count preparation, fitting and native frames.
No fit or new native execution is authorized until parent reviews the concrete
plan and implementation. This is internal orchestration review, not a new request
for user permission to change the architecture.

## Outside evidence checked by parent

[Zhang and Suganthan (2016)](https://doi.org/10.1016/j.ins.2015.09.025), primary
abstract/search excerpt, reports benefits from trainable direct input-output links
in a classification evaluation of random-feature networks. A useful counterpoint
is [Dudek (2020)](https://arxiv.org/abs/2003.13090), whose primary abstract reports
little benefit from those links on typical nonlinear regression tasks. This mixed
evidence supports a test, not an assumption of success. Neither study establishes
RL retention or Pong performance, and our TensorSketch is a different fixed map.
We are using the two output paths already present, not adopting either architecture.

## Concrete plan selected

The zero-camera conflict screen found 40 fc successes exceeding both named 3ac/eb
results; 18 were compatible with complete earned-memory retention after replacing
the selected target's old prefix. Choose later target 4195433914: eb=1, 3ac=1,
fc=6; its 201-row prefix contains two actual earned hits. Train that prefix, not
an unearned six-hit claim. It has no exact raw-input/action conflict under either
early-prefix or complete-prefix retention.

Retain ALL other current successful prefixes in one deliberate test. Existing
memory has 74,464 rows before replacing the target's old one-hit prefix. This
includes the acquired five-return behavior on 104562292 and avoids the later
forgetting seen with only early guards. No post-fit guard growth or target retry.

Use 4,901 existing coordinates: raw inputs conditioned by their fixed metric scale,
plus the fixed pairwise outputs. Apply reversible block mean-diagonal RMS scaling
for numerical conditioning only. Deployment scales, vision and map assets remain
unchanged. Minimize equal-episode full combined score change with cross-block
covariance and fixed 1% ridge. Undo coordinate transforms into a NEW linear-weight
file and pairwise bundle; validate both arrays and all original deployed choices.

Use active-set dual kernel solves, with Cholesky right-hand sides only for active
rows. Evaluate all constraint residuals using sparse raw inputs and dense 512-feature
products; do not materialize a 149,000 by 9,802 matrix or whiten every stored row.
One 60-second fit and one 900-second execution allocation, implementation recorded
separately. Actual target must earn at least two returns; representative32 includes
target plus a fixed uniform phase-balanced draw, then full192 and unused fresh
qualification if useful. Parent approved implementation and will review code/tests
before launch. Tests must cover distinct source/reference Q, nonzero carry, both
output arrays changing, coordinate undo and an explicit small primal comparison.

## Outcome

Implemented main7e53729,25focused tests passed. Fit18.53s verified all74,623
original constraint rows. Actual target1→4; full192493→523returns with no perseed
loss,174→175first and113→123games≥2. Fresh32 was mixed/negative:73→65total,
29→30first,21→19games≥2,later44/68→35/62. Thus integration and retention work
on practiced games; transferable improvement is not established. Preserve this
checkpoint as an integration scaffold, alongside all prior carries.

An environment interruption left63of64fresh receipts saved. Parent recovered
only the missing carry game under the original900s deadline. Native completion
865.718s including interruption; fresh300s substage overrun explicitly recorded.
At least105,587transition cameras+191initials documented; unsaved interrupted
work unknown. Exact all-saved-action/Q audits and192TRAIN invariants passed.
See results ledger for hashes, confidence intervals and complete accounting.
