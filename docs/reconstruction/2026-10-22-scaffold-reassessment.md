# Interception reassessment after nonlinear residual comparison

The approved [residual test](../plans/2026-10-22-nonlinear-residual-review.md)
failed to establish improvement or retention. Keep temporal ARtheta112, not either
failed endpoint. The general [continuous-learning plan](../plans/2026-10-21-continuous-learning-scaffolds.md)
remains the objective; no M1B completion claim.

## A rejected credit hypothesis, before spending new training

Sol independently proposed differentiating the likelihood of the latent Gaussian
logits rather than the conditional categorical actions. It preserves behavior,
architecture and reward, with analytic outer-product credit and no backpropagation.
For centered AR noise b and visual features x, the score is the initial
b0 outer x0 / sigma² plus successive noise innovations outer (x_t-rho*x_prev),
divided by (1-rho²)*sigma². It replaces rather than supplements the categorical
score. Holding latent logits fixed in the likelihood derivative matters.

Astra independently verified the identity against the categorical score and
finite-difference expected reward on a rank-two Gaussian, two-step toy with
action-dependent observations and early stopping: maximum errors1.16e-11 and
1.08e-11,0.909s, zero Pong observations or production imports. This establishes
an identity on that toy, not a variance advantage or learning benefit.

Sol's prospectively bounded retained comparison used864 complete episodes,
79,460 old feature rows,108 complete batches in each of two historical cohorts.
No new observations, updates, journals or replay;1.123s. On identical trajectories,
latent versus categorical paired-gradient dispersion was18–19x larger. Its
mean-gradient-energy/dispersion proxy and accumulated-direction coherence were
also worse. Historical head drift and few discordant pairs limit inference;
these are descriptive statistics, not a variance theorem. Both agents reject
allocating fresh training to the raw latent-score alternative on this evidence.

Artifacts: `/workspace/m1b-experiments/pong-latent-credit-research-20261022/`
contains `derivation.md`, `compare_retained.py`, `retained-comparison.json`;
`/workspace/m1b-experiments/pong-mechanism-review-20261022/`
contains `check_latent_score_identity.py/json`.

## Competing next directions, not yet allocated

Sol recommends an ordinary initial-leftward-serve scaffold, keeping the actor and
learning rule. It removes the opponent-first opening without teleporting near a
collision or supplying hidden state to the learner. Seed filtering is nevertheless
explicit task-specific training-distribution assistance. Astra challenges whether
this subset is already easy and whether it truly concentrates useful credit:
inspect retained per-direction success, horizon and paired-outcome discordance
before selection. A separate fresh unfiltered evaluation must test prior skill.

Another plausible explanation is insufficient collision precision or motor-state
information. The nonlinear test cannot repair missing input information; its
substantial trajectory changes without gain leave that explanation alive. A
representation change needs a concrete information deficit and potentially major
architecture review, rather than another random-feature sweep.

Astra read Florensa et al. (2017), introduction and sections3.1–3.2,
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf).
They distinguish a changing training-start distribution from evaluation on the
original distribution. Their method grows starts backward from a goal and assumes
reset access and useful local reversibility. Our possible serve filter is a much
narrower manual curriculum, not that algorithm; their results do not establish
transfer or learnability for this visual Pong actor. We import no backpropagation,
goal-state examples or success guarantee.

## Selected direction after retained split

Sol's direction split used existing reports only (70.7ms, zero observations,
replay or journal reads). On two fresh historical DEV cohorts with the SAME carried
ARtheta112, direct incoming starts yielded33/65 first returns; opponent-first
starts46/63. Median incoming times were0.842s versus2.575s. In the retained paired
training cohort, direct starts had18/104 outcome-discordant pairs versus11/112,
with mean46.46 versus134.38 camera transitions per episode. These retrospective
figures motivate a prospective learning test; they do not establish curriculum
benefit or explain causally why the original learner plateaued.

Astra selects one bounded learning arm on ordinary direct-serve starts, with fresh
matched carried-policy controls for direct-serve learning and separate unfiltered
full-game retention. Keep architecture, visual processing, paired categorical
credit, exploration, learning rate and cap unchanged. No concurrent ordinary-start
learning arm: this tests whether incremental scaffold learning occurs, not whether
filtering is causally more efficient than ordinary training. Exact protocol and
source require prelaunch review. Independent replication follows a positive result.

This is a shorter-horizon but empirically HARDER subset: the paddle has less time
to align. The competing prediction is that visual precision or motor timing remains
limiting and filtering will not help. Preserve the complete ordinary reset joint
state; initial height and angle share one random draw. Filter by predetermined
canonical direction only, never by outcomes, landing locations or changed parity.

Astra also challenged fixed feature-metric preconditioning as an alternative to
raw-coordinate updates. It could address correlated/rank-deficient features, but
would return to an optimizer intervention without resolving task or information
limitations. Sol's strongest alternative was restoring legal visual information
lost by projection; the post-projection tanh branch cannot do that. Neither is
implemented or ruled out; the direction evidence favors the smaller task-distribution
experiment now. No additional major architecture change is authorized by this choice.

## Accepted bounded protocol for implementation

One linear ARtheta112 learner, at most40,000 training transitions/15minutes,
400 batches maximum and20 complete batches minimum. Each batch has two fresh
ordinary initial-leftward serves with two independent action/noise replicas each.
Keep the10s first-hit training task, strong opponent and all learning constants.
Use the last complete endpoint; incomplete work counts as exposure but is not used.

Evaluate endpoint and exact carried control on64 fresh direct-serve starts through
first own return or ordinary terminal loss, and64 separate fresh unfiltered20s
full games. Existing greedy evaluator with first-hit cutoff serves the primary
assessment; no sampled actor in evaluation. Count all256 evaluation jobs, their
initial images and actual frames. Evaluation cap10minutes;25minutes overall.
No outcome-based extension or DEV checkpoint selection. Budget does not reset.

Primary direct-serve lift must be at least10 percentage points with a paired95%
confidence lower bound strictly positive. Absolute80% direct-serve competence is
separate. Full-game first-return and all-return retention each require the paired
95% lower bound above minus5 percentage points. Rightward subgroup is descriptive.
Positive learning requires independent replication before endpoint promotion;
restricted80% is not full-task80% or M1B completion.

Seed rejection uses only canonical ordinary initial direction and preserves the
entire reset state. Record proposed/rejected seeds and prevent collisions with
previously reserved streams, including unused seeds. Never expand reserved final
roots. This is assistance in training distribution, not learner observation access.
Sol implements compact coordinator reuse and a narrow worker-purpose validation
extension for explicit greedy first-hit evaluation; Astra reviews before launch.


## Checked stage and launch

Source `/workspace/Fly-Connectome-m1b-pong-leftward`,865a187602d273fb9ae8e877e8d3b1203890ad32.
Allocation `/workspace/m1b-experiments/pong-leftward-curriculum-20261022-r1`.
Driverdd72feaa9a9ba0b1074e8644da05a9d5e949bde0efe7d66d1f9b447e5214cc45;
protocolb5ea8994551315f3366de9e698087a05a120baf788ef0acbda828a1070c5ba7e;
prelaunch manifest57cd63b924c2e2a618c33daa72deabc58040fb5e104f4fbcdf434b47b727e4e8.
Sol13worker tests PASS1.32s,4coordinator fixtures PASS1.67s. Retained success/miss
cutoff proof171old cameras+2initials,5.786s; exact prefixes, native rewards, explicit
task stop, ordinary loss and completion/recovery checked, zero candidate observations.

Astra reviewed the narrow worker extension and coordinator. Final runtime accounting
now includes statistics/bootstrap, masking gates if reporting crosses a deadline.
Independent stage reconstruction73ms, zero observations: exact carried head/map,
source/file pins,4,128 accepted IDs,928 canonical environment starts,953 rejected
draws preserved,30,811 prior IDs excluded. Conditional schedules independently
regenerated, including rejected draws. No final streams expanded.

Launch exec81986, original clock1792688471502393151ns. Freeze source/driver/protocol
through both postexit checks; DEV blind. Previous residual journals were losslessly
archived3.447GB to.600GB,38.34s with full decompressed-byte verification before
launch. No archive replay or new observations. No training cap extension authorized.
