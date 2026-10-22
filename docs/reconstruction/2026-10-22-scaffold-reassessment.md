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
