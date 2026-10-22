# Visual-state baseline: next credit-assignment hypothesis

Status: design, not yet allocated. The temporal exploration screen failed to
establish incremental learning. Read the [general scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md)
and [checked temporal result](2026-10-22-temporal-exploration-research.md).
The target remains reliable interception, not winning or a claim of completed M1B.

## Why this question

Astra and Sol independently proposed a lagged learned state baseline after the
negative temporal test. Exploration changed trajectories and produced some outcome
contrasts, but updates did not establish transferable first-return improvement.
One explanation is noisy assignment of the same terminal reward across visually
different situations. Alternatives remain insufficient representation, poor motor
timing, insufficient dose, and limited useful exploration. This test cannot rule
all of them out.

The intervention would preserve the linear actor, frozen vision, existing reward,
action law and local analytic score update. An auxiliary regularized regression
predicts first-hit outcome from legal visual features. It neither chooses actions
nor receives teacher targets. Fit only on earlier completed batches; freeze it
through the next batch. No backpropagation, Q bootstrap or privileged observations.
This is a local variance-control experiment, not a new vision architecture.

## Derivation and limitations to challenge before execution

Let H_t include history, the pre-action image and current theta-independent AR
bias. For the actual conditional categorical probabilities p_t, the score is
s_t = (onehot(a_t)-p_t) outer x_t / temperature. Conditional on H_t,
E[s_t]=0. Therefore any finite pre-action baseline v_t, with coefficients frozen
before the episode, has E[v_t s_t]=0. The raw episode estimator
R sum_t s_t - sum_t v_t s_t has the same expectation as REINFORCE, including
variable episode lengths with the usual finite-horizon stopping convention.
The fitted baseline need not identify the full Markov state for this identity.
Aggregate step clipping is a separate nonlinear intervention already shared by
both arms; do not describe the final clipped update as unbiased.

Prediction quality is not sufficient evidence of useful gradient variance
reduction. Cross-time covariance matters. In particular, the paired-start
estimator gives exactly zero when paired outcomes agree; replacing it with a
learned baseline may add variance on those pairs. More nonzero updates alone
would not be success. Prequential prediction residuals and episode-gradient
statistics are diagnostics; fresh carried-start and matched-control comparisons
must decide whether learning occurred.

[Greensmith, Bartlett and Baxter (2004)](https://jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf),
sections5 and7.1, motivates distinguishing reward prediction from baseline
variance reduction and considering score-weighted fitting. Its stationary
POMDP/GPOMDP results do not directly guarantee our finite-episode AR actor's
variance. Astra read those relevant sections, not the entire paper. A regression
objective, regularization, clipping and weighting must be selected before data;
no post-result choice between candidate baselines.

## Pending protocol decisions

Sol independently challenges the derivation and proposes one economical fitting
rule after completing the bounded execution-speed proof. Astra then selects the
comparison and fixed allocation. Provisional common ancestor is the temporal
trial's declared primary correlated endpoint, which passed retention; it is not
selected by best checkpoint or highest observed win count. Both arms must share
actor rate, cap, exploration, fresh starts and training budget. Preserve the
current first-return learning, absolute competence and all-return retention gates;
replication remains required before promotion. No learning allocation is active
under this document yet.

## Selected protocol, pending implementation and prelaunch check

Astra selected Sol's independently specified score-norm-weighted ridge, one fixed
choice rather than a sweep. For capped214-dimensional actor input x, use
u=[1,x/16], v=clip(.5+beta dot u,0,1), initially beta=0. Fit float64 beta to prior
completed own-arm outcomes only, minimizing normalized weighted squared residual
plus1e-3 times squared coefficient norm (including intercept). Weight is squared
Frobenius norm of the actual conditional score. This emphasizes decisions that
contribute to the estimator, but optimizes only a per-decision proxy. Fit after
joint batch completion; every current episode/replica uses the previous fit.

Two arms share a shadow baseline fit for cost and counterfactual diagnostics;
only the designated estimator updates each actor. Control uses existing paired
outcomes; treatment uses mean_i[R_i E_i - sum_t v_old(x_it) score_it]. Two fresh
starts times two independent replicas per arm, matched across arms, alternating
execution order. Joint updates only after all eight episodes complete. Start at
the temporal trial's primary correlated theta112; retain its fixed AR noise,
temperature, learning rate and aggregate cap. Fixed execution interval16 is
explicit in both arms. No new reward, input, architecture or noise tuning.

Budget:40,000 new training transitions per arm,20 minutes training;64 common
fresh20-second greedy games each for treatment, control and carried start,
10 minutes evaluation;30 minutes overall on the original clock. Last complete
joint endpoint only; no cap extensions or best-checkpoint selection. Prelaunch
review must verify source/artifact pins, seed disjointness, exact warm start,
estimator/recovery checks and resource staging before observations begin.

Primary first-return learning: treatment minus start at least10 percentage points
and95% lower bound above zero. Absolute80% competence is separate. All-return
retention requires95% lower bound above-5 points. Treatment-minus-control needs
positive evidence to claim a baseline advantage; replication precedes promotion.
Weighted prequential residuals, gradient dispersion and nonzero updates on
same-outcome pairs are descriptive mechanism checks, not alternate success gates.

Inherited selected lineage319,170 training transitions:298,126 used,21,044 unused.
The previous IID arm's39,920 transitions remain study cost but are not ancestors
of this carried AR actor. The completed temporal journals were losslessly archived
3.295GB to.469GB with full decompressed byte verification after both checks and
all engineering raw reads; metadata, heads and required projection remain plain.
