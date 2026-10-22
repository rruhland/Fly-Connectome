# Research target: exploration that changes interception outcomes

Status: launched after independent code/protocol/calibration review. Source,
driver and protocol frozen through both outcome checks.
Ongoing orchestration remains authorized. Follow the
[continuous-learning scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md).

## Question and reason for changing direction

Can temporally sustained exploration produce useful return experiences and
retained interception learning without backpropagation? The previous
[paired-start actor](2026-10-22-action-credit-research.md) changed actions in
202/210 same-start pairs, often early, but changed the first-return outcome in
only 16. Conditioning the reward baseline on the serve removed many apparent
contrasts caused by easy versus difficult starts. Neither actor trial qualified
as incremental interception learning. Another rate or batch-size sweep is not
the selected response.

Astra proposed a temporal-exploration mechanism based on motor filtering. Sol
independently derived the conditional estimator, challenged amplitude confounding,
and compared representation/timing and insufficient update-size explanations.
The agreed comparison uses independent versus correlated additive logit biases
with the same stationary marginal distribution. This changes exploration while
preserving the visual representation, linear policy class and camera cadence.

## Fixed comparison

Carry the latest paired actor theta105, whose separate rally-retention screen
passed. Keep the frozen visual model, feature transform/cap, temperature,
learning rate and original aggregate update cap. Every episode begins normally;
no simulator-state forks, privileged observations, oracle labels or teacher.
Scalar first-own-hit reward remains the explicit interception scaffold.

At every camera, sample the action from
`p = softmax((theta x + b) / temperature)`.
The local eligibility remains `(onehot(action)-p) outer x / temperature`.
The Gaussian bias process is independent of theta, so conditioning on its sampled
history requires no noise-density derivative. No autodiff or backward propagation
through the visual system, recurrent state or physics is introduced.

- Independent arm: fresh centered Gaussian bias each camera.
- Correlated arm: stationary AR(1) centered Gaussian bias with
  `rho = exp(-.02/.25)`, approximately .9231.
- Both arms initialize from the same stationary distribution and use the same
  fixed marginal scale. Action and bias streams are independent within and
  between episode replicas.

The .25-second time scale is borrowed from the known motor dynamics. Record this
as hyperparameter assistance, not discovered knowledge. The policy still reacts
to every new camera; no action holding is added.

Choose one noise scale using only the existing reward-free generic feature cache:
expected disagreement with the **unbiased carried greedy action** targets 10%.
Calibrating disagreement with the biased winner would answer the wrong question.
Both arms share this scale and the existing temperature. Match the stationary
marginal on this cache; do not claim or force equal on-policy noise after states,
biases and episode survival become correlated. Report actual disagreement with
both unbiased and biased greedy choices, plus conditional action entropy.

Two fresh starts, each with two independent action/bias realizations, form each
arm's batch. The conditional paired baseline is unchanged. Use common fresh
starts across arms; alternate execution order. Apply both updates only after all
eight episodes complete. Partial joint work counts toward exposure but supplies
no accepted update. Last complete joint endpoints only.

## Bounded allocation and decision rules

Selected caps: 40,000 new training transitions per arm, 30 minutes training;
64 fresh common 20-second full games for correlated, independent and carried
start heads, 192,000 maximum evaluation transitions, 15 minutes evaluation;
45 minutes overall. These are caps, not a promised runtime. At least 20 complete
joint batches and both reward outcomes per arm are needed for an eligible result.
No outcome-based extension or development-set peeking.

Evaluation is greedy with **no exploratory bias**. Primary correlated-versus-start
learning screen: at least +10pp first returns and paired 95% lower bound above
zero. Absolute 80% interception is a separate provisional competence target.
All-return retention uses the separate -5pp lower-bound margin. Report the same
learning/retention screens for the independent arm and the paired correlated-minus-
independent contrast. Claim a coherence advantage only if correlated learning
passes and its advantage over independent exploration has positive evidence.
If both learn similarly, that supports learning without isolating coherence as
the cause. Replication is still needed before promoting a scaffold. Wins are
only descriptive. Report at least 2/3/4 returns and censored opportunities.

If correlation raises outcome discordance but greedy learning remains absent
despite substantial policy movement, exploration loses priority. Reassess credit,
visual precision and timing instead of automatically sweeping correlation, noise
scale or learning rate. If policy movement stays tiny, do not infer a visual
representation ceiling. The next decision must account for both outcomes and
mechanistic diagnostics.

## Outside evidence and execution efficiency

[Plappert et al., 2018](https://arxiv.org/pdf/1706.01905), introduction and on-policy
derivation, motivates separating temporal exploration from learning rules.
Its backprop-trained methods are not used. [Sehnke et al., 2010](https://archive.air.in.tum.de/Main/Publications/Sehnke2010a.pdf)
provides a competing perspective on action-noise variance. Neither paper proves
this intervention will help this visual Pong learner.

Before this trial, exact-byte graph-hash streaming removed redundant large
buffer copies while preserving full content verification and existing hashes.
32 focused tests passed; a retained checkpoint's hash was 1.50x faster in an
isolated benchmark. End-to-end throughput remains to be measured. Completed
journals are losslessly compressed only after all checks; staged transforms and
recovery metadata remain available.


## Launch record

Trial `/workspace/m1b-experiments/pong-temporal-exploration-20261022-r1`, source
`6d4923bfa55cd1f54b7c7197e49551fea3ed0fa1` in isolated
`/workspace/Fly-Connectome-m1b-pong-temporal-bias`; agent exec98424, original
start1792680038289134991ns. Driver SHA256
`3d4465d5a586fc79113d03ce78b84dfa291a0027d035150c68ebf2f9a7b4ed0b`;
protocol `199f2991ce01ec454e6c9c849cf144caef01b182e5aabb681da89bed6b9e8d17`.
24 focused checks pass in10.11s; staging6.07s, no candidate observations.

Shared sigma .06438251354725585 in score units. Centered Gaussian covariance
is sigma²(I−11ᵀ/3); deterministic quadrature gives .100000006 at order40 and
.099999931 at64. Parent independently sampled4,096 three-dimensional Gaussian
vectors with antithetic pairing: mean disagreement .099576, standard error
.000821. Independent stage/pin/seed/warm-start review passed in .456s, zero
candidate observations. All7,264 new IDs are distinct from9,000 prior reserved
IDs;864 environment starts also pass canonical physics-seed checks.

Inherited training279,449 transitions:258,405 used,21,044 unused. The paired
trial's550 completed journals were losslessly compressed1.656GB→.233GB in15.79s
with full decompressed byte verification; metadata, heads and required T remain
plain. Restore journals before further raw inspection.

## Alternative explanations checked during execution

Astra's mechanics-only diagnostic tested a 10,001-point grid of the initial
leftward serve family against the actuator's extreme-position envelope, including
wall reflections and crossing ticks. No tested target lay outside that envelope,
even after a five-tick (41.7ms) initial delay; minimum collision-reach margin .025.
This cost .050s and supplied no camera observations, policy execution or teacher
labels. It weakens a gross reach-limit explanation, but is only a necessary
reach envelope on a grid: it is not a constructive visual controller, an analysis
of opponent-return serves, or proof of achievable learning. Sol independently
agreed with that limited interpretation. Current training remains unchanged.

Separately, code inspection distinguishes spatial precision from missing feature
interactions. Current pooled relations already provide some interactions; a
higher-resolution spatial DCT would not automatically add arbitrary relationships
between channels or objects. [Konidaris et al. (2011)](https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf),
variable-coupling and limitations sections, informed this distinction. Its
Fourier-of-state construction differs from our spatial occupancy transform; no
benchmark performance is transferred. A capacity intervention remains unallocated.

## Performance finding prepared for the next cycle

During training, Astra profiled an exact replay of50 already-recorded first-actor
cameras; all returned records matched. No fresh candidate experience was added.
The profiled2.581s included150 fingerprint calls (three/camera), totaling1.677s,
about65% of wall time. Profiling overhead and concurrent work limit extrapolation;
nested function times must not be summed. This motivates a separate proposal for
research execution with less redundant checking, followed by actual equivalence
and throughput measurement. It does not authorize silently dropping integrity
checks or changing the live allocation. Sol will independently assess a bounded
proposal after both current outcome checks, preserving the full reference path.

An older completed confirmation trial's512 journals were losslessly compressed
1.453GB→.150GB in22.54s during training, preserving decompressed byte identity.
That maintenance overlap is part of actual elapsed training time, not excluded
from the trial or attributed to an algorithmic speed gain.

## Completed result: no established incremental learning

Both arms completed 112 joint batches. Fixed final checkpoints, no best-checkpoint
selection: carried start first returns31/64, IID32/64, correlated33/64. Primary
correlated lift+3.125pp,95%CI[-3.125,+9.375]; correlated-minus-IID+1.5625pp,
CI[-3.125,+7.8125]. Neither learning, temporal-coherence advantage nor absolute
80% competence was established. All returns49/113,50/113,62/124 respectively:
correlated retention passes, improvement remains unproven; IID retention is
inconclusive. At least two returns13/13/16 and three returns3/3/7 are descriptive.

Training79,641 transitions: IID39,920 (39,666 used,254 partial unused),
correlated39,721 all used. Evaluation48,063 transitions;1,090 initial images.
Elapsed1,932.81s:1,469.43 training and463.38 evaluation. Sol's source/update/noise
check passed8.829s and Astra's independent outcome check passed3.255s, neither
adding observations. Source freeze released; source integrated as8c0fed5.

The intervention worked mechanically: measured bias lag correlation approximately
0 versus.924; comparable10.5% disagreement with unbiased greedy actions.
Discordant same-start pairs28/224 versus37/224 and nonzero updates24 versus34
show modestly increased contrast, not useful learning. No update clipped. Net
movement relative to root-summed-squared steps was about1.05 in both arms,
consistent descriptively with cancellation; this is not a formal gradient test.

A small cache-flip count does not mean unchanged behavior:34/64 and35/64 fresh
games changed actions before first contact, but only3 and4 first outcomes changed.
Astra checked timing: median lead of first action divergence before contact was
1.025s and.767s across changed games. Thus uniformly late changes cannot explain
the null. This retrospective check is descriptive, not a causal experiment.

Next: bounded research-execution optimization, then independently challenged
visual-state baseline design. No automatic noise/rate sweep. A lagged baseline
would change credit variance, not rewards or action selection; it remains an
unallocated hypothesis until protocol review. Nonlinear residual capacity and
ordinary-serve curriculum remain alternative explanations, not implemented changes.

External evidence and check artifacts: temporal trial root above;
`/workspace/m1b-experiments/pong-mechanism-review-20261022/temporal-parent-result-check.json`,
`temporal-action-lead.json`; Sol interpretation under
`/workspace/m1b-experiments/pong-credit-state-research-20261022/after-temporal-actor.md`.
