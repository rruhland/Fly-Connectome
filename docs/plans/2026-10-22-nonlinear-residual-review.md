# Proposed next scaffold: fixed nonlinear residual policy

**Status: user explicitly approved the nonlinear residual test on2026-10-22.**
Completed: both independent checks pass; neither endpoint establishes learning or retention.
The user authorized ongoing experimentation but reserved genuinely major architecture
alterations for review. Astra and Sol independently recommend a capacity intervention
after the [checked visual-baseline result](../reconstruction/2026-10-22-visual-baseline-research.md).
This is a change to the learnable policy function class, so that boundary applies.

## Concrete change for approval

Retain the last retention-qualified actor, temporal ARtheta112, and its existing
214-dimensional capped visual vector x. Add64 fixed generic nonlinear features:
h(x)=tanh(A(x/16)+c), with A drawn once from a standard normal distribution and c
once from uniform[-1,1], using a declared independent seed. Freeze A and c forever.
They use only the current legal visual features, never game state or teacher labels.
Each feature is bounded, so the residual branch has squared norm at most64.
Preserve the old x cap separately; never jointly rescale x and h.

Compute action scores as two explicit branches:

    q(x) = theta x + B h(x)
    initial theta = exact carried actor; initial B = zero

Zero B must reproduce original scores, conditional probabilities, actions and
trajectories exactly under matched RNGs before any learning. Compute the old
branch in its original order to avoid changing numerical behavior just by
concatenating a larger matrix. Preserve exact policy/checkpoint provenance.

Learn only theta and B using the existing analytic categorical score eligibility
and paired first-return outcome estimator. There is no differentiation through
A/c or vision and no backpropagation. Keep the original exploration and aggregate
function-RMS update cap; measure that cap on the combined score change using the
fixed, score-free generic visual cache. Drop the failed learned-baseline
intervention from this comparison. No new reward or privileged observations.

## Bounded test if approved

Compare residual versus unchanged linear actor from the same exact carried
checkpoint, with matched ordinary starts and identical paired estimator/noise.
One fixed64-feature map; no dimension, seed, bandwidth or learning-rate sweep.
Both get at most40k training transitions;20-minute training,10-minute evaluation,
30-minute overall budget. Last complete joint endpoints,64 fresh common full-game
starts for both endpoints and the exact carried model. Apply the same first-return
learning,80% absolute competence and all-return retention gates; require independent
replication before promotion. Runtime/evidence work counts toward the fixed clock.
The implementation/protocol still needs its usual prelaunch correctness review.

## Why this, and what it would not prove

The local actor, same-start pairing, temporally correlated exploration and learned
visual baseline have not established further first-return learning. The latest
baseline learned its prediction proxy but increased descriptive update dispersion.
This justifies challenging a different assumption; it does NOT prove a capacity
ceiling. Existing visual relations already supply interactions, and changing the
representation can also change conditioning and exploration, not only capacity.

The residual adds only192 learned output coefficients, but could still worsen
sample efficiency. It cannot restore information already lost by the visual
encoder or projection, nor fix missing memory or incorrect motor timing. An
inconclusive small trial would not establish representational impossibility.

Sol independently proposed the concrete64-feature tanh branch and exact warm-start
requirement. Astra independently researched explicit nonlinear features and reviewed
capacity/conditioning and preservation risks. [Rahimi and Recht (2007)](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf),
intro and section3/Algorithm1, supports the general idea of fixed nonlinear maps
with linear readouts; its Fourier map and supervised kernel results do not establish
properties of this tanh map or Pong learning. No performance guarantee is imported.

The smaller authorized alternative is an explicitly assisted ordinary initial-
leftward-serve curriculum with full-task retention checks. We recommend capacity
review first because another task restriction currently has weaker evidence and
would make the claim narrower. No experiment is launched simply to keep busy while
this architectural decision is pending.

## Approval and implementation boundary

The user answered the explicit architecture question: “Approve the nonlinear
residual test.” Reuse Sol for implementation; no repeated permission request.
Legacy greedy evaluation uses mapped raw-feature scores while the actor uses
capped projected x. Their linear greedy choices are equivalent under the positive
common cap, but reported score magnitudes differ. Nonlinear evaluation must use
the same capped x and separate score branches as sampling. Verify exact actor
scores/probabilities at B=0 and matched legacy greedy actions separately; do not
claim equality of differently scaled diagnostic score records. Implementation
choices remain subject to independent prelaunch correctness review.

Additional primary source: [Rahimi and Recht (2008)](https://pages.cs.wisc.edu/~brecht/papers/08.rah.rec.nips.pdf),
intro/Algorithm1/Theorem1, directly treats fixed bounded random nonlinearities
with learned output weights. Astra read those sections. Its independent supervised
sample/empirical-risk assumptions do not establish online RL learning, a suitable
64-feature size, or our estimator's sample efficiency. This is architectural
precedent, not evidence that the Pong intervention will work.

## Fixed-map check and interpretation limit before launch

Map seed1639497361; A sampled standard normal then c uniform[-1,1], artifact
`/workspace/m1b-experiments/pong-nonlinear-residual-20261022-r1/residual-map.npz`,
SHA256a4caf0613b4e8abdfbdfa0e41d2227ac208e5b167f118f3166fec918edab9423.
Astra's score-free1024-row cache check found all64 new feature directions outside
the existing linear span (residual singular values.1725–1.218), but only.7323% of
centered feature energy outside that span. Existing features plus intercept have
numerical rank183; residual feature squared norms13.98–30.97, saturation above.99
only.0153%. Cost45ms; no candidate observations, labels, policy fitting or map
selection. Artifact `pong-mechanism-review-20261022/residual-feature-span.json`.

Astra challenged whether to remove the fitted linear component. Sol recommended
retaining the exact approved map: subtracting and renormalizing a small remainder
would alter scaling, boundedness and extrapolation from a limited cache. Astra
agreed. No map, seed or scale changed. This is a test of one fixed feature
augmentation, including conditioning effects, not a clean isolation of nonlinear
capacity. A null cannot refute nonlinear capacity; a gain must not be attributed
solely to nonlinear expressivity. No extra diagnostic training branch is added.

## Implementation and independent prelaunch review accepted

Isolated source `/workspace/Fly-Connectome-m1b-pong-residual`, commit
38a097f53621aef6a452e834c19c8b03639accd9. Allocation root
`/workspace/m1b-experiments/pong-nonlinear-residual-20261022-r1`.
Driver7fccbbc2e2a640ce7c3e7870b12227c592e2a5975aa77dd8bf6224f7678ca3f4;
protocold61b228e901a023c7ba67e258e5c1bde6c75f1e8cab0225cf16e962d64227fb6;
prelaunch manifest1fb9877c7c5652507e3c1cdf019e43c523bf413506230853a639e3460524582f.
Sol44 focused core tests PASS32.16s and2 joint-driver tests PASS1.42s. Astra reviewed
actor/update/evaluation/recovery/worker-completion code and independently verified
artifact/source pins, exact warm start/B0, fixed map generation, constants and
7,264 new IDs excluded from23,540 prior IDs;864 canonical environment starts
unique. Parent stage check56ms, zero new observations.

Retained proof470 old cameras plus4 initial images, zero candidate observations:
12.518s. Actor q/p/actions/noise/E exactly match at B0; full-game greedy actions
and physics match legacy, excluding deliberately different score/policy-norm
magnitudes. Two full-game greedy cases total376 cameras,8.040s including export
and completion pin;10-minute evaluation budget is plausible but not guaranteed
under contention. All three evaluated policies use the same capped projected
score path. A/c arrays are embedded and content-validated on recovery; only T
remains a required external checkpoint dependency. No learned baseline is used.

Astra authorized launch after these checks. Source/driver/protocol freeze through
both outcome checks, no DEV peeking. Another482 completed old journals were
losslessly archived1.434GB to.203GB in15.26s, with decompressed byte verification,
before launch to reserve disk; zero candidate observations/replay.

Completed exec71017, original launch1792686199720929873ns. Both checks pass; source freeze released and source integrated as fb55f7d.


## Checked result and decision

111 complete joint batches; 79,443 training transitions (256 unused partial),
61,228 evaluation transitions and 1,082 initial images. Execution 1,165.712s,
including 749.976s training and 415.736s evaluation. No budget extension or recovery.

| Measure | Carried ARtheta112 | Linear control | Nonlinear residual |
|---|---:|---:|---:|
| First returns | 38/64 | 40/64 | 37/64 |
| All returns/opportunities | 79/140 | 89/149 | 78/140 |
| Games with at least two returns | 21 | 23 | 21 |
| Games with at least four returns | 5 | 6 | 5 |

Residual first-return lift is -1.5625pp, paired 95% CI [-7.8516,+4.6875];
linear lift +3.125pp, CI [-3.125,+9.375]. Residual versus linear -4.6875pp,
CI [-10.9375,+1.5625]. Neither learning nor 80% competence nor advantage passes.
All-return retention is inconclusive for both: residual lift -0.7143pp,
CI [-8.7173,+7.4803]; linear +3.3030pp, CI [-5.4208,+11.9517].
This does not establish harm. No endpoint promoted; retain ARtheta112.

Residual changes were active: combined net score RMS .01158 versus .00395 for
linear, 26 versus 6 generic-cache greedy flips, and 48/64 fresh residual trajectories
changed before first contact (median lead 1.133s). Only five first outcomes changed.
The branch did not simply fail to move the policy. Its raw features were largely
linearly redundant, so this null does not refute nonlinear capacity generally.
No automatic map, seed or rate sweep follows.

Sol's complete source/RNG/eligibility/update/recovery-pin check passed in25.888s;
Astra's independent raw-outcome, update and bootstrap reconstruction passed in4.647s.
Both added zero observations or replay. Parent artifacts:
`/workspace/m1b-experiments/pong-mechanism-review-20261022/residual-parent-result-check.json`
and `residual-result.png/svg`; executor evidence is the allocation's
`completion-verification.json`. Source is integrated; raw reads are finished.
