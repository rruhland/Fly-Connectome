# Proposed next scaffold: fixed nonlinear residual policy

**Status: user explicitly approved the nonlinear residual test on2026-10-22.**
Implementation and a frozen prelaunch review are authorized; no launch yet.
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
