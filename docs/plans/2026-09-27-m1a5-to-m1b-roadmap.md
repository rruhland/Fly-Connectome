# Current Milestone 1 direction: efficient continual visual control

This supersedes the original architectural hypotheses. On 2026-09-27 the user
approved the hybrid probabilistic visual candidate for production and prioritized:
fast online learning, transferable experience with continued plasticity, and low
compute. Biology/connectome structure is a useful hypothesis, not a veto on a
working engineered component. No backpropagation is introduced by this promotion.

**Compute policy update, 2026-09-27:** the user clarified that compute/sparsity
targets are measurements and optimization priorities, not hard vetoes on useful
learning. Do not discard a stronger learner solely for exceeding a latency or
memory target. Preserve its result, report its cost, and investigate equivalent
lower-cost execution later. Finite experiment runs prevent aimless investigation;
they are not architectural compute ceilings. This clarification supersedes any
hard-budget wording below.

## Decision

**Keep the promoted visual state and its existing predictive learner. Start the
smallest M1B learning loop after a short continual-learning/interface gate.**
Do not neuralize vision, add a second predictive head, or start another large
latent-world-model project now. The current history/outcome associations already
learn a distribution of future displacements. The association state also already
has temporal memory. Neither is a general causal, action-conditioned world model.

New recurrence must earn its cost by improving a defined failure: long hidden
history, action consequences or delayed credit. Replacing a working algorithm with
neurons is not itself progress toward the user's priorities.

## What is and is not achieved

The production baseline separates actual observations, carried entity hypotheses,
learned contextual corrections and spatial future mixtures. It learns context and
dynamics from observed endpoints online, with bounded memory and no backprop.
Its local credibility observer has pretrained weights; `step(learn=True)` does
not retrain that observer. Runtime inputs include every-sample grayscale images
and events. Engineering supplies detection/association and coordinate transforms.

Held-out synthetic transfer, calibrated marginal intervals and passing memory
tests are real results. They are **not yet continual learning across games**.
The 256-example context FIFO can forget; uniform dynamics reservoir sampling can
adapt too slowly after a shift. Four observed displacements can alias physically
different futures. Independent per-entity forecasts do not model interactions.
The model is not action-conditioned. A mixture mean is not a safe substitute for
multiple possible futures. Unfitted reidentification remains a .5 prior, explicitly
accompanied by zero calibration evidence.

The previous point/event-only failures remain documented limitations, not current
acceptance requirements. Sparse central encoding does not make the dense image
front end or exhaustive memory retrieval event-driven.

## Remaining M1A.5: two bounded deliverables

**2026-09-27 execution status:** generic interface implemented; continual audit
finished with useful shared-domain benefit but one failed changed-dynamics retention
seed and censored shared-domain acquisition speed. See
[handoff findings](../experiments/2026-09-27-m1a5-handoff-findings.md).
The continual upgrade is not yet accepted; preserve the promoted baseline while
the deeper predictive-memory question is reassessed.

### 1. Continual transfer/retention audit before adding architecture

Implement a generic camera-stream A -> B -> A benchmark with three arms: the
promoted continuously learning model, its frozen copy, and a fresh learner at each
switch. Keep visual state/weights through switches; reset only transient scene
state when the environment actually resets. No game identifier or hidden object
label enters the learner. Keep scoring probes frozen so probe retraining cannot
masquerade as transfer. Use five seeds, initially 2,000 camera samples per segment;
this is a finite budget, not a convergence claim.

Separate two changes: new appearance/layout with shared dynamics (forward transfer),
then changed motion rules with familiar appearance (adaptation/interference).
Test original A before and after B, with no extra A retraining for the first return
measurement. Record samples and wall-clock time to a criterion fixed from separate
development data, prequential NLL, calibration, binding errors and retained A skill.
Do not select thresholds using the final A/B streams.

Proposed promotion gates for a *continual upgrade*: at least 20% fewer observations
to the same held-out B criterion than fresh learning on shared structure; return-A
proper-score degradation no more than .2 nat and binding reduction no more than
five percentage points; retain the existing per-family/coverage gates. Report
paired seed results rather than one lucky trace. Positive backward transfer means
improvement on A after B; simply retaining A is not that improvement.

If the current memory fails, compare **one** recent+stable memory design with the
current FIFO/reservoir: fast recent associations adapt, a bounded long-term bank
preserves experience, and surprise/local context controls retrieval. Credit and
admission use real observations, not invented rollouts or game names. Do not run
a long bank-size/decay sweep. If failure is due to indistinguishable histories,
test added context rather than pretending better memory replacement can solve it.

### 2. A compact, generic central interface

Keep the structured output as the source of truth. Add an adapter, not a neural
replacement. Supply observed position, recent observed displacement, observation
age/status, support strength, contextual information and predictive distributions
at declared horizons. Also preserve coarse current visual context so large static
surfaces excluded by compact-region extraction are not silently lost. No ball,
paddle, wall, intercept, ownership or collision labels.

Normalize coordinates with explicit aspect-preserving geometry and include sensor
cadence. Do not silently reuse frame-horizon calibration at a new physical cadence.
Track IDs are routing handles, never semantic feature values. Use shared entity
features, permutation-invariant pooling and bounded nearest-neighbor relations;
do not give each persistent slot an unrelated learned policy meaning.

For efficient decision learning, provide sparse feature indices/values alongside
the structured state. A population code is a transport encoding, not the final
learned representation and not a requirement to simulate spikes. Preserve distinct
future modes plus tail mass/uncertainty, not just mean velocity. Start with full
32-component mixtures in the correctness control; accept a compressed code only
after a bounded fidelity/control-utility comparison. Expose overflow/missingness.
Do not cap away inconvenient entities without reporting it.

Test no oracle data, invariance to ID relabeling and entity ordering, episode/reset
handling, and missing-evidence semantics. Measure encoder cost and active feature
counts. Initial end-to-end target: 50 decisions/s on the reference CPU at 64x64,
including **online updates**; report p50/p95 latency, acquisition/I/O separately,
and results for two versus eight visible entities. Treat 20 ms p95 as a target,
not an achieved guarantee. Roughly 5 ms for the controller is an initial engineering
allocation, not a reason to reject better learning.

These two deliverables close the immediate M1A.5 handoff. They need not establish
general vision or solve all new dynamics before useful M1B experiments begin.

## M1B: learn actions directly from the useful state

### Stage B1: minimal online control with an honest baseline

Define an environment boundary: camera observation, legal action set/actuator
description, reward, termination and elapsed time. Only camera information reaches
vision. The controller gets the encoded visual state, previous action and reward.
Environment ground truth belongs only in scoring. Actions may differ across games;
do not assume numeric action ID 1 means the same behavior in every environment.
Allow actuator-specific output heads while sharing perception/associations; report
which parameters transfer and which are new. Discover controllability from observed
action consequences instead of labeling the controlled entity.

Start with **on-policy sparse linear action values and true-online Sarsa(lambda)**,
using exploration and eligibility traces. This establishes whether the state is
actionable under reward with very cheap updates. TD error times eligibility updates
local active weights; there is no BPTT or end-to-end gradient through vision.
The eligibility trace handles delayed reward; it is not a learned world model.
Implement the actual Dutch trace and correction terms of true-online Sarsa;
plain TD-error-times-eligibility alone is not that algorithm. Verify against its
forward-view reference on small deterministic trajectories before game learning.
Mask terminal bootstrapping, distinguish time-limit truncation, and scale discount
with physical elapsed time. Specify reward/exploration schedules before comparison.

First debug on a small generic visual control task, then evaluate Pong and a
substantially different visual interaction task. Keep environment score/reward in
the environment adapter; do not restore the legacy hidden paddle-alignment reward
as a generic visual teaching signal. Use random/no-op and an allowed fixed-control
diagnostic, plus state-only versus state+forecasts ablation. The control learner must
find which predictions matter; M1A.5 does not hardcode a future intercept.

Freeze vision initially to isolate policy credit; next enable its existing online
updates and test drift/retention. The final continual system must pass with learning
enabled. Measure reward versus real observations and wall-clock seconds, not only
simulator frames or final score. A policy failure does not automatically mean that
vision needs a recurrent neural overhaul.

### Stage B2: let the connectome earn a central role

Compare three central representations with identical input encoding and reward
learning: direct sparse state, a small fixed/homeostatic recurrent connectome
subgraph, and a size/degree/sign-matched randomized recurrent control. Use existing
measured central/motor pathways as the starting roster; choose the bounded subset
by documented anatomy and compute budget before seeing reward results. Do not
execute the entire slow optic graph simply to justify its inclusion.

Inject sparse/graded feature activity through an explicit adapter. Preserve internal
measured connectivity initially; input projection and readouts are declared engineered
interfaces. Fixed recurrent dynamics first, plastic readouts/associations first.
Avoid changing vision, internal recurrence and reward rule simultaneously. If spikes
buy no measurable advantage, graded dynamics are an allowed alternative.

Adopt the connectome arm only if it produces a material reproducible gain: proposed
screen is >=20% fewer observations to matched competence, or clear retention/delayed-
credit improvement. Report the combined runtime rather than rejecting a promising
arm solely for exceeding the initial compute target. Compare both equal
observation budgets and equal wall-clock budgets, with matched readout capacity and
five seeds. If it fails, keep it optional and keep the successful direct controller.
An anatomical prior is a hypothesis, not a guarantee of useful dynamics.

### Stage B3: add only the missing capability revealed by B1/B2

- If choices require predicting consequences of actions, learn a small
  **action-conditioned transition/affordance association** on observed state,
  action, next observation. Model-free action learning comes first; the current
  passive forecaster cannot serve as a counterfactual action simulator.
- If identical recent inputs require different actions because of long hidden
  history, test a bounded recurrent central state against the direct control.
- If current decisions fail because uncertainty is discarded, improve the belief
  interface/risk-sensitive readout before replacing the visual learner.
- Add local reward-modulated recurrent plasticity only if frozen recurrence helps
  and adaptation to new temporal structure remains the limiting factor.

Any learned rollout must retain multimodality and uncertainty; its guesses are not
ground-truth visual targets. A large joint recurrent latent world model is the last
option, justified by a concrete partial-observability/action-planning failure.

## Practical interpretation of “concepts” and transfer

Measure reusable skills: motion/continuity, grouping through missing evidence,
relative spatial relations, controllability and reward-relevant associations. Do
not claim semantic understanding because a trajectory is decodable. Use held-out
games/layouts with shared learned components, fresh-policy and fresh-vision controls,
and A -> B -> A revisits. Observe both fast acquisition and retained competence.
No method here currently guarantees lifetime plasticity or backward transfer.

## Research grounding, not proof for this implementation

True-online TD work supports an efficient online eligibility-trace baseline with
linear function approximation; it does not guarantee cross-game generalization:
[van Seijen et al., JMLR 2016](https://www.jmlr.org/papers/v17/15-599.html).

Connectome-constrained fly vision obtained useful functional predictions with task
optimization as well as anatomy. This supports testing anatomy as an inductive
bias, not expecting the graph alone to solve central learning:
[Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3).

Continual-learning research distinguishes retaining old performance from retaining
the ability to acquire new skills. Its deep-network results do not prove this
episodic learner is immune, so the A/B/A and adaptation measurements are essential:
[Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7).
