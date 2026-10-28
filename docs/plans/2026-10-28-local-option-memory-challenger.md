# Independent challenger: local option memory after the tree pilot

Research only. This worker changed only this note; no implementation, fitting,
native camera work, or promotion. The latest user mandate already waives further
architecture approval. The tree pilot remains a separate worker's experiment.

## Ranked hypotheses

1. **Local reward-trained option memory.** The failed globally fitted policies
   may combine individually useful repairs destructively. A small memory can
   retain local action alternatives without changing actions everywhere. Choosing
   an actual 16-camera option also tests temporal credit, rather than asking a
   primitive action to explain a reward 40–80 cameras later. This is a new
   deployed controller trained by its own outcomes, not a classifier of successful
   prefix actions.
2. **Learned temporal predictive hierarchy, specifically older OgmaNeo2.** Its
   learned sparse recurrent state and local next-state prediction could separate
   contexts that the current frozen representation aliases. This is more distinct
   than another reservoir/static feature sweep, but costs substantially more
   integration and needs a realistic acquisition budget.

Neither is proved by the historical failures. LSPI's 23→2 return collapse followed
72.55% unsupported greedy action choices and only 29 retained reward rows. Failed
actor fits show poor transfer and interference, not that all reward learning or
all current visual information is inadequate. The new epsilon collection improves
one-step action coverage but does not contain deliberate 16-camera alternatives
with matched outcomes. Conversely, hold10 SARSA previously failed; temporal
actuation alone is not a fresh explanation or guaranteed solution.

## One concrete next experiment: paired local option memory

**Purpose:** test whether reward-earned local alternatives can change actual
later play without destructive global fitting. Keep 40069 vision and its
primitive controller frozen. No BP, expert actions, simulator state, seed/time
input, or runtime simulator queries.

Use eight sites from eight distinct admitted TRAIN executions: four later misses
and four earned later returns, balanced by phase. Select deterministically from
the existing execution order before branch outcomes. A site is 64 cameras before
the next recorded own hit or natural miss, after at least one earlier own hit;
exclude a site if this would cross the previous hit. Existing outcome-based
selection is training curriculum only, never a runtime input. Including successful
sites permits negative as well as positive causal advantages.

At every site acquire the same four options: follow frozen 40069 for 16 cameras,
or hold each of the three available commands for 16 cameras. Continue afterward
with frozen 40069 until the first own hit, natural termination, or 160 cameras
from the site, whichever occurs first. Option execution stops at a natural end.
The fixed hold commands are proposed actions, not expert labels; retain all their
actual successes and failures. Score a completed first hit as
`G = 0.95 ** elapsed_seconds`, natural miss as zero. A 160-camera truncation is
censored, not a failure; exclude an incomplete paired comparison. This target
measures a local intervention followed by the stated baseline tail, not optimal
long-horizon Q. Reuse an exact existing baseline continuation certificate where
valid. Charge every newly replayed prefix and every branch camera.

For each completed site/option store the initial legal scene, executed option,
actual duration, outcome/time, baseline-paired `G(option)-G(baseline)`, source
execution/site identifier, action/reward receipts, and tail-policy hash. Identifiers
are provenance only. Update this memory after each completed own-reward trial;
do not keep only winners or use the maximum return ever observed.

### A legal, explicit scene neighborhood

Use the existing encoder's anonymous entity rows and coarse 8×8 camera context.
Discard entity IDs. A row contains normalized position, observed velocity,
observed/velocity-known flags, and clipped observation age. These are already
camera-derived fields in `vision/interface.py`; no new object-role detector or
physics target is needed. Normalize position by its existing image normalization,
clip to [-1,1], map each velocity component through `v/(1+abs(v))`, and clip age
to one second. Unknown velocity is zero with its known flag retained.

Compare two entity sets by minimum-cost assignment of their seven-coordinate
rows. Pad the smaller set with dummy rows; real/dummy cost is 1 and dummy/dummy
cost is 0. Real/real cost is mean squared coordinate difference divided by 4;
divide total assignment cost by the larger set size. Empty/empty cost is 0.
This is anonymous content matching, not persistent object correspondence.
Add equally weighted coarse-context squared distance, also divided by 4; use
an explicit missing-context flag. The final distance is the mean of these two
bounded distances. Preserve exact normalization/source hashes.

Fix the neighborhood radius before seeing new option rewards: use the median
distance between legal scenes 16 cameras apart in a fixed episode-uniform
256-pair sample of existing TRAIN observations. This is one geometry calibration,
not a reward-tuned radius sweep. Use all stored sites within that radius, weighted
uniformly. For each option average the *paired advantages* of its completed trials;
do not average only successful records. Choose the greatest strictly positive
mean advantage, ties using the existing action order. If no supported positive
alternative exists, execute the baseline option. Report effective sites and
distinct source executions behind every learned decision. Sparse support or zero
learned decisions is acquisition failure, not baseline performance credited to
learning. No radius, option-duration, or confidence-threshold retries.

At deployment the chosen option actually runs for 16 cameras, then retrieval
repeats from the new observed scene. Thus the controller is fully specified and
does not replay a seed- or time-indexed action script. The strongest objection is
that training used a baseline continuation while deployed retrieval can intervene
again. Effects can compound and invalidate the estimated advantage. Anonymous
scene distance can also match visually similar but dynamically different states.
These are reasons for immediate full-game deployment testing, not claims that
the memory estimates already solve them.

### Resource bound and decisive outcomes

Implementation allowance: at most 15 minutes, separately reported. One collection
clock of at most 120 seconds and 30,000 actual new cameras, including every prefix;
at most the eight sites and four options/site above. Stop with a partial record
if the bound arrives; never invent missing paired outcomes. Trusted TRAIN-state
cloning is permitted if existing APIs support it; cloned physics is never inspected
by the policy and runtime cloning/planning is prohibited. Use existing exact null
certificates to avoid redundant cameras.

Immediately freeze the acquired memory and run the same eight TRAIN full games
used to expose LSPI, then at most the fixed 32-game TRAIN screen if useful. Reject
clear total/later-return regression. If useful, evaluate 32 genuinely unused paired
fresh games against 40069. Charge evaluation separately; use one 180-second
evaluation clock, with every partial game identified. Screen gains justify further
online acquisition, not a reliable-control claim. The user's milestone remains
at least 95% first/later returns on fresh games with at least 200 later
opportunities plus independent replication, and evidence of actual online
adaptation. A small frozen-memory pilot cannot certify that goal.

## Outside evidence checked in this review

* Blundell et al., *Model-Free Episodic Control* (2016),
  https://arxiv.org/abs/1606.04460, is the conceptual precedent. Direct paper access
  was blocked by this environment's network proxy; do not claim a fresh paper
  reread. The checked independent implementation explicitly describes its
  random-projection route as excluding the VAE, and its code averages nearest
  values at query time but uses `max(old_return,new_return)` for an exact stored
  state: https://github.com/astier/model-free-episodic-control/blob/master/README.md
  and https://github.com/astier/model-free-episodic-control/blob/master/mfec/qec.py.
  The proposed paired-mean option memory intentionally changes this optimistic
  update. It inherits no MFEC performance guarantee and imports none of that code.
* OgmaNeo2 source pinned at
  `c5be9ec41d26aeeb2fc805961140781c851542cf`:
  https://github.com/ogmacorp/OgmaNeo2/blob/c5be9ec41d26aeeb2fc805961140781c851542cf/source/ogmaneo/Actor.cpp.
  `Actor::learn` forms `q + g*value`, updates a linear one-hot value readout directly,
  and applies direct signed TD policy deltas. `Predictor.cpp` likewise uses local
  one-hot updates against observed next-code targets. These inspected paths have
  no propagated hidden-layer gradient. The entire image-encoder and acquisition
  integration still need audit before claiming a legal complete controller.
* **Do not substitute the current AOgmaNeo successor without checking legality.**
  Its README calls it the preferred successor, but pinned
  `645a54ace656b0ac2476a56a0dac19faacbd87ab` `actor.cpp` explicitly differentiates
  softplus dendrites (`sigmoidf(act) // store derivative`, then error times that
  derivative before weight updates):
  https://github.com/ogmacorp/AOgmaNeo/blob/645a54ace656b0ac2476a56a0dac19faacbd87ab/source/aogmaneo/actor.cpp.
  Under a strict no-BP requirement this nonlinear actor is not an automatically
  admissible drop-in. Local hierarchy training and absence of autodiff do not
  erase this derivative path. The 2020 Pong blog is relevant prior reported
  evidence, but could not be freshly fetched here and does not establish our
  sample budget or sustained-return criterion.

Do not start another actor-feature, guard, temperature, native CEM/ARS, or
winner-only imitation sweep after a tree failure. The memory pilot above tests
a specific alternative: local, finite-duration actions with actual paired reward
evidence and a separately tested deployed retrieval policy.
