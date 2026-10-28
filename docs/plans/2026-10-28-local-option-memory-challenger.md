# Independent challenger: local option memory after the tree pilot

Original research note followed by an authorized completed pilot. The latest user
mandate already waives further architecture approval. The separate tree pilot
failed its actual eight-game screen, 23→1 returns, before this pilot began.

## Completed pilot: rejected, no sweep follows

Parent approved one implementation and the prepared native spec
`f3cf81f06fdd43dc89c9c7c47b1b2a6d345309a926b7c644375bba91eac718d2`.
Runtime uses the exact existing SIX canonical entity fields, with no age or
camera-context addition. The assignment metric alone uses the fixed median
16-camera calibration radius 0.0039341945374085875, obtained from 256 predeclared
pairs in the 2,048 saved continuous-calibration scenes. Baseline actions are
reconsidered every camera; earned commands actually hold 16 cameras.

Artifact root: `/workspace/m1b-experiments/pong-local-option-memory-20261028-r1`.
Eight distinct TRAIN sites, four later misses and four later successes, were
available as specified. All 32 actual option branches completed. Acquisition
cost 4,984 transition cameras + 8 initials, comprising 2,904 once-per-site exact
prefix cameras and 2,080 actual cloned suffix cameras, in 18.413 seconds. The
24 completed paired advantages include TWO genuine miss→hit rescues, TWO
hit→miss losses, and 20 zeros. No positive came only from reaching an already
successful hit earlier. Both negative outcomes remain in the learned memory.
Memory SHA: `b5c5c8bcb73318c6ab83b07b70b065820524ad359624fd2f907c974f24986fb4`.

Immediate frozen full-game screen8: **23→19 total returns, 7→5 first returns,
16→14 later returns**. Later fraction 16/21→14/17 rises because the candidate
loses opening coverage and changes exposure; this is not a sustained-return gain.
Eight complete games cost 4,207 transition cameras + 8 initials, 17.164 seconds.
Nineteen earned options produced 285 held continuation cameras and 267 changed
primitive actions. Sequential replay of all 4,207 actual pre-action scenes, raw
features, baseline scores, choices and hold histories passed exactly at zero
additional camera cost. No raw-only certificate reuse occurred. Source, assets,
and memory stayed frozen. No screen32 or fresh work was launched and no model
was promoted. Combined cost: **9,191 transition cameras + 16 initials**, 35.576
seconds across the separately bounded acquisition/evaluation clocks.

The strongest observed failure is inappropriate transfer into openings: both
losing games first changed before their first hit (one loses 5→0 returns, the
other 1→0), and both retrieve the same single supporting site from a later miss.
Two games gain one return each after their first hit. Thus the mechanism acquired
useful actual reward evidence but the declared neighborhood/deployment rule did
not transfer it safely. Do not respond with a radius/duration/guard sweep.

The memory was assembled once from matched own-reward trials; continuously
adapting the deployed policy was NOT tested. The artifact field named
`online_acquisition_updates` records paired learning evidence, not live policy
updates; `acquisition-audit.json` and `final-audit.json` state this limitation.
The user's 95% first/later-return milestone remains unmet.

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
