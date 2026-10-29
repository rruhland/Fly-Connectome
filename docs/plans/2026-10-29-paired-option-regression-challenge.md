# One conditional fallback: directly learned paired option advantage

Implementation authorized after corrected Ogma regressed on the fixed32screen.
No native acquisition or fit has started. Parent reviews the pinned48-site spec
before the one180-second acquisition allocation.

**Recommendation:** if corrected Ogma fails, run one direct paired option-advantage
regressor before a learned world model. This changes credit and generalization
together for a specific reason: actual matched options have already rescued two
misses and produced two deployed later gains, while fixed scene-distance retrieval
transferred those actions incorrectly. Train reward-sensitive partitions directly
on observed intervention effects. Do not propagate a primitive Bellman target,
clone successful action labels, maximize a lucky return, or recursively predict
anonymous visual futures. A world model adds transition prediction, rare-reward
prediction, rollout-error control and a useful planning horizon before it can
address the same action decision.

**One prototype.** Reuse the frozen 40069 baseline and its normal camera cadence.
Collect 48 sites from 48 distinct admitted TRAIN starts, fixed before branch
outcomes: 16 later-miss anchors, 16 later-hit anchors, and 16 episode-uniform
post-hit camera positions, with equal phase counts within each group. Use the
existing 64-camera-before-outcome rule for the first two groups; the uniform
group must ignore future success/failure. Require every site to follow a recorded
own hit and have prefix length at most 512. If the exact allocation is unavailable,
report the shortfall rather than changing selection after seeing rewards.

At each site execute ALL FOUR matched options: baseline, or each available command
held 16 cameras, followed by baseline. Charge one exact shared prefix and every
actual cloned suffix. Define the target explicitly as **an own hit within the
next 160 cameras**. A hit gives 1, natural termination without a hit gives 0,
and a fully observed 160-camera window without a hit gives 0 window reward.
That last event is not labeled a natural miss or terminal. Actual environment
timeouts or deadline/quota partials before the full window remain censored.
Retain complete action-balanced sites; do not silently fill missing alternatives.
Learn the three paired differences `window_hit(option)-window_hit(baseline)`.
This excludes improvements that merely reach an already successful hit earlier.
The uniform post-hit sites are essential: outcome-aligned anchors alone omit
outgoing states where the deployed controller will also query its model.

Use the existing six-field `canonical_scene` rows, generic horizontal-first content order,
padding/presence indicators and entity count. Sort original six-field rows by
fields `[1,0,2,3,4,5]`, because position is `[y,x]`; preserve coordinate meanings. Fix the padding capacity from the
already saved 2,048 unlabeled calibration scenes before new rewards; an overflow
is explicitly unsupported. Add the three frozen baseline scores as legal compact
context from the existing full visual transport, never as targets or expert
labels. This exposes whole-scene coordinates rather than mean-pooled low-rank RFF
features or a hand-weighted scene distance. Padding/sorting are generic and use
no entity ID, object role, seed, physics state, or geometry-derived action target.
Canonical ordering can still change discontinuously when tracks appear/cross;
this is a limitation, not a claim of a perfect scene representation.

Fit ONE multi-output `ExtraTreesRegressor`: 64 trees, squared error, depth 5,
minimum leaf 2 distinct sites, all features considered, no bootstrap, one fixed
RNG seed, one thread. The three outputs share partitions learned from all paired
positive/negative/zero effects. No hyperparameter search or fitted-action target.
The estimator chooses among random feature thresholds by target variance
reduction and averages leaf means; these splits use reward evidence, unlike the
earlier fixed random 73-leaf geometry. Checked implementation documentation:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
(`ExtraTreesRegressor` docstring). This is methodological support, not a Pong
performance claim.

Deployment retains the already explicit past-own-hit context: baseline until an
actual scalar hit reward, then choose the greatest predicted positive paired
advantage, or baseline if none is positive/representation unsupported. Baseline
is reconsidered every camera; earned commands actually hold 16 cameras. Run the
entire learned policy from reset and audit its chronological hold/reward history.
Training tails use baseline while deployed tails may intervene again, so predicted
advantage is not an unbiased value for the deployed policy. Full-game outcomes
must resolve that approximation; no policy-improvement theorem applies here.

**Envelope and stopping.** At most 15 minutes implementation, then one 180-second,
60,000-camera acquisition allocation including every prefix. The stated geometry
gives an absolute design maximum of 48×(512+4×160)=55,296 transition cameras plus
48 initials. One fit, then a separate 180-second full-game evaluation: same fixed
8 TRAIN starts, conditional 32 TRAIN, conditional 32 genuinely unused paired
fresh starts. A short acquisition with fewer than32 complete representation-supported,
action-balanced distinct sites, any missing one of the six predeclared strata,
or entirely zero paired targets is acquisition failure. Any nonzero paired target
is sufficient for the single fit/screen; no minimum rescue-count gate applies.
Acquisition interleaves all six strata. Retain all complete supported sites with
fixed1/6totalweight perstratum, uniformwithinstratum; this is ordinary uniform
weighting if all48 complete. Report rescues and lost hits separately. Stop on screen8 total/later regression or no total
gain; retain first-return parity. On screen32 require total/later gains and no
second-return/fraction loss before fresh. Do not promote on the small screen.

**Strongest objection / abandonment rule.** Forty-eight sites are still only 48
independent scene-level examples, not 192 independent counterfactual states.
The forest can fit a few rare rescues while lacking conditional counterfactual
coverage on new trajectories. Nonnegative-looking predictions are not evidence
of safe support, and repeated intervention shifts the tail policy. Abandon this
declared prototype if its actual complete 8/32-game policy fails those gates, or
if useful TRAIN gains reverse on untouched fresh games. Do not answer a negative
with depth, duration, threshold, temperature, or representation sweeps. A positive
can justify another explicitly bounded online acquisition cycle on the deployed
policy's states; it does not yet establish continuous adaptation or the user's
95% first/later-return milestone with 200+ later opportunities and replication.


Before-outcome implementation details: capacity4 comes from all2048 saved
calibration scenes; the feature vector has32coordinates. All branch scenes are
saved as exact float64 N+1 anonymous rows plus N+1 baseline scores, compact actual
actions/hits/elapsed/ticks/terminal/truncated flags, and explicit censor reasons.
This costs no extra cameras and enables separate future transition-learning
research. Acquisition retains every complete, failed, zero and censored trial;
only complete action-balanced supported sites fit this forest. Overflow remains
unsupported at deployment and is reported. Vectorized exported forest routing
uses sklearn float32 inputs and exact sequential float64 tree accumulation;
training and2048calibration predictions/actions must match sklearn exactly.
Representation support only means capacity fits, never proven action coverage.
