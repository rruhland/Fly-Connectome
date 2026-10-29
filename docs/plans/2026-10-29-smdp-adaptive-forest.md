# Dense adaptive option-value forest: one bounded offline prototype

Use real temporal options to increase the effect of an action, and reward-trained
tree partitions to select relevant coordinates. This is a different estimator and
credit assignment mechanism from the old 73 fixed leaves, the 48-context direct
paired-advantage forest, and the geometric transition graph. None establishes
that this learner will improve Pong.

Parent review is required after inventory/spec and before the sole fit sequence.
No native games are authorized by this plan. Implementation is confined to new
`pong_smdp_adaptive_forest.py`, `run_smdp_adaptive_forest.py` and focused tests.

## Data and actual options

Reuse the 60 continuous-actor TRAIN traces and 192 dense paired-option suffixes.
There are 43,271 source cameras from 75 distinct TRAIN starts. Saved prefixes
without compact scenes are excluded. No fresh data or new acquisition is used.

Options 0/1/2 execute the actual corresponding constant command until the first
own hit, natural termination, or 16 cameras. A changing-action sequence is never
relabelled as a held command. A shorter window without a hit or natural end is
unsupported. Option 3 is one camera of frozen baseline 40069, and is admitted only
when the actual action equals its actual baseline choice at that state. Its label
is an observed outcome, never an expert action target. A nonterminal boundary
needs its real successor; a verified natural terminal needs no successor.

Inventory before fitting:

| Option | Actual windows | Distinct starts | Eligible source rows | Known MC labels |
| --- | ---: | ---: | ---: | ---: |
| Held command 0 | 1,750 | 70 | 1,734 | 1,366 |
| Held command 1 | 1,453 | 70 | 1,453 | 1,436 |
| Held command 2 | 957 | 71 | 957 | 930 |
| Baseline one camera | 40,695 | 75 | 40,555 | 37,383 |

Reject 38,990 changing-action starts, 119 short censored macros, two macros with
missing successors, and eight baseline steps with missing successors. Empty or
overflow source scenes remain recorded but do not train the forest: 156 windows,
including 15 overlapping command-0 windows whose future boundary is a hit. This
does not mean 15 distinct actual hits were lost. No natural-end window is lost.
One eligible baseline continuation has an unsupported visual successor.

All windows remain in the 739,856-byte prepared dataset. Warm-start unknown tails
are omitted from the MC fit only: 368/17/27/3,172 eligible rows by option. They
remain eligible for actual-transition FQI. A prior fully observed 160-camera
window without a hit is not a natural failure for this next-hit target.

Natural row fractions are 3.90%/3.24%/2.13%/90.73%. The pinned weighting gives each
represented option 25% total mass, then each distinct start equal mass within an
option, then windows uniform mass within that start-option group. This prevents
baseline-step counts and repeated windows from determining the entire fit, but
does not turn correlated windows into independent experiments.

## State, credit and estimator

State is the same anonymous scene information: up to four horizontally sorted
six-field rows, presence flags, past observed own-hit context, and five-way public
camera-phase onehot. Four continuous fields use the frozen calibration scales;
there are 34 state coordinates. The four options use four onehot coordinates,
avoiding ordinal action semantics. Empty/overflow scenes fall back to baseline.
There are no object roles, entity IDs, seed inputs, policy-version inputs, baseline
scores, geometry targets, expert labels, or gradients.

The common physical target is expected `2**(-time_to_next_observed_hit/3.2)`.
Discount THROUGH the hit camera interval for every option. Thus a macro ending
with a hit uses `beta=2**(-sum_actual_dt/3.2)` and zero bootstrap; a natural end
without a hit uses zero; a completed nonterminal option uses `beta*max Q(next)`.
Grouping the same constant-command trajectory into one macro or baseline-sized
steps therefore gives the same reward timing. A hit is an absorbing boundary for
this prediction, not an environment termination. Runtime command holds end early
on that same actual hit or natural termination.

Initialize one forest from actual observed next-hit/natural-end MC returns.
These are BEHAVIOR-tail labels, not unbiased values of the deployed policy;
unknown censored tails stay unknown. This supplies long-delay own-reward credit
without hundreds of cold one-step iterations. Then perform up to 32 genuine FQI
refits from actual option transitions, using the updated critic's supported
maximum. A baseline-only chain still advances one camera per refit, so residual
distant warm-start bias remains possible.

Use one shared scalar `ExtraTreesRegressor`, 32 trees, minimum leaf four rows,
uncapped depth, all coordinates considered, no bootstrap, squared error, one fixed
seed 2026102903, one thread. Splits are recomputed from current reward-value
targets at each fit. Support means at least one actual queried-option training
row appears in the queried leaf in a strict majority, 17/32 trees. Distinct-start
counts are diagnostic only; leaf size four is not independent evidence. This
light support gate prevents completely absent option support from being treated
as measured value, but gives no counterfactual confidence guarantee.

## Budget, exports and scientific limits

One 180-second allocation includes loading, one MC fit, up to 32 FQI refits,
diagnostics, export, serialization and parity checks. Reserve eight seconds for
finalization; all time remains charged. Keep the last fully completed iteration,
never the best native/training outcome. Save initial MC and final endpoints only.
At least one completed FQI update is required for an SMDP-FQI claim; otherwise
report MC-only. A clock receipt prevents automatic restart. No sweep or new native
work follows without parent review.

Training predictions and leaf queries use sklearn C `predict`/`apply`, including
chunked expanded successor-option inputs. Portable NumPy tree traversal is used
only for final parity and runtime. Validate exact scores on every training input,
all four options on 512 saved states, and serialized query parity. Export all
values and actual option leaf evidence with read-only arrays. New artifacts must
remain below 50 MB. Runtime preserves baseline opening until an actual hit,
requires baseline and selected-option support and a strict predicted gain, and
uses the declared early-ending command holds.

The strongest objection is remaining conditional coverage and state aliasing:
75 starts and overlapping windows do not identify every counterfactual, and a
reward-adaptive forest can generalize a wrong advantage. MC initialization also
mixes behavior tails, and 32 updates do not guarantee removal of that bias.
Actual complete-game comparisons must decide usefulness.

[Ernst, Geurts and Wehenkel (2005), sections 5.3.5 and 5.3.8](https://jmlr.org/papers/volume6/ernst05a/ernst05a.pdf)
support the methodological distinction: output-adaptive partitions resisted added
irrelevant variables better than fixed kernels in their control problems, and
both model building and target prediction cost time. Their experiments do not
establish Pong performance or convergence of this changing-partition estimator.

## Completed offline sequence

Parent approved final spec
`7c4cd62eea3a7d0d0a0d9060b44405cc7d9045ea8afe662ab89960b52f8e221c`.
Implementation and tests were committed as `3d156b9` before launch, and their
source hashes stayed unchanged. Seven focused tests passed.

The one sequence completed in 168.313 seconds of its original 180-second clock,
including export and parity checks. Initial MC fitting took 4.667 seconds, using
41,115 known observed-return labels and omitting 3,584 unknown tails. Twenty-six
FQI updates completed. Iteration 26 is the final endpoint because another complete
update would not fit the remaining allocation; no outcome was used for selection.
This is not a convergence claim.

Exact exported predictions matched sklearn on all 44,699 training inputs. The
serialized model also matched all four option values/support on 512 saved states,
2,048 queries. The final forest has 551,958 nodes. All new artifacts occupy
15.48 MB, below the 50 MB bound. No native cameras were used.

Artifacts are under
`/workspace/m1b-experiments/pong-smdp-adaptive-forest-20261029-r1`:

- Final `final-forest.npz`: SHA
  `ca99b349d1ea26a2036214e37d04e58a2d39a59cf26b44d5aeed7e18484df295`.
- Initial `initial-mc.npz`: SHA
  `8eb52c1d2a54a7aed255b143de85e64f85426a938bf0ca57bf7b004a9c1d8c85`.
- `fit-summary.json`, `iterations.json`, `fit-clock.json`, `inventory.json`,
  `data-manifest.json` and the pinned `spec.json` preserve the sequence and data.

Diagnostic support on the 512 saved audit states is 65.0%/63.1%/76.2%/99.8% by
option. These are saved-state support statistics, not confidence bounds or native
performance. The final API and hash were sent to the evaluation worker for parent
review before any complete-game comparison.
