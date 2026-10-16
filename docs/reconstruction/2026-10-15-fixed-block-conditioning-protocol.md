# Fixed block conditioning: separate delayed-toy mechanism gate

This is a new development allocation, declared before its calibration or learning
data are generated. It preserves the failed original delayed pilot and the exact
unscaled Pong comparison. No final Pong seeds are used. The current Pong control
will finish before its pinned source is changed.

## Single intervention

Keep `visual-transport-v1` and its structured/raw sparse values unchanged. At the
learner and frozen-policy boundary only, multiply each block by a fixed positive
constant. Do not center, add features, change reward, adapt scales during learning,
or introduce game labels. Sparse zeros and missingness remain zero; raw counts
retain their meaning because each block receives one constant linear scale.

Fit scales from mean squared block norms on a separately seeded score-free camera
stream. Reward/side labels are never arguments to the fitting procedure. For
nonzero calibration blocks, set scale `clip(c / sqrt(mean_norm_squared), .125, 8)`;
blocks absent throughout calibration retain scale one. Solve scalar `c` by fixed
bisection so the sum of mean squared scaled block norms equals the original sum.
This controls total update energy while changing feature conditioning. Freeze and
hash the resulting mapping before training; all subsequent policy calls use it.

## Calibration and matched learning

Use the delayed visible-choice task and the same frozen prior/64x64/.02-second
camera path. Calibration: environment seed 482977133, independent uniform action
seed 143332057, exactly 64 balanced episodes = **512 transitions**. No learner is
constructed or updated. Accumulate block norms from actual pre-action observations,
including each episode's initial camera observation. Count calibration data and
wall time separately from reward-learning data. Preserve records and fitted scales.

Matched learning seeds are **531028119, 1047199272, 1958971091**, paired across raw
and conditioned arms. Each gets 25 episodes = **200 learning transitions**; total
learning is **1200 transitions**. Keep lambda=.8, alpha=.0001,
gamma_per_second=.95, epsilon=.2, zero initial weights, identical target schedules
and identical initial learner RNGs. The scale mapping is the only arm difference.
Never replace seeds, extend training or stop an arm on apparent success.

Fresh evaluation seeds, paired by replicate, are **62697349, 1197177281,
2007311671**. Evaluate each learned policy on 20 balanced delayed episodes; total
is **960 evaluation transitions**. Include shared initial/random baselines on the
same three schedules, with random action seeds **704719329, 1310920477,
2031518761**; this adds **960 evaluation transitions**. Only defensive copies of
readout weights and the declared frozen transformation cross into fresh evaluation.
Reset camera/vision/encoder scene state on every episode, use greedy ties to index
zero for learned/initial policies, and forbid learner updates.

Total cap: **3632 new transitions**, comprising 512 calibration + 1200 learning +
1920 fresh evaluation. Reset/initial-camera observations are additional and must
be reported separately. Per execution chunk: <=200 transitions and <=180 wall
seconds with save reserve. Aggregate cap: 20 minutes, including interpreter starts,
replay and persistence. Keep exact partial allocations if a cap is exhausted.

## Mechanism evidence and next decision

Report all raw/conditioned seeds, side-specific first-choice correctness, return,
training action counts, scale mapping, unchanged total calibration energy, and
first-choice Q contribution by block. Compare the paired fresh correctness change
and whether visual state contributions overcome action-dependent common bias.
The previously fixed 14/20-per-seed delayed screen is descriptive here, not a
retrospective rewrite of the original failure. A useful positive mechanism signal
is improvement in at least two seeds and no regression in the third, with the
conditioned arm reaching >=14/20 in every seed. Deterministic first choices on
two reset templates can score only 0, 10 or 20 out of 20, so this requires 20/20
in every conditioned seed; repeated scenes do not add independent visual trials.

Failure will trigger inspection of exposure, credit and the measured feature
interactions before another change. It does not automatically rule scaling out for
Pong, and does not authorize a multi-parameter sweep. A successful gate plus flat
unscaled Pong score evaluation supports a separate finite Pong conditioning arm,
keeping original seeds/settings for paired interpretation. Declare that allocation
before executing it. Final Pong seeds remain untouched until a complete frozen
configuration and acceptance protocol are agreed.

The preceding representation audit used 36 rendered observations and no policy
actions or rewards. It showed two tracked entities and full rank across the four
cue/consequence combinations from sample 3 onward, with equality interaction
carried by generic relation features. Thus a permanent lack of the equality
interaction is not assumed as the explanation for delayed-pilot failure.
