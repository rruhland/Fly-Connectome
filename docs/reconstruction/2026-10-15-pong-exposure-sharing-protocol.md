# Longer .25-opponent exposure versus generic spatial credit sharing

The completed 6000-transition raw .25 screen did not improve upon its strongest
simple baseline. Its 70 training rallies contained only four own-hit-followed
wins. This is a small exposure allocation, not evidence of asymptotic failure.
Exact trajectory replay found current/future feature medians of only four/two
visits per action; 40%/44% of their active coordinates lacked at least one action.
By 6000, state-dependent Q spread exceeded common action bias, making an automatic
metadata-normalization intervention less compelling. This study tests longer
exposure alongside one specific response to sparse spatial support.

Train **two from-zero 30,000-transition policies**, original development seed
**172900127**, opponent speed cap **.25**, alpha **.001**, lambda **.995**,
gamma_per_second **.95**, epsilon **.2**, frozen default vision and score-only ±1
terminal reward. Arm `raw` uses unchanged policy features. Arm `sharing` applies
the fixed invertible policy-boundary map below. Both use the same actual body,
legal actions, camera cadence, episode/reset rules and **2400-tick** rally limit.
There is no RMS block balancing, reward shaping, new learned representation,
oracle input, action persistence or longer-credit change. Original .45 training
remains paused; no result here claims original-opponent success.

For each 16x16 current/future plane, let P2/P4 average within fixed 2x2/4x4 cells
and repeat the mean to their members. The sharing map is
`D = (I + P2 + P4) / sqrt(2.0625)`. It is invertible:
`D^-1 = sqrt(2.0625) * (I - P2/2 - P4/6)`. Dimension and linear function class are
unchanged. It preserves unit-point energy and gives neighbor similarity .51515
inside the same 2x2 cell, .15152 inside the same 4x4 cell and zero outside it.
Unit-point preservation is insufficient for mixtures. Independent **1024
score-free random-camera transitions**, environment **849375169**, action seed
**686753009**, .25 opponent, fitted fixed compensation **.9847227472512416** for
current and **.9573019386665991** for future. This preserves each block's actual
calibration mean squared norm. All other blocks and raw transport stay unchanged.
The complete versioned calibration artifact is frozen with policy weights and
included in evaluator mutation checks. Its 1034 actual images (including resets)
and 47.80-second cost are accounted separately, with no learner updates.

Start both arms from zero rather than continuing the selected old screen state.
Save exact weights at **6000, 15000 and 30000** for descriptive trajectory/weight
diagnostics. The evaluated/selected endpoint is **30000**, regardless of the
earlier checkpoints; no opportunistic checkpoint selection or budget extension.
Verify the new raw first-6000 trajectory and weights equal the old raw screen.

Freshly evaluate both final policies on **24 attempted rallies** of shared
development stream **904271833**, .25 physics and the same 20-second limits.
Extend all four simple controls to this exact 24-rally allocation once: initial/
constant-down, constant-up, no-op and uniform-random (action seed **447201113**).
Observation-independent controls skip perception using the already validated
exact action/physics path; verify their first-12 records equal the old controls.
No final seeds are used. Policies share serve instances; these are paired results,
not independent trials for every policy-by-serve cell.

Report total wins and own-hit-followed wins per attempted rally; natural serve
wins (zero own hits), losses, all truncations, completion fraction, completed-rally
score differential, physical durations/return and hits. Truncations are nonwins
and do not separately disqualify acquisition. The prospective **development
screen** requires both total wins and own-hit-followed wins to exceed their
respective largest simple-baseline counts by **at least four of 24**. This is a
promising-signal rule, not final proof of acquisition. If both arms pass, choose
most total wins, tie to raw for parsimony; freeze the complete config and 30000
budget before independent-seed replication. If neither passes, report both
failures without automatic extension or more modifier arms. Any replication,
blank-camera ablation or untouched holdout receives a separate finite protocol.

Maximum allocation: **60,000 new learning transitions**, **144 attempted
development evaluation rallies** across six policies, conservative **172,800
evaluation transitions**, and **45 aggregate wall minutes**. Use two independent
learning workers and at most one concurrent lightweight baseline worker on the
four-core environment. Expected training cost from the measured .030–.033
seconds/transition is about 15–17 wall minutes for parallel arms; fresh evaluation
adds policy-dependent cost, about 15–20 minutes if every rally truncates.
Each durable chunk contains <=200 new transitions, up to five chunks/interpreter,
shared 160-second work and 180-second subprocess bounds. Compact progress reports
omit duplicated full timing arrays; complete timing remains in SQLite. Recovery,
per-transition hashes and complete checkpoints remain unchanged.

Preserve full source/runtime snapshots, exact manifests/configs and calibration,
all journals, weights/hashes, process costs, numerical/Q/feature diagnostics and
all partial/failure allocations. Existing reserved final seeds remain unused.
