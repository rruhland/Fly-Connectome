# Reliable-opponent return learning: completed exploratory screen

Neither trained policy passed the registered return-learning screen. Holding an
action for ten camera frames performed better than camera-rate control, but its
small advantage over simple controls was uncertain. This is one training seed
and 24 shared development serves, not demonstrated acquisition or standard Pong
competence. No final holdout stream was used.

## What was tested

[Registered protocol](2026-10-16-reliable-return-learning-protocol.md): two
from-zero DCT-SARSA policies, seed 172900127, 12,000 camera transitions each,
alpha .001, epsilon .2, physical gamma .95/second and camera lambda .995. Hold10
uses lambda .995^10 and discounted macro rewards; vision and durable commits
remain per camera. The exact frozen 214-coordinate transform and cap256 were
unchanged. The frozen visual prior still has empty long-context banks and uses
its four-step fallback. No backprop was used.

The opponent speed was .65. Training paid **only +1 for an own-paddle hit**;
game-score coefficient and survival reward were zero. Raw game scores remained
separate and still ended rallies. Contact ground truth entered the scalar reward
only. Fresh greedy evaluation used no bonus and performed no learner updates.

All seven policies faced the same 24 independently seeded, fresh-phase serves,
with a ten-second limit. An incoming opportunity means an actual left collision-
plane crossing, including misses. An opponent-going serve is not an opportunity
until it comes back. First-incoming ball positions, velocities and crossing ticks
matched exactly across policies, giving a clean paired first-return comparison.

## Fresh results

| Policy | First incoming returned | All returns / opportunities | Rallies with ≥2 returns | Longest streak | Returns/second | Losses / timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained hold1 | 3/24 (12.5%) | 3/27 (11.1%) | 0 | 1 | .060 | 24/0 |
| Trained hold10 | 9/24 (37.5%) | 13/36 (36.1%) | 3 | 3 | .154 | 23/1 |
| Initial / constant down | 6/24 (25.0%) | 7/31 (22.6%) | 1 | 2 | .108 | 24/0 |
| Constant up | 7/24 (29.2%) | 10/33 (30.3%) | 2 | 3 | .138 | 23/1 |
| No-op | 7/24 (29.2%) | 8/32 (25.0%) | 1 | 2 | .117 | 24/0 |
| Random hold1 | 5/24 (20.8%) | 5/29 (17.2%) | 0 | 1 | .087 | 24/0 |
| Random hold10 | 4/24 (16.7%) | 5/29 (17.2%) | 1 | 2 | .086 | 24/0 |

There were **zero wins or natural-serve wins**. The opponent returned 105/105
evaluation opportunities, including 49/49 after own returns. This verifies its
reliability on these trajectories, not globally. Both timeouts had an unfinished
opponent-going approach, rather than an uncounted incoming opportunity.

Relative to the strongest simple control, hold10 improved first-return success
by 8.33 percentage points (paired 95% interval −25.0 to +20.83) and overall return
fraction by 5.81 points (−20.23 to +19.24). Hold1 lifts were −16.67 and −19.19
points. Neither met the predeclared ≥20-point lift on both metrics with both
intervals above zero. The 10,000 bootstrap draws resampled 24 serve clusters and
recomputed the strongest control within every draw; contacts within a rally were
not treated as independent samples.

## What this resolves, and what it does not

The strong opponent removed natural-win contamination and hit-only reward removed
the need to turn a successful return into a terminal score. Nevertheless, each
learner experienced only 22 or 29 training hit rewards, over approximately 240
physical seconds. Their completed training rallies numbered 86 and 69, with one
unfinished rally each. A negative at this budget does not establish that delayed
credit or the representation is intrinsically incapable.

The persistence intervention occurred: median training action runs were .025
seconds at hold1 and .20 seconds at hold10. It also changed update exposure:
12,000 versus 1,228 updates, with a six-camera pending hold at the hold10 endpoint.
Thus the result does not isolate exploration coherence from update frequency or
macro credit geometry. Later training states also differ after actions diverge.

Weights and Q values remained finite (maximum absolute Q 3.91 and 1.93), with
actual policy norm² bounded by 256 plus roundoff. Raw feature norm² reached
10,878 at hold1 and 267 at hold10. Above-256 raw rows occurred only after a prior
own hit in both runs. High norms alone do not identify tracker corruption; prior
field-level evidence makes full feature reliability a remaining concern.

The parent's separate .65 vision check covered 2,048 frames: observed-or-predicted
ball coverage within three pixels was 99.80–99.90%, with mean available error
.27–.28 pixels. Fresh detections were intermittent (71.7–81.1%). Frozen hashes
were unchanged and no encoded speed exceeded five field units/second. This short
localization assay does not validate every feature or exclude rare long-run faults.

The most informative next diagnostic would test immediate privileged action
feedback **online on the learner's own closed-loop states**, using the same visual
inputs and update machinery. The earlier cached bandit's 83–86% held-out agreement
did not test that distribution. This would be quarantined teacher supervision,
never a reward-learning candidate or exported initializer. No such allocation is
launched here. If further cadence work is chosen, camera-rate learning with only
temporally correlated exploratory actions would separate exploration persistence
from the full hold10 intervention.

## Integrity, cost and artifacts

Both arms and all 168 evaluation jobs completed: **24,000 training + 22,838
evaluation transitions**, zero recovery replay, 757.32 seconds shared wall
(12.62 minutes). There were 47,163 rendered images including initial frames;
30,930 passed through vision and simple baselines skipped perception for 16,233.
The 192 persistent tasks consumed 1,831.87 summed task-wall seconds, which is not
CPU time. Registered physics annotation replayed all 46,838 saved transitions;
its cost is included in coordinator wall time. Existing 1,024-row calibration was
reused without new calibration observations.

Source commit **0ea413a**, branch `reconstruction/m1b-continuation`. Focused checks
covered hit-only/legacy reward arithmetic, real held-action partial boundaries,
exact crash replay, opportunity/first-trajectory semantics, bootstrap controls and
zero-work recovery. Three invocations passed 8, 12 and 6 tests (including repeats).
The complete source archive preserves the original pinned protocol bytes.
The parent's independent raw-tick audit verified all 46,838 physical/reward rows,
13,228 TD equations and all 24 paired first-incoming trajectories in 34.90 seconds,
with zero new observations. Its separate bootstrap implementation exactly matched
all reported point estimates and intervals.

All artifacts are in
`/workspace/m1b-experiments/pong-reliable-returns-20261016-r1/`: manifest, source
archive, journals, endpoint theta/W/cap, `summary.json`, `return-annotations.json`,
`training-mechanism-summary.json`, `cost-summary.json` and a copied parent audit
summary. Full independent evidence is at
`/workspace/m1b-experiments/reliable-return-parent-audit-20261016/`.
The separate opponent probe used 42,178 .45 and 46,072 .65 physics-camera events,
each replayed once through scalar physics (40.18 + 46.79 seconds). Its .65
tracking reference returned 139/139 at hold1 and 60/81 at hold10. Probe and vision
work are diagnostic compute, not candidate learning exposure. The external
cumulative ledger retains these costs and earlier failed allocations separately.
