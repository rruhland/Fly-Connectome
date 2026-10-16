# Sustained exploration: completed camera-rate return-learning comparison

Sustaining exploratory drives increased training contacts but did **not** improve
frozen fresh control. Neither learner passed the registered screen or exceeded
the strongest simple control. This is experimental knowledge, not demonstrated
Pong acquisition. One training seed and 24 shared development serves do not
establish asymptotic failure or a universally harmful exploration method.

## Fixed comparison and fresh results

[Protocol](2026-10-16-sustained-exploration-protocol.md): two from-zero policies,
matched seed 736187003, exactly 24,000 camera transitions **and SARSA updates** each.
Both used hold1, alpha .001, lambda .995, physical gamma .95/second, epsilon .2,
the unchanged frozen DCT/cap256 and .65 opponent. Learning paid only +1 per own
contact, with zero game-score or survival contribution. No teacher labels,
backprop or final holdout use.

Independent exploration redraws each camera. Sustained exploration draws mode
with probability .2 per ten-frame block and reuses one random action only during
exploratory blocks. Greedy actions remain responsive each camera. This removes
the previous full-hold experiment's update-frequency/macro-credit change.

All seven frozen policies shared 24 **new**, independently seeded, fresh-phase
ten-second serves. Both learned evaluations used greedy hold1, raw score only,
and no updates. First-incoming ball positions, velocities and ticks matched
exactly across policies. Actual collision-plane crossings define opportunities;
opponent-going initial serves and unfinished approaches do not inflate them.

| Policy | First incoming returned | All returns/opportunities | Rallies with ≥2 returns | Longest streak | Returns/second | Losses/timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent exploration | 6/24 (25.0%) | 8/32 (25.0%) | 2 | 2 | .105 | 24/0 |
| Sustained exploration | 3/24 (12.5%) | 4/28 (14.3%) | 1 | 2 | .064 | 24/0 |
| Down/initial | 7/24 (29.2%) | 7/31 (22.6%) | 0 | 1 | .094 | 24/0 |
| Up | 1/24 (4.2%) | 1/25 (4.0%) | 0 | 1 | .019 | 24/0 |
| No-op | 9/24 (37.5%) | 10/33 (30.3%) | 1 | 2 | .114 | 23/1 |
| Random hold1 | 10/24 (41.7%) | 12/35 (34.3%) | 2 | 2 | .125 | 23/1 |
| Random hold10 | 4/24 (16.7%) | 4/28 (14.3%) | 0 | 1 | .063 | 24/0 |

There were zero wins or natural wins. The opponent returned **144/144** evaluation
balls, including 46/46 after own contacts. Both control timeouts had unfinished
incoming approaches, reported rather than counted as missed or successful returns.

Sustained minus independent was **−12.5 percentage points** for first returns
(paired 95% interval −37.5 to +12.5) and **−10.71 points** overall (−31.14 to +11.47).
The intervals cross zero; these descriptive comparisons do not establish general
harm and are not multiplicity-adjusted definitive evidence. Relative to the
strongest simple control, first/overall lifts were −16.67/−9.29 points for
independent and −29.17/−20.00 for sustained. Neither passed the predeclared ≥20-point
lift on both metrics with both interval lower bounds above zero. All intervals
resampled the same 24 serve clusters over 10,000 draws; comparisons against simple
controls recomputed the strongest control within each draw. Correlated contacts
were not independent trials.

## What changed in training

| Observed mechanism | Independent | Sustained |
| --- | ---: | ---: |
| Positive contact rewards | 46 | 60 |
| Exploratory camera fraction | 20.08% | 19.76% |
| Exploratory physical-time fraction | 20.03% | 19.78% |
| Strictly non-greedy selections | 13.19% | 13.51% |
| Median exploratory action run | 1 camera | 10 cameras |
| Median overall actuator run | 1 camera | 2 cameras |
| SARSA updates | 24,000 | 24,000 |

The intended persistence occurred without increasing exploration occupancy or
reducing update count. Sustained collected about 30% more hit rewards over the
same approximately 479 physical seconds, but that advantage did not transfer to
frozen greedy return performance. Training completed 160 losses for independent
and 128 losses plus one timeout for sustained, with one unfinished rally each.
Only 46/60 positive events remain limited exposure. Later states and episode
counts differ after actions diverge; training ratios are not matched-state causal
estimates or independent acquisition evidence.

All weights/Q remained finite (maximum absolute Q 4.90/27.25), with actual policy
norm² bounded by 256 plus roundoff. Raw norms reached roughly 20,611/21,300;
this alone does not identify their cause. Exploration mode/phase is hidden from
Q features, an explicit behavior-memory caveat. Pre-update selection metadata
stays with the pending action, separately from execution-time Q and the next
bootstrap selection. Actual truncation discarded its unused bootstrap/block tail.

## Next distinction, not a new allocation

This result does not justify another tiny epsilon/lambda/hold sweep. If further
reward learning is chosen, the strongest tentative alternative is direct policy
optimization of **hit-count return in the same frozen linear class**, using the
existing mirrored search machinery. That would test useful learning without
requiring a linear Bellman-Q fit. Prior score-only searches failed; high search
dimension, sparse reward and camera cost remain limitations, so success is not
promised and any allocation needs a new fixed protocol. A frozen exploratory
deployment test would need matching randomized and bias-only controls and would
not by itself prove acquisition. No teacher diagnostic is the default next step.
No further training, replication or holdout is allocated automatically.

## Integrity, runtime and artifacts

Source **77310e7910ba90f8bd1787930f645bb7ad36779a**, locally archived before the
clock started. Eighteen focused tests passed in 14.87 seconds, including exact
block1 legacy action/RNG/update equivalence, current versus bootstrap metadata,
partial-block truncation, exact crash replay and mapped greedy evaluator handoff.
All source/protocol/runtime/config pins and per-camera durability were retained.
The parent's independent audit passed all **73,649 physical/reward rows**, 48,000
TD equations, 24 paired first-incoming trajectories and exploration block/action/
bootstrap continuity, including the real training truncation. Its separate
bootstrap code exactly reproduced both max-control screens and the direct-arm
contrast. This diagnostic reexecution took **57.12 seconds**, with no new camera
observations or learning; it is recorded separately from allocation wall time.

All jobs completed: **48,000 training + 25,649 evaluation camera transitions**,
zero recovery replay, **1,345.97 seconds / 22.43 minutes** shared wall. There were
74,108 rendered frames including initial frames, of which 55,268 passed through
vision and 18,840 used the validated baseline perception skip. The 216 persistent
tasks consumed 2,964.88 summed task-wall seconds, not CPU time. Registered physics
annotations reexecuted all 73,649 saved actions within coordinator wall time.
The existing 1,024-row score-free calibration was reused with no new observations.

Artifacts: `/workspace/m1b-experiments/pong-sustained-exploration-20261016-r1/`:
manifest, complete source archive, journals, theta/W/cap, `summary.json`,
`return-annotations.json`, `extract_evidence.py` and `final-evidence-extraction.json`.
Independent evidence is at
`/workspace/m1b-experiments/sustained-exploration-parent-audit-20261016/`.
Its summary is also copied into the study directory.
The cumulative external ledger preserves this allocation and diagnostic replay
separately from all previous failures; no prior negative was replaced.
