# Frozen score transfer: neither policy passes

Both preselected return-learning policies were evaluated once against a .35
opponent, with their original generic visual prior and frozen heads. This is
transfer evaluation, not learning from score reward. The [registered protocol](2026-10-21-pong-score-transfer-protocol.md)
used64 fresh paired development serves,10s limits and all seven controls.

| Cohort | Wins /64 | Own-hit wins | Natural wins | Losses | Timeouts |
|---|---:|---:|---:|---:|---:|
| Learned seed1 |14|5|9|43|7|
| Learned seed2 |16|7|9|44|4|
| Seed1 initial |12|3|9|52|0|
| Seed2 initial |9|0|9|55|0|
| Down |10|1|9|54|0|
| No-op |12|3|9|51|1|
| Up |10|1|9|54|0|
| Random1 |11|2|9|53|0|
| Random10 |11|2|9|53|0|

Seed1's lift over the strongest control is **3.125pp**, paired95% CI
**[-6.25,7.8125]**; seed2's is **6.25pp [-3.125,12.5]**. Both fail the unchanged
>=10pp/positive-lower-bound screen. Total-win and own-hit-win lifts/intervals
coincide because the same nine natural wins cancel on every paired serve. They
are two accounting views, not independent evidence. The fixed-two-policy average
lift is4.6875pp [-3.90625,8.59375], descriptive only.

Return behavior persists: first returns25/55 and37/55; all returns41/84 and54/98,
versus the strongest simple first-return rate14/55 and all-return rate16/67.
There is no convincing score-winning transfer. Changing opponent speed also
changes paddle contact positions and returned trajectories, so this result does
not isolate reward misalignment or prove the policy cannot learn to win. A future
native-score learning attempt at fixed.35 is distinct from another frozen speed
probe; no opponent sweep or retraining occurred here.

All576 jobs completed in539.247s (462.622s rollout), with75,899 new transition
cameras,576 initial images and39,196 encoded images. No actor or visual updates
occurred. Annotation reexecuted75,899 physics transitions; independent audit
reexecuted the same75,899 in33.061s, with no new vision or candidate observations.
Own all-pin/ordered-row verification plus nine actual zero-work recoveries passed
5.984s with zero replay transitions. Both audits agree on raw physics, score and
return counts, shared geometry and bootstrap statistics. A verifier-only BLOB
serialization failure before actual recovery is preserved separately; candidate
data were unchanged.

The prelaunch default bridge reexecuted64 already observed cameras+2 initial
images in4.970s and exactly matched both qualified-policy archived prefixes.
35 focused checks passed, followed by8 final harness checks. A postlaunch metadata
audit found260 previously omitted high uint32 action IDs; all actual new seeds
were disjoint (2,108 total checked IDs). No live source, root or cohort changed;
future exclusion collection covers the full uint32 range. Final streams remain
unexpanded.

Source **10acd5e**, exec12221/PID93633 and all workers exited0. Artifact
`/workspace/m1b-experiments/pong-score-transfer-20261021-r1/` contains immutable
source/input/seed pins, raw journals, outcome and completion verification.
Independent evidence:
`/workspace/m1b-experiments/score-transfer-parent-audit-20261021/`.
Next approved work is the separate new-episode pendulum visual-learning ON/OFF
continuation from the predetermined experienced20k boundary, with unchanged
learner settings and original competence gate. M1B remains incomplete.
