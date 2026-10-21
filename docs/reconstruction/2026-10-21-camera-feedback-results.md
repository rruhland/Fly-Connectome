# Later visual updates benefit the fixed two-policy set

The [registered feedback test](2026-10-21-camera-feedback-protocol.md) passed its
prespecified aggregate sensitivity criterion. Across BOTH independently acquired
final policies, live images improved first-return success by **12.50 pp** over
both held and missing images after the same opening. This is a fixed-policy-set
causal sensitivity result, not forecast benefit, individual-seed significance,
score-winning mastery or completed M1B.

Source **415f066351c9be30a996d13fadf685b7b080dac7**; artifact
`/workspace/m1b-experiments/pong-camera-feedback-20261021-r1/`.
[Figure](/workspace/m1b-experiments/pong-camera-feedback-20261021-r1/feedback-summary.png)
([SVG](/workspace/m1b-experiments/pong-camera-feedback-20261021-r1/feedback-summary.svg)).
No learning updates, teacher inputs, policy selection or final-stream use.

## Fresh paired outcomes

Both policies consumed the first 20 genuine images including the initial image.
At .375 s, subsequent observations stayed live, repeated image20, or used honest
missing-frame calls with zero events. All conditions retained normal physics,
vision/encoder cadence, .65 opponent and ten-second full-rally evaluation. The
same 32 fresh-phase serves were shared across six cohorts.

| Fixed policy / input | First returns | All returns | Rallies ≥2 | Returns/s | Loss / timeout |
| --- | --- | --- | --- | --- | --- |
| Seed1 live | 10/32 | 19/47 | 7 | .168 | 28 / 4 |
| Seed1 warm hold | 6/32 | 7/38 | 1 | .095 | 31 / 1 |
| Seed1 warm missing | 9/32 | 12/43 | 2 | .136 | 31 / 1 |
| Seed2 live | 15/32 | 27/55 | 9 | .189 | 28 / 4 |
| Seed2 warm hold | 11/32 | 12/44 | 1 | .131 | 32 / 0 |
| Seed2 warm missing | 8/32 | 9/41 | 1 | .109 | 32 / 0 |

Zero wins; opponent returned **138/138**, including **84/84** after own returns.
These counts measure this cohort's opponent reliability. Later opportunities
are affected by each policy. The ten-second horizon also limits repeated-return
counts; max three is not a measured capacity ceiling.

Live rates **10/32 and 15/32** are lower than the earlier independent 64-serve
cohorts' **41/64 and 39/64**. Report that variability intact. This new cohort has
no simple-control acquisition screen and neither replaces the earlier passes
nor proves uniform generalization.

## Paired effects

Equal policy weight; resample **32 serve clusters**, retaining every policy/mode
cell, 10,000 draws with seed 194870293. Never treat the 64 policy-serve cells as
independent serves or pool policy-dependent contact denominators.

| Live minus | First-return lift, 95% interval | All-return lift, 95% interval |
| --- | --- | --- |
| Warm hold | **12.50 pp [4.69,20.31]** | **21.91 pp [13.53,29.61]** |
| Warm missing | **12.50 pp [3.13,21.88]** | **19.83 pp [12.17,27.55]** |

Both primary aggregate lower bounds exceed zero, each policy's two point effects
are positive, and no bootstrap draw is invalid: joint criterion passes. This is
not the 20-point acquisition gate. Per-policy first-return effects: seed1 hold
+12.50 pp [0,28.13], missing **+3.13 pp [−9.38,15.63]**; seed2 hold +12.50 pp
[3.13,25.00], missing +21.88 pp [9.38,37.50]. Thus dependence is not statistically
established for every seed/condition individually. Secondary intervals are descriptive.

Held images assert a false stationary scene. Missing frames preserve finite
tracking/context carry but also set frame-present bias to zero in a transform
fit on present images. Both interventions are OOD. The result supports benefit
from later visual updates for this tested policy set; it does not establish
optimal opening-only control is impossible, isolate object tracking, or show
explicit forecasts caused the benefit.

## Integrity and cost

All 192 jobs completed in **272.00 s**, within 12-minute rollout/15-minute overall
bounds, using **29,724 transition cameras +192 initial images**; no new training.
Four workers, 192 tasks, zero failures/recovery replays, summed task wall 955.00 s.
Same call cadence did not imply equal cost: per-seed summed live task wall
206.06/245.61s, hold 133.25/156.66 s, missing 110.22/103.20 s. Durable heartbeat p95
was 45.48–46.29ms live, 44.02–45.78ms hold and 31.46–31.87ms missing; shorter
rallies and changed visual work both affect runtime. The 20 ms target remains unmet.

All 38 focused tests passed. Archived 886d05d/current default replay matched both
actual endpoint prefixes' records, physics/vision/encoder, previous/latest state
and learned hashes exactly. Across initial/final focused runs and that bridge,
preflight counted 1,270 diagnostic camera executions (166 recovery subset), 104
initial images and2,982 physics ticks; zero novel candidate data. Setup initially
rejected equivalent projection descriptors with different absolute provenance
paths; both were retained separately before any allocation clock or cameras.

Own verification passed 2.93 s: SIX actual mode×seed zero-work recoveries, all 192
completion pins, all29,724 record/state/chain rows and weights/inputs unchanged;
zero new/replayed cameras. Independent [parent audit](/workspace/m1b-experiments/camera-feedback-parent-audit-20261021/summary.json)
passed 13.60 s: every raw physics row, common 20-action prefixes, mask onset,
geometry, seed/source/readout pins and both aggregate/per-policy statistics.
All candidate/audit workers exited; source unlocked. Raw journals and source,
protocol, lineage, exclusions, runtime and weight pins remain retained.

Next: one prospectively registered unchanged third full-condition training seed
paired with a state-only arm on common fresh evaluation serves, using the old-core
worktree. This tests explicit forecast-channel availability; one pair supports
only a development signal. Full M1B and untouched final evidence remain open.
