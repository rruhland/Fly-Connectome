# Same-class hit-return continuation: no demonstrated extra learning

The [fixed protocol](2026-10-21-direct-hit-continuation-protocol.md) completed at
source **aea715b**. Seven new updates failed the unchanged acquisition screen and
did not establish improvement over inherited theta9. Whole-input blanking reduced
performance, which does not establish ongoing tracking or reward acquisition.
No final streams were expanded, and this is one trained lineage, not complete M1B.

Artifact: `/workspace/m1b-experiments/pong-direct-hit-continuation-20261021-r1/`.
It retains committed source archive, protocol/configuration/runtime pins, exact
initial/mapped readouts, all journals, raw reports, anchors and paired statistics.

| Frozen policy, same 64 fresh DEV serves | First returns | All returns/opportunities | Rallies with ≥2 returns | Returns/physical s |
| --- | ---: | ---: | ---: | ---: |
| New endpoint, cumulative update 16 | 35/64 | 48/105 | 12 | .1581 |
| Original own initialization | 32/64 | 42/103 | 9 | .1513 |
| Down | 8/64 | 9/73 | 1 | .0557 |
| Up | 11/64 | 12/76 | 1 | .0691 |
| No-op | 16/64 | 20/82 | 4 | .0998 |
| Random1 | 21/64 | 28/88 | 6 | .1195 |
| Random10 | 16/64 | 24/85 | 7 | .1099 |
| Inherited theta9, secondary | 37/64 | 48/107 | 9 | .1587 |
| New endpoint blank, secondary | 16/64 | 20/82 | 4 | .0998 |

Primary lifts against the six-control maximum were **+4.69 pp first** (95% interval
−12.50 to 21.88) and **+4.94 pp all** (−6.90 to 16.67): failed ≥20 pp/both-positive-
lower-bound screen. New-minus-theta9 was **−3.13 pp first** (−14.06 to 7.81),
**+.85 pp all** (−7.97 to 9.47): no extra-learning evidence. Full-minus-blank was
**+29.69 pp first** (12.50 to 46.88), **+21.32 pp all** (8.26 to 34.36).
All intervals resampled the same 64 serve clusters 10,000 times; primary contrasts
recomputed the strongest control inside every draw. No invalid draws occurred.

Endpoint outcomes were 0 wins / 57 losses / 7 timeouts in 303.65 physical seconds;
theta9 0/59/5 in 302.48 s; initial 0/61/3 in 277.63 s. All nine cohorts had zero natural wins.
The opponent returned 579/579 balls, including 246/246 after an own return; this is
reliability on these tested trajectories, not a global guarantee. Blank selected
no-op on all 10,054 recorded transitions, exactly matching that physical baseline.

## Learning signal and cost

Training stopped at 159,842 new camera transitions: **131,983 used fitness**,
**18,885 unused partial-g17 fitness**, **8,974 diagnostic anchors**. The last
complete theta was selected; global g17's consumed draw was skipped. Its next RNG
and global index 18 are retained. Across the original and continued lineage,
319,715 training-phase observations comprise 277,544 used fitness, 33,197 unused
partial fitness and 8,974 anchors. Calibration reuse is separate.

Fixed pre-update anchors compared checkpoints 11/13/15 with theta9 on six serves
each: total hits 2 vs 3, 3 vs 3, 6 vs 5; hit-bearing breadth 2 vs 2, 2 vs 2, 5 vs 4.
These different checkpoint comparisons do not form one 18-serve policy evaluation.
Population contrasts remained populated; final function RMS .3844 and noise/policy ratio .6504
do not support exploration collapse. Calibration action variation is not a
visual-necessity or competence test.

Training took 1,740.42 s; evaluation including annotation 914.79 s; total 2,655.24 s
(44.25 min), inside all registered bounds. There were 103,900 evaluation cameras,
265,124 rendered images including resets and 215,238 encoded images. All 1,383
persistent tasks succeeded, with 9,937.95 summed worker seconds and zero recovery
replays. Durable heartbeat p50/p95 was 35.40/53.95 ms for fitness and 32.14/50.49 ms
for the endpoint; includes snapshots, excludes first/chunk-boundary steps, is not
a claim of 20 ms p95 or parallel throughput as single-agent latency.

## Verification and next experiment

All 23 focused checks passed. Actual zero-work recovery added/replayed 0 steps;
all 263,742 record/state/chain rows and every readout/optimizer boundary remained
unchanged. All 1,380 complete semantic pins passed in 8.59 s. Six partial/unstarted
jobs remain explicit. Parent independently verified every raw physics row, seven
manual optimizer updates, skipped RNG draw, candidate/anchor mappings, all 64 paired
first trajectories and every bootstrap contrast in 209.76 s, zero new observations:
`/workspace/m1b-experiments/direct-hit-continuation-parent-audit-20261021/summary.json`.

Cached old training accounting found 63,473/145,561 cameras strictly AFTER first
hit (43.61%; the first-hit transition stays). Binary first-hit split-half serve
updates had median cosine .011, supporting a serve-noise concern, not predicting
twelve-serve performance. The next reviewed design is a fresh-start fixed
first-return→full-rally curriculum: 8 directions × 12 serves, 120k/24 min, then 8 × 6,
80k/16 min; 40 training / 60 overall minutes. Evaluate the fixed phase-one checkpoint
only at the end as secondary, never select the better stage. This new 200k budget
is explicit; it changes objective emphasis and sample allocation, not an isolated
causal comparison. Implement/test/register before launch. Fixed T's actual pre-repair
calibration stream has not received an exact before/after validation; no refit was
made or silently assumed equivalent. Preserve all earlier negative evidence.
