# Independent curriculum replica: return acquisition reproduced

The [registered independent replica](2026-10-21-first-return-replication-protocol.md)
passed its unchanged development screen, following the [first positive seed](2026-10-21-first-return-curriculum-results.md).
Fresh own initial → first-return stage → final gave **6/64 → 37/64 → 39/64**
first returns; all returns **9/72 → 53/111 → 64/122**. This supports return
acquisition in two independent training seeds. It does not establish score-winning
mastery, forecast benefit, ongoing feedback dependence or complete M1B. Three-seed
evidence and the broader [completion checklist](2026-10-21-continuous-m1b-plan.md) remain open.

Source **886d05d48d19f1cf31754f27bfc638f5e5da1ed9**; artifact
`/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/`.
[Two-seed chart](/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/replication-summary.png)
([SVG](/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/replication-summary.svg)).
Fresh initializer/training/DEV/action streams, unchanged vision/T/cap/.65 opponent,
optimizer and whole staged procedure; no inherited trained weights, teacher labels,
privileged policy inputs or backprop. All earlier negatives remain retained.

## Fresh frozen results

Each cohort used the same 64 new fresh-phase ten-second serves, frozen greedy
hold1 and no updates/task cutoff. The two studies used distinct DEV cohorts;
they are not a paired head-to-head comparison between trained seeds.

| Policy | First returns | All returns | Rallies ≥2 returns | Returns/s | Losses / timeouts |
| --- | --- | --- | --- | --- | --- |
| Final | 39/64 | 64/122 | 22 | 0.183 | 58 / 6 |
| Own initial | 6/64 | 9/72 | 2 | 0.061 | 63 / 1 |
| First-return checkpoint | 37/64 | 53/111 | 16 | 0.159 | 58 / 6 |
| Constant down | 10/64 | 12/75 | 2 | 0.073 | 63 / 1 |
| Constant up | 11/64 | 16/77 | 4 | 0.091 | 61 / 3 |
| No-op | 16/64 | 16/80 | 0 | 0.087 | 64 / 0 |
| Random hold1 | 16/64 | 22/85 | 5 | 0.108 | 63 / 1 |
| Random hold10 | 14/64 | 19/80 | 5 | 0.098 | 61 / 3 |
| Final, blank camera | 10/64 | 12/75 | 2 | 0.073 | 63 / 1 |

The opponent returned **524/524** opportunities, including **218/218** after own
returns. Zero scored/natural wins; reliability is measured on this cohort. Final
physical time 349.93 s. The ten-second horizon limits possible consecutive returns
to three: a fourth cannot occur before approximately 10.34 s at the physics speed
bound. Thus max three is not a measured policy capacity ceiling; longer-rally
competence is untested. Unfinished incoming approaches remain separately reported.

Primary lift over own initial plus five simple controls:

| Metric | Point lift | Paired 95% interval |
| --- | --- | --- |
| First incoming return | **35.94 pp** | **15.63 to 46.88 pp** |
| All incoming return fraction | **26.58 pp** | **13.81 to 33.65 pp** |

Both POINT lifts exceed 20 pp and both CI lower bounds exceed zero: **passes**.
Intervals need not lie wholly above 20 pp. Bootstrap used 10,000 paired draws of
64 serve clusters, recomputing the strongest control for each metric inside each
draw; zero invalid draws and identical first-incoming geometry. Later opportunities
are policy-dependent, not independent trials. No checkpoint was chosen by DEV.

Prespecified final-minus-owninitial: first **+51.56 pp [37.50,65.63]**, all
**+39.96 pp [27.03,51.79]**. Phase0-minus-initial also improved both metrics.
Final-minus-phase0 was **+3.13 pp [−7.81,14.06]** first and
**+4.71 pp [−3.36,12.74]** all: additional second-stage benefit is inconclusive
in this replica. These descriptive intervals are unadjusted; the procedure's
replicated acquisition does not prove either stage necessary or superiority over
a same-budget alternative curriculum/optimizer.

Live-minus-blank was +45.31 pp [29.69,60.94] first and +36.46 pp [25.15,47.39] all.
All **8,215** blank transitions chose constant down, with exactly matching down
action/physics traces. This shows whole-input dependence under OOD blanking,
not ongoing tracking or forecast-specific benefit.

## Exposure, runtime and integrity

Six first-return updates used **112,224/119,904** cameras; 7,680 unused partial
rows and consumed global6 were skipped. Four full-rally updates used
**76,165/79,867**; 3,702 unused rows/global11 were skipped, next global12. Its
phase-one carry derived its own theta/RNG/global7, rather than hardcoding the
first seed's five-update/global6 boundary. Ten total updates, **199,771 training
cameras = 188,389 used + 11,382 unused**, then **95,918 evaluation** cameras.
Initial images 2,216; rendered 297,905; encoded 251,506. First-hit task stops
retained the full camera and raw game flags, never fake wins.

Training **30.79 min**, evaluation/annotation/statistics **12.41 min**, total
**43.20 min**, within every original cap. 2,216 persistent tasks, zero failures
or recovery replays; summed task wall 9,718.63 s. Durable heartbeat p50/p95:
28.05/46.09 ms first-return, 28.32/46.82 ms full-rally, 27.81/44.83 ms endpoint;
snapshot steps included, first/chunk-boundary pairs excluded. The 20 ms p95
target remains unmet.

All **32 focused tests** and actual-transform/default-equivalence/worker proof
passed before source freeze. Own zero-work recovery passed **10.58 s**: zero new
or replayed transitions, all **295,689** ordered record/state/chain rows and
readouts/phase boundaries unchanged, **2,213** completion pins valid.
Independent full audit passed **221.52 s**: all 295,689 raw physics rows, ten
manual optimizer updates, RNG/candidates/phase carry/skipped draws, geometry and
primary plus all four secondary bootstrap comparisons matched exactly. Three
unused unfinished rollouts each retained one raw hit while summary hits remained
zero (summaries count completed rallies); partial fitness was never included.
The first auditor attempt assumed otherwise and replayed 199,371 rows before
stopping; its elapsed time was not captured. The corrected audit and an actual
partial-state recovery (0.314 s, zero new/replayed steps, unchanged state/row
hashes) confirm expected semantics, not a candidate fix.
Independent audit evidence is retained at
`/workspace/m1b-experiments/first-return-replication-parent-audit-20261021/summary.json`.
Audit/recovery computation is separate from candidate exposure.

Storage management overlapped the unchanged clock for 49.63 s (31.85 total CPU s):
900 CLOSED older score-search-continuation journals were losslessly compressed
from 2.680 GB to .376 GB after full streaming byte verification before unlink.
Their hash manifest, restore script/command and all plain metadata/readouts/source
remain in that cold artifact; live journals were untouched. No deadline extension.
See `/workspace/m1b-experiments/storage-management-20261021/cold-continuation-context.json`.

Next reviewed small allocation is a frozen live/warm-held/warm-missing camera
comparison of BOTH endpoints on 32 new paired serves, after integrity verification.
This tests these policies' sensory dependence, with explicit OOD/bias/finite-carry
limits; no training or final streams. A third unchanged full-condition seed plus
a matched state-only comparison is a subsequent prospective design direction,
not an allocation or completed evidence here.
