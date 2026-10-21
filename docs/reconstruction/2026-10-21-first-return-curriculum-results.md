# First-return curriculum: audited single-seed acquisition screen passes

The [registered procedure](2026-10-21-first-return-curriculum-protocol.md) passed
its unchanged development screen. On 64 fresh paired serves, first-return success
rose from **14/64 initially →29/64 after the first-return phase →41/64 finally**.
Final all-incoming returns were **63/114**, versus the strongest control's 20/82.
Independent physics/optimizer/statistics audit and actual zero-work recovery passed.
This establishes a positive single-seed hit-return result. Independent training
replication, winning/score-task competence and broader M1B remain open.

Source **36b05e71bd11abe175df6aa07e8fe12fa3339291**; artifact
`/workspace/m1b-experiments/pong-first-return-curriculum-20261021-r1/`.
The artifact retains source archive, protocol/helper/runtime pins, raw journals,
all coefficients/RNG states, `summary.json`, `cost-summary.json`,
`completion-verification.json` and `acquisition-summary.png`/SVG.

## Fresh frozen results

Each cohort used the same 64 fresh-phase ten-second serves, opponent speed .65.
All learned policies were frozen greedy hold1. No evaluation updates or task cutoffs.

| Policy | First returns | All returns | Rallies ≥2 returns | Returns/s | Losses / timeouts |
| --- | --- | --- | --- | --- | --- |
| Final | **41/64** | **63/114** | 18 | .179 | 50 /14 |
| Own initial | 14/64 | 19/80 | 4 | .104 | 61 /3 |
| First-return checkpoint | 29/64 | 40/102 | 10 | .156 | 62 /2 |
| Constant down | 8/64 | 9/73 | 1 | .061 | 64 /0 |
| Constant up | 14/64 | 17/81 | 3 | .096 | 64 /0 |
| No-op | 14/64 | 20/82 | 5 | .105 | 62 /2 |
| Random hold1 | 13/64 | 18/79 | 4 | .100 | 61 /3 |
| Random hold10 | 11/64 | 12/75 | 1 | .075 | 63 /1 |
| Final, blank camera | 14/64 | 20/82 | 5 | .105 | 62 /2 |

There were **zero scored/natural wins**. The opponent returned 475/475 evaluation
opportunities, including 214/214 after own returns; reliability is measured on this
cohort. Final physical time 352.17 s; maximum 3 consecutive returns. One failed
incoming crossing occurred before a timeout, so63/(63+50) would give the wrong
ratio: the correct denominator is114. Unfinished approaches are reported separately.

Primary final lift over the maximum of own initial plus five simple controls:

| Metric | Point lift | Paired 95% interval |
| --- | --- | --- |
| First incoming return | **42.19 pp** | **23.44 to50.00 pp** |
| All incoming return fraction | **30.87 pp** | **16.23 to37.51 pp** |

Both points exceed 20 pp and both lower bounds exceed zero: **screen passes**.
The 10,000 bootstrap draws resampled 64 serve clusters, preserving paired policies;
the strongest control was recomputed separately for each metric inside each draw.
Zero invalid draws; all first-incoming geometries matched. Later opportunities
are policy-dependent, so this is not a pooled independent-ball sample.

Prespecified secondary stage comparisons also improved: phase0 minus own initial
first **+23.44pp [6.25,40.63]**, all **+15.47pp [1.42,28.99]**; final minus phase0
first **+18.75pp [4.69,32.81]**, all **+16.05pp [6.60,25.83]**. Both stages were
evaluated only after training; the final checkpoint was never selected by DEV.
These descriptive comparisons have no multiplicity adjustment. They support
progress within this procedure, not causal superiority of the curriculum over
same-budget full-rally search with a different seed.

Blanking reduced first returns by 42.19 pp [28.13,56.25] and all returns by 30.87 pp
[19.01,43.14]. All 9,537 blank transitions chose no-op, with exact no-op action/
physics projection. This establishes whole-input dependence under OOD blanking;
it does not establish ongoing tracking or forecast-specific benefit. An explicitly
post-hoc final-minus-owninitial comparison is retained separately and changes no
screen/selection: first+42.19pp [26.56,56.25], all+31.51pp [18.65,44.50].

## Exposure, runtime and integrity

Frozen repaired vision, exact prior score-free DCT/rank183/cap256 and model bytes
were unchanged. Fresh random RMS.10 initialization; no teacher labels, privileged
policy inputs or backprop. Five first-return updates used 99,041 of 119,827 cameras;
20,786 cameras from165 completed rollout tasks in an unfinished population were
unused. Its consumed global 5 draw was skipped. Four full-rally updates used all
79,849 cameras; an unexecuted global 10 draw was skipped. Nine total updates,
**199,676 new training cameras /178,890 used /20,786 unused**, then92,013 evaluation
cameras: **291,689 total**. Initial/reset images2,085; rendered293,774; encoded250,629.
First-hit task stops retained the entire hit camera and raw game flags/score;
they were not recorded as wins or natural terminations.

Training 31.06 min; evaluation plus annotation/statistics 12.28 min; total**43.34min**,
within all registered bounds. 2,086 persistent tasks, zero failures/recovery replays;
summed task wall 9,812.86 s. Durable heartbeat p50/p95 were 28.83/46.63 ms (phase0),
29.04/47.03 ms (phase1),29.86/47.60 ms (final evaluation); snapshot steps are included,
first/chunk-boundary pairs omitted. These do not meet a 20 ms p95 target.

Own verification took 13.19 s: actual zero-work recovery of a completed first-hit
task and full-rally evaluation, zero new/replayed transitions, all 291,689 ordered
record/state/chain rows and readouts/phase boundaries unchanged, all 2,085 completion
pins valid. Independent audit took**225.35s**, verifying all raw physics, cutoff
truth, fresh initializer, nine manual optimizer updates, candidates/RNG/phase
carry/skipped draws, 64 paired geometries and all registered bootstrap comparisons:
`/workspace/m1b-experiments/first-return-curriculum-parent-audit-20261021/summary.json`.
Audit/recovery compute is separate from candidate exposure. All earlier negatives,
including the [failed same-class continuation](2026-10-21-direct-hit-continuation-results.md),
remain retained. Source was unlocked only after both proofs passed.

Next reviewed condition is **one independent from-scratch whole-procedure replica**,
with fresh initialization/training/DEV/action roots, same class/objectives/budgets
and unchanged screen. Retain any failure/partial; no seed replacement. No optimizer,
area-feature or architecture change is bundled. Final streams remain untouched;
see the [M1B completion checklist](2026-10-21-continuous-m1b-plan.md).
