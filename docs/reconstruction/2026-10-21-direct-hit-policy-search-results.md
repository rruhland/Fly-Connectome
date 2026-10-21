# Direct hit search: positive point estimates, acquisition screen not met

The fixed endpoint returned 16/32 first incoming balls and 21/51 incoming balls
overall, versus its own initial policy's 8/32 and 8/40. It outperformed the simple
controls on both point estimates but **failed the preregistered development
screen**. This is encouraging one-seed evidence, not replicated acquisition,
score-only Pong competence or completed M1B. Final streams remain unused.

## Fixed experiment and fresh evaluation

The [protocol](2026-10-21-direct-hit-policy-search-protocol.md) retained corrected
frozen vision, the exact generic DCT/rank 183 class, cap 256, .65 opponent and hold1.
Eight mirrored directions shared six new ten-second serves per generation.
Fitness was mean total own hits; no teacher, opportunity labels, backpropagation
or policy-input physics. Initialization was random score-free function-RMS .10.
Noise .25, step .10 and all other search settings stayed fixed.

Training stopped at its camera bound: **nine full updates**, 159,873 observations,
of which 145,561 informed completed populations and 14,312 belonged to an unused
partial generation. The last complete checkpoint, theta9, was evaluated; there
was no best-checkpoint selection or partial-population update. The partial draw
and post-draw RNG are retained, with original next global index 10.

All seven frozen cohorts used the same 32 fresh-phase DEV serves:

| Policy | First incoming returns | All returns/opportunities |
| --- | --- | --- |
| Trained endpoint | 16/32 | 21/51 |
| Own restricted initialization | 8/32 | 8/40 |
| Constant down | 9/32 | 10/42 |
| Constant up | 7/32 | 7/39 |
| No-op | 6/32 | 8/40 |
| Random hold1 | 7/32 | 8/40 |
| Random hold10 | 6/32 | 7/39 |

Against the strongest comparator, first-return lift was **+21.875 pp**, paired
95% interval **−3.125 to +31.25**; overall lift was **+17.367 pp**, interval
**+1.772 to +25.359**. The screen required at least 20 pp on both metrics and both
lower bounds above zero; it failed unchanged. Serve-cluster resampling used 10,000
draws and recomputed the comparator maximum inside each draw. No invalid draws.

The endpoint produced four rallies with at least two returns, maximum streak 3,
and .1625 returns/physical second; constant down produced one such rally and
.1082 returns/second. The endpoint had 30 losses and 2 timeouts. Across all cohorts,
the opponent returned 173/173 balls, including 68/68 after own returns. No natural
serve wins occurred. All 32 first-incoming trajectories matched across all seven
policies. Contacts within a rally are not independent trials.

Descriptively, the endpoint retained all nine constant-down first-return successes
and gained seven; against its own initialization it gained thirteen and lost five.
Comparator switching in the maximum-control bootstrap still matters. Neither this
discordance nor a positive overall interval changes the registered decision.

## Training signal and interpretation

All nine updates were nonzero; 60/72 mirrored mean contrasts were nonzero. Completed
populations supplied 329 hits. Rewards reached multiple serves, but some updates
were serve-sensitive: leave-one-serve-out function cosine fell to .227. Rotating
serve sets mean population hit totals are **not a learning curve**.

Endpoint function-RMS was .3319, perturbation/policy ratio .7533; calibration
actions were 414/353/257. These checks do not support constant-policy collapse or
vanished exploration on the calibration set. They do not establish visual
necessity, forecast benefit or Bellman-Q adequacy. The much larger observation
budget also prevents attributing its advantage over earlier SARSA point estimates
solely to the learning rule.

## Integrity and cost

Source **2e20707**, 23 focused passing tests, exact archived source/protocol and
[full artifacts](/workspace/m1b-experiments/pong-direct-hit-policy-search-20261021-r1/summary.json)
are retained. One environment restart killed the workers. Recovery preserved the
original clock/configuration and all completed updates. Journals account for 399
pre-interruption committed observations absent from settled task-cost increments;
99 existing steps were replayed exactly, adding zero novel observations. Interrupted
uncommitted tails and lost task-wall tails are unknown, not claimed zero.

Allocation cost: **159,873 training + 32,467 evaluation observations**, 1,957.55 s
shared wall including restart; 193,513 rendered images including initial frames,
171,736 encoded images, plus separately counted 99 recovery reexecutions. Recorded
parallel task wall totals at least 6,762.80 s; this is neither CPU time nor elapsed
wall. Imports-only GC freezing remained fixed; no deployment speedup is inferred
from comparisons with different earlier learners. Training heartbeat p50/p95/p99
was 29.73/46.84/59.56 ms, excluding restart/chunk boundaries; 20 ms p95 remains unmet.

[Post-exit verification](/workspace/m1b-experiments/pong-direct-hit-policy-search-20261021-r1/completion-verification.json)
performed actual zero-new-work recovery, preserved every one of 192,340 ordered
record/state/chain rows and frozen weights, and verified 1,171 complete semantic
completion pins in 6.25 s. The [independent parent audit](/workspace/m1b-experiments/direct-hit-parent-audit-20261021/summary.json)
passed all 159,873 training and 32,467 evaluation physics rows, reconstructed all
nine optimizer updates and pending-draw accounting, and verified paired geometry
in 152.82 s. Independent bootstrap results matched exactly. This replay supplied
zero new candidate observations.

Continuous iteration remains authorized. The next approved design is a separately
registered larger fresh DEV confirmation of the unchanged theta9 and original
initial policy, with simple controls and secondary blank-camera diagnostic.
There is no new training, threshold revision or final holdout in that confirmation.
The [broader completion checklist](2026-10-21-continuous-m1b-plan.md) remains open.
