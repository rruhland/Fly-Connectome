# Pendulum continuation: no added mean-survival gain

Another 20k SARSA transitions per arm did not improve mean frozen survival over
the shared experienced 20k starting head. Both retain gains over simple controls,
but neither meets the registered five-second competence screen. Continued visual
plasticity does not have a clear mean effect relative to freezing in this paired
training-root experiment; that is not equivalence.

[Fixed curves and retention figure](/workspace/m1b-experiments/pendulum-onoff-parent-audit-20261021/pendulum-onoff.png).

| Frozen B head | Mean survival | Reached 5 s | Increment over starting head, paired 95% CI |
|---|---:|---:|---:|
| Shared 20k start |2.552 s|7/64|—|
| Visual learning ON, 40k total |2.460 s|1/64|−0.092 s [−0.414,0.228]|
| Visual learning OFF, 40k total |2.440 s|5/64|−0.112 s [−0.506,0.277]|
| No-op, strongest simple |1.334 s|0/64|—|

ON/OFF still exceed the strongest simple by 1.126 s [0.911,1.359] and 1.106 s
[0.838,1.390]. The unchanged screen also requires mean>=3 s and at least 50%
reaching 5 s; both fail. ON's cap fraction falls 7→1/64, a −9.375 pp paired effect
CI [−18.75,−1.56]. OFF's cap change is −3.125 pp [−14.06,7.81]. ON−OFF mean is
+0.020 s [−0.297,0.336]; its cap effect is −6.25 pp [−14.06,0]. Do not attribute
the ON-versus-start loss specifically to visual plasticity from this contrast.

All fixed curves use the same first 32 starts: shared 20k→25k→30k→40k means are
2.265→2.212→2.111→2.510 s (ON) and 2.265→1.854→2.341→2.635 s (OFF). Their
last points improve descriptively on this subset while the full 64 incremental
means do not. The primary endpoint is always 40k; no midpoint is selected.

The original A readout/T/cap received no A retraining. ON's Pong first returns
change 36→34/64, effect −3.125 pp CI [−7.81,0]; all returns 65/118→62/115,
effect −1.17 pp [−8.39,5.93]. These REUSE the old online-development cohort and
before-B evidence. OFF's complete reset visual content/count and full policy/
environment configuration are identical to the prior E20k probe, so its 64 old
trials are explicitly reused with zero new work. There is no formal retention
noninferiority claim or claim of no forgetting.

The [protocol](2026-10-21-pendulum-continuation-protocol.md) fixed the predetermined
experienced 20k theta/memory, new episodes and paired independent environment/
exploration roots, reset traces, alpha .001/lambda .995/epsilon .2/gamma .95 per second
and unchanged T/cap 256. Weights are installed before the first observation/action.
Only visual learning changes between arms. Native survival reward and genuine
failure/time-limit bootstrap semantics remain unchanged. Frozen checkpoint skill
is distinct from deployed plastic-system competence. No teacher, privileged policy
inputs or backpropagation were used.

Source **51293d2** (integrated unchanged into main as **3c09df9**), all readouts,
snapshots, clocks, raw outcomes and source pins:
`/workspace/m1b-experiments/pendulum-onoff-continuation-20261021-r1`.
Exactly 40,000 new training + 50,043 B-evaluation + 16,497 A-retention transitions;
1,126 initial images, 94,915 encoded images and 20,205 new visual-learning images.
ON cumulative learned images 42,487; OFF remains 22,282. All 706 new jobs completed
in 1,688.17 s (28.14 min): 832.62 s training and 855.56 s evaluation/scoring, within
20/30/50 min caps. No task failures, numerical aborts or recovery replay.

Committed-camera median/p95 latency is 36.18/52.77 ms training, 36.71/57.92 ms B
evaluation and 49.33/68.20 ms A evaluation, including integrity/snapshot costs.
The 20 ms target remains unmet. OFF adds a frozen-memory hash per image; timing
differences must not be interpreted as the cost of visual learning alone. Worker
process-lifetime peak RSS 379 MiB. All preflight/diagnostic exposure remains separate.

Own verification passed 30.61 s: 706 pins, 106,540 ordered record/state/chain rows,
every checkpoint artifact and eight actual zero-work recoveries. Independent
audit passed 19.59 s: all 706 jobs/106,540 physics rows/1,126 initials, TD arithmetic,
mapped heads, paired statistics and full OFF reuse proof. Audit/figure:
`/workspace/m1b-experiments/pendulum-onoff-parent-audit-20261021/`.
Exec 97305/PID 94283, both training workers, evaluation workers and audits exited 0;
source is released. No final streams were expanded.

Next is the reviewed warm-start native-score search at fixed .35 using the first
qualified Pong lineage, not another frozen opponent sweep. B's plateau motivates
one prospective learning-rule/exploration decision from the complete curves and
failure behavior; it does not establish representation impossibility. Stable B,
native score acquisition, final evidence and B2 scope review remain outstanding.
