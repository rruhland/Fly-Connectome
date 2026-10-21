# Native-survival pendulum: partial acquisition and retained Pong behavior

Both new SARSA heads learned to survive longer with visual learning enabled.
Neither meets the registered five-second competence screen. Prior Pong visual
experience provides no clear B advantage in this one paired training-seed study.
No backpropagation, teacher labels or privileged policy inputs were used.

[Learning and retention figure](/workspace/m1b-experiments/pendulum-parent-audit-20261021/pendulum-acquisition-retention.png).

| Frozen B policy / control | Mean survival | Reached5 s | Lift over strongest control, paired95% CI |
|---|---:|---:|---:|
| Pong-experienced vision |2.734 s|8/64|+1.390 s [1.088,1.698]|
| Fresh generic vision |2.812 s|9/64|+1.469 s [1.187,1.763]|
| No-op (strongest simple)|1.343 s|0/64|—|
| Random1 / random10|1.074 /0.714 s|0/64|—|
| Negative / positive constant|0.404 /0.404 s|0/64|—|

The unchanged screen requires mean>=3 s, at least50% reaching5 s, a>=1 s lift
over the strongest control and a positive paired CI lower bound. Both fail the
first two conditions. Experienced-minus-fresh mean survival is−0.079 s
CI[−0.484,0.344], not evidence for transfer superiority or equivalence.

All fixed learning-curve points use the SAME first32 starts. Initial→5k→10k→20k
means are0.406→1.735→1.804→2.735 s (experienced) and
0.406→1.415→2.148→2.830 s (fresh). These are after-training frozen checkpoints,
not early stopping or best-checkpoint selection. The final competence screen
uses all64 starts. Survival summed across equal camera doses is accounting,
not a learning curve: it is nearly constant per camera.

After B, the original Pong readout/T/cap were tested without A retraining:

| Visual lineage | First returns before→after B | All returns before→after B | All-return effect, paired95% CI |
|---|---:|---:|---:|
| Experienced |36/64→36/64|58/112→65/118|+3.30 pp [−2.75,9.43]|
| Fresh |39/64→42/64|63/114→61/113|−1.28 pp [−10.00,7.44]|

The experienced arm preserves every first-return success/failure on this cohort;
its empirical paired-bootstrap difference is exactly zero. That degenerate sample
interval is not a population guarantee. Fresh first-return effect is+4.69 pp
CI[−4.69,14.06]. These explicitly REUSE the old online-development64 serves and
before-B evidence. There was no registered noninferiority margin, so this is
functional-retention evidence without a formal noninferiority claim.

The [protocol](2026-10-21-pendulum-acquisition-protocol.md) fixed20k cameras per arm,
new zero heads, one common1,024-transition frozen-prior calibration/T, alpha.001,
lambda.995, epsilon.2, physical gamma.95/sec and cap256. Genuine failure is terminal;
the5 s time limit bootstraps and truncates. Visual memory persists across episodes,
including eager next-episode initial images at the final dose boundary. The
predetermined experienced start is seed1's2k Pong-exposure snapshot, not a selected
lineage or4k prequential endpoint. The two conditions share one environment/learner
seed recipe; they are not independent training-seed replications. New heads imply
no policy-head transfer. No B-specific visual-necessity claim is made.

Source **2055109**, archive/manifest/seed/source-bridge pins and all outcomes:
`/workspace/m1b-experiments/pendulum-online-acquisition-20261021-r1`.
Exactly40,000 training and75,941 frozen-evaluation transitions,1,230 initial images,
104,101 encoded images;40,526 new visual-learning images. Training took754.38 s;
evaluation/scoring1,118.25 s; allocation31.21 min plus2.62 s final sealing, within
both phase/overall caps. No failed tasks, numerical aborts or recovery replay.
Endpoint visual snapshots are4.03/2.65 MB. Worker process-lifetime peak RSS353 MiB.
Committed-camera p50/p95: training31.83/49.03 ms, B evaluation41.35/69.00 ms,
A evaluation51.57/77.00 ms; these include integrity/snapshot costs and miss20 ms.

Own verification passed24.80 s:706 completion pins,115,941 ordered record/state/
chain rows, all checkpoint artifacts and nine actual zero-camera recoveries.
Independent audit passed29.76 s: all706 jobs/115,941 physics rows/1,230 initials,
TD arithmetic, maps and B/A statistics. Both preliminary audit-launch environment
issues were auditor-only, preserved, and did not touch candidates. Audit/figure:
`/workspace/m1b-experiments/pendulum-parent-audit-20261021/`.

A separate128-camera+1-initial saved-prefix profile matched every original record,
state and chain hash (9.06 s profiled; zero novel candidate data). Fingerprinting
dominates this frozen-evaluation path; profiler timings are not an uninstrumented
speedup claim. Calibration, unit/runtime/handoff fixtures and audit reexecution are
separate from candidate exposure in the cumulative ledger. The metadata-only test
seed pre-expansion is disclosed; no observations or rewards were inspected and
approved roots/cohort were retained. No final streams were expanded.

Next is the agreed one-shot frozen.35 Pong score-transfer evaluation. A subsequent
prospective B experiment can restart new episodes from the predetermined experienced
20k theta/memory, traces reset, contrasting plastic versus frozen vision with new
paired environment/exploration roots. Improving curves favor further acquisition;
this is not representation impossibility or a reason for an architecture sweep.
