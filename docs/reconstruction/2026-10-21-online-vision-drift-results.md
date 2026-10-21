# Fixed-policy online visual-learning integration

The existing visual learner ran across episodes with both previously qualified
Pong policies fixed. After 2,000 learning transitions per policy, fresh frozen
probes show no clear benefit or harm: equal-policy AFTER−BEFORE first-return
lift **−1.56 pp [−6.25,2.34]**, all-return lift **−1.75 pp [−7.54,4.05]**.
There was no registered noninferiority margin; intervals crossing zero do not
prove retention, equivalence or intrinsic vision quality. This completes the
bounded integration/drift study, not joint policy acquisition or cross-task B1.

| Fixed policy | BEFORE first | AFTER first | BEFORE all returns | AFTER all returns |
|---|---:|---:|---:|---:|
| Qualified seed 1 | 39/64 | 36/64 | 63/114 | 58/112 |
| Qualified seed 2 | 40/64 | 41/64 | 61/118 | 62/120 |

The paired bootstrap resampled64 shared serves, preserving all four
policy×snapshot cells and equally weighting these two fixed policies. It does
not treat128 policy outcomes as independent serves or infer a training-seed
population effect. Closed-loop exposure trajectories belong to the treatment;
these probes cannot isolate representation drift from its experienced data.

Both separate prequential streams also reached2,000 learning transitions.
Their raw returns were7/14 and5/15 opportunities, descriptive without IID
confidence intervals. The first included two hits in its final unfinished rally;
completed-rally summaries alone would omit them. Exact inherited/current visual
learning counts, scene resets and pending-credit retirement are recorded.

Actual cost: **74,696 transitions,297 initial images,74,993 visual images**,
including8,041 learning images; no actor updates. Overall **1,209.59 s** of the
1,800 s cap; summed concurrent task wall4,140.25 s is separate. No failures or
recovery reexecution occurred during the allocation. Peak worker-process lifetime
RSS was about295 MiB, not an isolated per-job peak. Input readouts/observer/T/cap
and all mutable banks/calibration/private RNG were pinned/checkpointed.

Learned-memory inference is slower: frozen BEFORE/AFTER heartbeat medians
**44.54/58.98 ms**, p95 **66.89/80.36 ms**, on nearly equal33k-transition samples.
The median increases about32%; the20 ms target remains unmet. Two-worker learning
stream medians29.20/32.78 ms use different concurrency and are not directly
comparable with four-worker frozen probes. No durability/hash checks were weakened.

Own verification passed25.42 s: all260 job pins, eight actual zero-camera
recoveries and the original scoring-inclusive outcome eligibility unchanged.
Independent audit passed61.44 s: all74,696 raw physics rows,297 initial images,
learning counters/snapshot inheritance and paired statistics exact. Preflight
and implementation fixtures remain separate diagnostic computation.

Next is one native-survival task with existing plastic vision and a new SARSA
head, comparing the predetermined first policy's **2k exposure-AFTER** snapshot
with a fresh generic prior. Do not select a later snapshot or the better point
estimate. Preserve the Pong head/T and immediately probe A after B before any
A retraining. No task-learning allocation has started; numerical/session tests
and the common score-free camera calibration preflight precede its fixed recipe.

[Protocol](2026-10-21-online-vision-drift-protocol.md) · [Artifacts](/workspace/m1b-experiments/pong-online-vision-drift-20261021-r1/summary.json) · [Independent audit/chart](/workspace/m1b-experiments/online-vision-parent-audit-20261021/online-vision-drift.png)
