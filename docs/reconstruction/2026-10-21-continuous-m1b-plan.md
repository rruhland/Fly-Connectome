# Continuous M1B execution: authority and completion checklist

The user authorized continuous iteration on 2026-10-21 until M1B is complete or
they stop the work. This supersedes earlier allocation-level pauses. Each
experiment still has a concrete finite protocol, independent review/audit and
preserved negative/partial outcomes. Adapt the next experiment from evidence;
do not stop the overall task after one failure or run an unbounded sweep.

No backprop, no privileged policy inputs, no oracle-fitted initialization or
teacher labels in candidate learning. Declared own-hit scalar reward is a
return-learning objective, not original score-only Pong competence. Major
architecture changes require the user's review; ordinary bounded experiments
and smaller implementation changes remain authorized. Final evaluation streams
are reserved for prospectively frozen qualified candidates, never adaptive tuning.

## What must be distinguished

The [current roadmap](../plans/2026-09-27-m1a5-to-m1b-roadmap.md) describes a broader
milestone than any single Pong screen. Freeze task-specific acceptance and scope
decisions prospectively with the orchestrator; never call a successful proxy or
passing unit tests full M1B completion.

- [x] Generic visual/action/reward/elapsed-time boundary, actual true-online SARSA
  with forward-view checks, explicit terminal/truncation semantics and durable
  exact recovery. These establish correctness, not useful control.
- [x] Two-template immediate toy acquisition; delayed toy finite-budget failures
  and later cached conditioning evidence remain separately labeled.
- [x] Two independently trained policies acquired hit-reward Pong RETURNS relative
  to initialization and strongest matched simple controls on fresh development
  serves. The third FULL seed improves but misses the unchanged strict screen.
  Score winning, final confirmation and cross-task retention remain separate.
- [x] Fixed-policy later-camera feedback sensitivity: paired live versus held/missing
  camera interventions benefit the two-policy aggregate. Individual-seed uncertainty
  and OOD limitations remain; this is not forecast-specific necessity.
- [x] One prospectively paired, budget-matched state-only versus state-plus-forecasts
  comparison completed and independently audited: no clear advantage either way.
  This is an inconclusive one-pair ablation, not population inference/equivalence.
- [ ] Reward learning on a substantially different visual interaction task with
  generic transport and declared actuator-specific heads, not Pong-specific labels.
  Native-survival pendulum now has audited positive gains over controls with plastic
  vision; the registered five-second competence condition remains unmet.
- [ ] Enable the existing visual online learner after the frozen-vision control;
  measure drift, retained skill and acquisition with learning enabled. Preserve
  A→B→A and shared-experience versus fresh-policy/fresh-vision distinctions.
  Fixed-policy A→A drift and first plastic B acquisition/old-A probes are audited;
  reused-cohort A behavior is retained empirically without formal noninferiority.
- [ ] Report observations, physical time, wall time, p50/p95 end-to-end/update
  latency, memory and active features. Targets are priorities, not hard vetoes on
  stronger learning; optimize equivalent execution without weakening recovery.
- [ ] Present the roadmap's scoped central-representation comparison for review:
  direct state, bounded anatomical recurrent arm and matched randomized control,
  same reward/readout budget, independent seeds and material measured benefit.
  Obtain user review before genuinely major architecture implementation. Keep a
  failed connectome optional; record an explicit scope decision rather than imply
  that a hit-only direct controller completed this stage.
- [ ] Add any Stage B3 capability only when a specific missing-action/history/
  uncertainty mechanism is demonstrated. Passive forecasts are not a learned
  action-conditioned simulator. Do not start a large latent-model rewrite by habit.

## Current evidence and next experiment

The [first curriculum](2026-10-21-first-return-curriculum-results.md) and
[independent whole-procedure replica](2026-10-21-first-return-replication-results.md)
both pass the unchanged return screen. Fresh initial → final first returns were
14/64 →41/64 and6/64 →39/64; final all-return fractions63/114 and64/122.
All physics/optimizer/phase/statistics and actual zero-work recovery checks passed.
This is two-seed reward-based return acquisition, not three-seed/final evidence,
score winning or complete M1B. Incremental second-stage benefit is inconclusive
in the replica; whole-procedure gains do not prove either stage necessary.

The [feedback test](2026-10-21-camera-feedback-results.md) is complete: equal-policy
first-return gains12.50pp over held and missing cameras, both paired aggregate CI
lower bounds positive. Seed1 missing effect remains individually inconclusive;
OOD limitations and lower live performance on this fresh cohort stay explicit.
This establishes later-camera benefit for the fixed tested set, not explicit
forecast benefit or an optimal-controller necessity theorem.

The [forecast comparison](2026-10-21-forecast-comparison-results.md) is complete:
FULL−state-only first/all lifts −4.69/+4.13 pp, both paired CIs cross zero. Both
arms improve over their own initial readouts; neither passes the unchanged
strongest-control gate. FULL's all-return point lift is exactly19.9369342 pp,
so it remains a failure without rounding tolerance. No extra forecast sweep.

The original Pong heads/transforms remain retained. Runtime preflights exercise
real bank growth; all preflight observations stay separate from candidate data.
Score winning, stable second-task competence, final-stream evidence and the scoped
B2 representation review remain outstanding; user review is required for genuinely
major architecture.

The [fixed-policy online-vision study](2026-10-21-online-vision-drift-results.md)
is complete: mutable-bank integration and exact recovery work; paired skill effects
are inconclusive without noninferiority proof. Learned banks increase inference
cost. The [plastic B acquisition study](2026-10-21-pendulum-acquisition-results.md)
is now complete: survival means2.734/2.812s versus1.343s best simple control, both
paired lift CIs positive; only8/64 and9/64 reach5s, so strict competence fails.
Fixed curves improve; A behavior survives the20k B exposure on reused development
serves, with no registered noninferiority margin. No policy-head transfer claim.
The [ONE frozen.35 transfer study](2026-10-21-pong-score-transfer-results.md) is
complete:14/16 wins versus12 best control; both registered score screens fail,
while return behavior persists. Opponent changes also alter returned trajectories,
so absent transfer does not isolate reward misalignment. The subsequent
[B continuation](2026-10-21-pendulum-continuation-results.md) is complete: +20k
updates with plastic/frozen vision did not improve full 64 mean survival over the
shared 20k head. ON cap success fell 7→1/64; ON−OFF is not a clear plasticity effect
or equivalence. A behavior changed modestly; no formal retention guarantee.
The [native win-only adaptation](2026-10-21-native-score-search-results.md) is
complete and audited: endpoint 18/64 wins versus starting head 16/64, paired
lift 3.125 pp CI [−6.25,12.50]; strict screen failed, no established incremental wins.
Five complete updates used 165,212 fitness rows; total training 199,853 includes
20,588 unused rows and 14,053 diagnostic anchors. No partial update or best policy
selection. Usable reward contrasts did not establish fresh endpoint improvement.

The [epsilon .02 B continuation](2026-10-21-pendulum-low-exploration-results.md)
is complete and audited. It did not establish further learning or superiority to
the reused .2 control: 2.502 versus 2.758 s, paired mean −0.255 s
CI [−0.654,0.141], with only4/64 versus6/64 five-second balances. Retain the
nonmonotonic fixed curves without selecting a midpoint. No epsilon grid follows.
The [native-win executed-child allocation](2026-10-21-validated-win-children-protocol.md)
is now live from audited theta5, source6df746f in its isolated checkout, with
160k training transitions and30/20/50min training/evaluation/overall limits.
It uses separate fresh training validation at fixed .35 and retains only the last
committed incumbent; false acceptances and partial-cycle costs remain explicit.
A possible later B complete-episode survival actor search
keeps visual memory immutable to test genuine native-reward optimization within
the same greedy class; it remains external/unallocated and cannot prove online
visual competence or isolate optimizer superiority across unequal experience.
Stable B, score-winning and final/B2 evidence remain open. No final-root expansion.
