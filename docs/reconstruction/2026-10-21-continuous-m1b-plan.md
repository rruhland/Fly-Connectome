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
- [ ] Genuine reward-based acquisition on held-out Pong trajectories, relative to
  own initialization and strongest simple controls, reproducible across independent
  training seeds. Return learning is a first component proof; winning/score-task
  transfer and the original opponent's scoring limitation stay explicit.
- [ ] Actual visual-state necessity checks for a successful candidate. State-varying
  calibration scores and privileged fitted-controller witnesses are insufficient.
- [ ] State-only versus state-plus-forecasts comparison with matched budgets and
  frozen prospective criteria. Merely exposing forecasts proves no benefit.
- [ ] Reward learning on a substantially different visual interaction task with
  generic transport and declared actuator-specific heads, not Pong-specific labels.
- [ ] Enable the existing visual online learner after the frozen-vision control;
  measure drift, retained skill and acquisition with learning enabled. Preserve
  A→B→A and shared-experience versus fresh-policy/fresh-vision distinctions.
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

## Current next experiment

The initial policy-search launch was held after exact saved-action replay exposed
a selected-lineage continuity defect: the encoder differenced positions from
different beam hypotheses under one routing ID. A generic, prospectively versioned
repair preserves association hypotheses while retiring discontinuous velocity,
context and forecast credit. Validate it on both failures and ordinary occlusions;
then ran the approved [corrected-vision24k SARSA screen](2026-10-21-continuity-repair-protocol.md)
first. It completed stably but failed fresh controls (11/32 first returns,
15/45 overall versus up11/32 and14/45). The repair suppresses false continuity,
not association ambiguity. Keep the fixed transform and all negative evidence.
The [direct hit-return search](2026-10-21-direct-hit-policy-search-results.md)
completed at source 2e20707: nine updates, fresh first returns 16/32 and all 21/51,
versus strongest control 9/32 and 10/42. Positive point estimates failed the
unchanged development screen. Read HANDOFF and the audited artifacts before
registering the next bounded experiment; retain the original frozen outcome.

Prior score-only search, its continuation and a stationary-opponent curriculum
failed their development screens. Camera-rate hit-SARSA, full action holds and
sustained exploration also failed to beat simple controls. Existing class-capacity
witnesses do not prove adequate Bellman-Q representation. The next hypothesis is
that directly optimizing controllable hit return can learn useful action scores
without fitting Bellman values or waiting for return-to-score reward.

Approved first design: unchanged frozen generic DCT/rank183 class, cap256, hold1,
.65 opponent; random score-free initial function-RMS .10. Keep mirrored search
8 directions/top4, noise .25, step .10. Each generation shares six new fresh-phase
ten-second training serves across all 16 candidates. Fitness is mean own-hit count
only. Maximum 16 complete generations / 160k new training cameras / 35 shared
training minutes / 45 overall, four workers. Last complete generation is the
endpoint, never the best; at least two are needed for a valid endpoint screen.

Endpoint, own initialization and five simple controls share 32 new fresh-phase DEV
serves. First-incoming returns and all-opportunity return fraction are co-primary;
opportunities remain evaluation-only. The strongest-control max includes the own
initial policy. A promising development screen remains ≥20 percentage-point lift
on both metrics with paired serve-cluster 95% lower bounds above zero. This is
not final acceptance or automatic M1B completion. No intermediate DEV selection.

Prelaunch work pinned new seed recipes, preserved the existing transform,
implemented variable-serve generation/fitness and generalized metrics, ran narrow
mapping/update/partial-generation/recovery checks, and archived committed source.
The orchestrator reviewed the launch-ready result. A bounded GC optimization
passed exactness/lifetime checks and was explicitly pinned; no codec change.
After every outcome: audit, update results/ledger/handoff, choose the next bounded
test and continue under the user's authorization.
