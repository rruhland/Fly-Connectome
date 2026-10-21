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

The [first-return curriculum](2026-10-21-first-return-curriculum-results.md)
passed its audited single-seed screen: initial14/64 →phase0 29/64 →final41/64 first
returns, final63/114 all returns, both stage contrasts positive. It is a component
acquisition result; the reproducibility checkbox above remains open. No winning,
forecast-benefit or ongoing visual-feedback claim follows from blanking.

Next is ONE independent from-scratch replication of the entire fixed120k/80k
procedure, unchanged rule/features/objectives/budgets and screen, with new optimizer,
training, DEV and action roots. Retain every negative/partial outcome; no seed
replacement or best-stage selection. Source parameterization must preserve the
original recipe and derive phase carry from each seed's own frozen state. Further
independent seeds/final acceptance remain prospective; final streams are untouched.
