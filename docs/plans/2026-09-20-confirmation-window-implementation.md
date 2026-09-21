# Local confirmation window implementation plan

> Execute inline using superpowers:executing-plans; approval covers routine chunks.

Goal: execute the approved conditional experiment without changing legacy behavior.
Spec: `2026-09-20-local-confirmation-window-proposal.md` (approved by user).
Stack: existing Python/PyTorch CPU reference, pytest, measured local checkpoints.

## Constraints

No backprop, invented edges, sign/delay/dynamics changes or final seeds. Gate A
failure/inconclusiveness stops this plan before production learning changes.
Use the existing `codex/milestone-1` branch and continuation ledger. Keep checkpoints
read-only during Gate A; record provenance and do not replace failed candidates.

## Tasks

- [x] Gate A diagnostic: add `scripts/audit_history_gate.py` and
  `tests/test_history_gate.py`. Test causal polarity/age strata, within-stratum
  shuffling, constant/singleton strata, unique referenced-event counts and exact
  candidate selection. Reconstruct physical traces for only the three fixed
  measured edges. Record frame-end traces, real targets and forecasts during an
  unchanged frozen run; construct future labels only afterward. Retain frames
  [12, steps-12), matching the previous trace diagnostic. For each seed and edge,
  compute correlations/covariance and ten conditional shuffles, counts, all-window
  controls and permutability. Require two pairs to pass on both seeds; report
  fail/inconclusive explicitly. Run tests before the fixed 2000-frame experiment.
- [x] Gate A review/report: inspect independent review and correct errors with
  regression tests. Run full suite, save JSON/report and ledger, commit and push.
  Stop if gate does not pass. No thresholds or candidates may be changed.
- [ ] Only if Gate A passes: refine implementation details for the approved rule
  in `plasticity.py` and a focused window-state module. Add versioned config and
  compatibility checks in training/checkpoint/evaluation interfaces; no legacy
  path refactors. Build hand-calculated signed updates, expiry/first-event order,
  immutable snapshots, zero-weight eligibility, exact resume and reward isolation
  tests first. Reject unsupported experimental backend/batch modes. Complete Gate B
  delayed-association fixtures and full suite before measured training.
- [ ] Only after Gate B: version one B=1 seed-1 run config, train from initial up
  to10000 frames with saves every100 and safety check at2000, then frozen matched
  evaluation. Follow every Gate C baseline/metric and stopping criterion in the
  approved spec. Preserve original M1A gates and all source artifacts.

Review focus: never-seen versus stale histories; future leakage at issuance;
event-count inflation by repeated windows; non-permutable history strata; one
successful seed/pair incorrectly satisfying the across-both-seeds gate.

Outcome: Gate A failed (0/3 pairs pass both seeds; 2 required). The two conditional
implementation/training tasks are not executed under the approved stopping rule.
198 tests passed, four CUDA skips; fresh review found no actionable defects.
