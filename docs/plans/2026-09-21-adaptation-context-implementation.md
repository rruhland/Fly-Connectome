# Adaptation Context Implementation Plan

> Use superpowers:executing-plans for inline execution, as requested in this task.

**Goal:** Test the approved adaptation-dependent expression and matching local credit.
**Architecture:** Experimental subclass of area-matched currents; only returned visual
forecast changes. FramePrediction captures sparse eligibility times issue-time gain.
**Tech stack:** Existing Python/PyTorch reference runner, NumPy, pytest.
**Spec:** [Approved proposal](2026-09-21-adaptation-context-proposal.md).

- [x] Add failing gain, physical invariance/reconstruction, and issue-time credit
  tests in tests/test_adaptation_context.py. Implement scripts/adaptation_context.py;
  run focused tests and resolve failures before integration.
- [x] Register only opt-in variants in scripts/controlled_visual.py and reject
  incompatible tick supervision. Add integration regression; run full suite.
  Commit the working experiment implementation before long runs.
- [ ] Add staged scripts/adaptation_experiment.py. Train A and B on the saved
  mixed schedule seed9060 from identical initial weights for200 trials each.
  Save weights, physical stability, update sums, runtime and allocation estimates.
- [ ] Frozen paired evaluation at2/3/4/6,15 trials seed9073: each candidate's
  initial/trained predictions, uncoupled initial/trained, zero and persistence.
  Assert matching targets and immutable evaluation weights; save each result.
- [ ] Apply unchanged criteria per tempo and baseline comparison. Confirm only
  qualifying candidates on30 trials/tempo seed9074. If none qualify, stop both
  orientations without tuning or extending training.
- [ ] Record findings, all metrics, hashes and limits; update implementation
  ledger and push. No fallback architecture is implemented without its proposal.

Ruling: evaluate the approved expression as baseline P plus (g_E-1)E+(g_I-1)I.
This is algebraically the same formula and preserves baseline floating-point
accumulation exactly at unit gains. Directly summing g_E E+g_I I introduced a
2.98e-8 identity-control discrepancy. The independent signed reconstruction
still passes within floating-point tolerance; physical dynamics remain bit-exact.

Verification: eight focused tests pass; full suite 243 passed, four skipped.
Real-crop preflight matches frozen spikes/targets and neutral-gain learning
bit-for-bit. Independent implementation review found no correctness blockers;
its request for learning-path runtime and temporary-tensor profiling is addressed
by the separate benchmark stage (zero rates hold activity fixed for comparison).
