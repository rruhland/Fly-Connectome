# Short-term Synaptic Context Implementation Plan

> Use superpowers:executing-plans inline; ongoing user authorization covers chunks.

**Goal:** Execute the approved D/F experiment with unchanged topology and local learning.
**Architecture:** Add an environment-aware arrival-impulse hook with baseline identity.
Experimental subclasses store compact predictive-edge release state and last-arrival
ticks, updated only at arrivals. Reuse cached impulses in physics and eligibility.
**Tech stack:** Existing Python/PyTorch reference, NumPy and pytest.
**Spec:** [Approved design](2026-09-21-short-term-synaptic-context-proposal.md).

Constraints: D utilization .5, F utilization .5, recovery .100s; existing area-matched
20/5ms waveforms, fixed signs/delays/edges, frame horizon8, no backprop. No new gate.
Review focus: environment identity, duplicate queries, silent recovery, old trace
rescaling, and changed observation targets. Each receives a test or preflight check.

- [ ] Write failing tests in tests/test_short_term.py for analytic paired impulses,
  bursts/recovery, independently reconstructed current/eligibility, private batch
  state, repeated queries, and exact unit-release learning. Implement
  scripts/short_term.py and default Network.visual_arrival_impulse(env,edges).
  Route dynamics, signed_kinetics and Plasticity arriving credit through this hook.
  Formula test: second D gain=1-.5*exp(-interval/.1); second F=1+.5*exp(-interval/.1).
  Run focused tests, then full suite; commit verified implementation.
- [ ] Register opt-in short-term-D-v1/F-v1 in controlled_visual; frame supervision
  only. Add staged scripts/short_term_experiment.py and real-crop preflight5trials
  pertempo2/3/4/6 seed9080. Save gains/eventquiet state, spikes, targets, finite/bounds,
  and runtime. Compare observation targets to baseline; mismatch stops all training.
- [ ] If preflight passes, two200trial training runs with saved schedule9060 and
  original weights. Frozen15trial eval seed9081 at2/3/4/6; owninitial, trainedbaseline,
  zero/persistence controls. Assert identical observations and frozen weights.
- [ ] Apply unchanged gates; confirm qualifiers only with30trials seed9082 and
  unit-release ablation. If none qualify, stop without tuning. If preflight fails,
  document that failure and do not execute dependent training/evaluation.
- [ ] Fresh code review, relevant final checks, findings/metrics/hashes and ledger;
  commit/push on existing codex/milestone-1. M1A remains blocked absent actual pass.

Ruling: continue the existing milestone branch and data workspace, as explicitly
requested. The small default hook is needed because release is private per
environment; the prior edge-only impulse interface cannot represent that identity.
Baseline hook delegates directly to visual_impulse, preserving default arithmetic.
