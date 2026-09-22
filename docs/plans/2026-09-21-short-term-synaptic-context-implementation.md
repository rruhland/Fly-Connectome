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

- [x] Write failing tests in tests/test_short_term.py for analytic paired impulses,
  bursts/recovery, independently reconstructed current/eligibility, private batch
  state, repeated queries, and exact unit-release learning. Implement
  scripts/short_term.py and default Network.visual_arrival_impulse(env,edges).
  Route dynamics, signed_kinetics and Plasticity arriving credit through this hook.
  Formula test: second D gain=1-.5*exp(-interval/.1); second F=1+.5*exp(-interval/.1).
  Run focused tests, then full suite; commit verified implementation.
- [x] Register opt-in short-term-D-v1/F-v1 in controlled_visual; frame supervision
  only. Add staged scripts/short_term_experiment.py and real-crop preflight5trials
  pertempo2/3/4/6 seed9080. Save gains/eventquiet state, spikes, targets, finite/bounds,
  and runtime. Compare observation targets to baseline; mismatch stops all training.
- [x] Conditional training decision: NOT RUN because preflight failed. If it had
  passed, two200trial training runs with saved schedule9060 and
  original weights. Frozen15trial eval seed9081 at2/3/4/6; owninitial, trainedbaseline,
  zero/persistence controls. Assert identical observations and frozen weights.
- [x] Apply unchanged gates; confirm qualifiers only with30trials seed9082 and
  unit-release ablation. If none qualify, stop without tuning. If preflight fails,
  document that failure and do not execute dependent training/evaluation.
- [x] Fresh code review, relevant final checks, findings/metrics/hashes and ledger;
  commit/push on existing codex/milestone-1. M1A remains blocked absent actual pass.

Ruling: continue the existing milestone branch and data workspace, as explicitly
requested. The small default hook is needed because release is private per
environment; the prior edge-only impulse interface cannot represent that identity.
Baseline hook delegates directly to visual_impulse, preserving default arithmetic.

Outcome: cc97645 implements the core and preflight. Eighteen focused tests pass;
final full suite261passed4skipped. Independent fresh review found no blockers or
minor issues; native/checkpoint support and biological parameter validity remain
outside this experimental claim. Dependent learning stages intentionally not run.

Ruling: the observation gate audits every neuron, reporting scored L3 separately,
because internal observations feed local learning too. All injected targets match,
but D/F internal targets differ at every tempo. The approved stop rule is honored.
Offline reconstruction from emitted spikes, fixed signed feedforward weights and
delays matches downstream targets exactly after the initial history boundary.
This is changed physical observation, not an encoding defect or failed learning.

The all-neuron equality requirement is too broad for physical STP. A separate
target-comparison amendment is proposed, not applied, allowing internal physical
observations while preserving identical external/scored targets. Its approval is
the next decision; do not silently start training under the original stop rule.
