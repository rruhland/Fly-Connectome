# Promoted M1A.5 Vision Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce promoted online visual-service latency while retaining exact outputs and learning on deterministic CPU streams.

**Architecture:** Batch independent scalar interval searches by equal mixture size. Reuse distributions and per-sample calibration values already computed in `StreamingVisualState.step`, then profile and optimize the remaining measured cost without changing inference or credit rules.

**Tech Stack:** Python 3.11, PyTorch 2.5, pytest.

**Spec:** [Promoted vision performance design](../specs/2026-09-27-promoted-vision-performance-design.md)

## Global Constraints

- Keep the promoted `load_default()` algorithm, checkpoint version, and public output schema.
- Preserve all forecast modes, marginal intervals, online updates, identities, and sparse interface values.
- Benchmark `step(learn=True)` plus `encode` at 64x64 with one CPU thread and two/eight visible entities.
- Leave the legacy measured-graph Pong trainer outside this work.

## Review Focus

- Mixtures with one to 32 components must produce the same marginal interval as scalar bisection.
- Calibrated probability endpoints zero and one must retain infinite interval endpoints.
- Missing frames must preserve cached forecasts and censor endpoint credit.
- Online bank and rank updates must match the scalar reference, including after checkpoint reload.
- Eight-entity transport must preserve every mode and sparse feature value.

---

### Task 1: Batched exact interval calculation

**Files:**
- Modify: `src/fly_connectome/vision/dynamics.py`
- Test: `tests/test_vision_performance_parity.py`

**Interfaces:**
- Consumes: `mixture_quantile(centers, weights, probability)`.
- Produces: `batched_mixture_quantiles(centers, weights, probabilities)`, where centers are `[B,N,2]`, weights `[B,N]`, probabilities `[B,2,2]`, and the result `[B,2,2]`.

- [x] Write a test comparing each batch result with two independent scalar `mixture_quantile` calls for singleton and 32-component mixtures, varying batch size and including probability bounds.
- [x] Run the test and observe the missing function failure.
- [x] Implement the helper by nesting `torch.vmap` over the existing scalar routine.
- [x] Run the focused test and existing spatial-belief tests.
- [x] Commit the tested change.

### Task 2: Reuse issued distributions and batch forecast export

**Files:**
- Modify: `src/fly_connectome/vision/state.py`
- Test: `tests/test_vision_performance_parity.py`

**Interfaces:**
- Consumes: `batched_mixture_quantiles` from Task 1 and the existing `predict_with_credit` result.
- Produces: unchanged public `ProbabilisticVisualState.step` output and saved learning state.

- [x] Write an online scalar-reference test comparing complete public results, transport, learned banks, and calibration over a multi-entity stream with missing frames and a reload.
- [x] Run it against a deliberately performance-bounded expectation to observe failure of the current repeated export path.
- [x] Retain per-sample issued `(centers, weights)` privately, compute horizon summaries/bounds once per sample, and batch active intervals by mixture component count.
- [x] Run focused parity tests and the full suite.
- [x] Rerun the same two/eight-entity benchmark and report medians and p95 against baseline.
- [x] Commit the tested change.

### Task 3: Reprofile and finish measured safe improvements

**Files:**
- Modify only the remaining measured hotspot in `src/fly_connectome/vision/` if its output can be preserved.
- Test: the focused parity test for that hotspot.
- Record: `docs/experiments/2026-09-27-promoted-vision-performance.md`.

**Interfaces:**
- Consumes: Task 2's unchanged public service.
- Produces: evidence-backed final latency and parity report.

- [x] Profile the optimized eight-entity path and identify the largest remaining avoidable execution cost.
- [x] Add a failing parity/performance-path test for one concrete change, implement it, and rerun focused and full tests. If no safe, material change remains, record that finding instead.
- [x] Repeat both benchmark arms without concurrent test work; report measurement limits and unachieved targets.
- [x] Commit the source, tests, and measurement report.
