# Eight-step integration: useful predictor, blocked promotion

Historical failure report. The subsequent user-authorized consensus repair and
promotion are recorded in [these findings](2026-09-27-context-consensus-findings.md).
The failed run and its data remain preserved here; reproduce its original code
at commit `ca94064`. The current integration runner validates the repaired rule.

The user approved integration and promotion, including the earlier two-seed
retention exception. The installed implementation and version-2 checkpoints are
complete. **The production default remains version 1:** the active online run
exposed a different, unaccepted outage-support regression.

## Implemented and verified

`ProbabilisticVisualState.upgrade_temporal_context()` upgrades a version-1 model
in place and resets the scene. Existing four-step examples initialize the short
append-all bank; the eight-step bank starts empty. Nine consecutive actual
positions and at least 32 long-bank outcomes enable the specialist. Missing
observations break the contiguous history. Both banks learn only actual observed
endpoints; calibration uses the distribution issued before the outcome arrived.

Version-2 saves both banks, their state, and calibration. Version-1 load/save
behavior remains exact. The bundled prior is unchanged. Tests match the approved
experimental implementation exactly, including active camera learning, noise,
multiple entities, outage caches, scene reset and checkpoint continuation.
The isolated wheel runs camera inference and version-2 save/load without research
imports. Independent review identified and closed a missing cached-forecast
availability gate; no model/checkpoint defect was identified.
Final repository verification: **717 passed, four skipped**. Focused camera,
checkpoint, legacy compatibility and CLI checks also pass (11 tests).

## Decisive active-camera run

Fixed before testing: 192 training scenes with seed 291027, six generic motion
families, dot/square training shapes, actual noisy camera observations only.
No hidden positions enter learning, no held-out selection, no parameter tuning.
The restored candidate has 1,712 / 1,136 / 369 long-bank outcomes at horizons
1 / 4 / 8. Evaluation uses the existing 216-scene transfer split.

| Check | Active trained candidate |
|---|---|
| Forecast NLL, horizons 4 / 8 | 2.485 / 3.009 |
| Nominal 90% marginal coverage | 93.54% / 89.85% |
| Aggregate and each-family fixed-control gates | Pass |
| Missing forecasts among 6,264 eligible targets | 0 |
| Frozen evaluation parameters | Unchanged |
| Spatial / appearance contextual relations | 64/64 each |
| Noisy / outage identity retention | 36/36 each; zero final extras |
| Cached outage forecasts | 900 evaluations; zero missing |
| Outage spatial support | **29/180; fail** (required >=95%) |
| Matched-prefix ambiguity and visible contradiction | Pass |
| Confident phantom entities in 600 noisy blank frames | 0 |

These forecast results pass the original distributional contract. They do not
establish a point-forecast contract, joint/conditional coverage, or lifelong
retention. The run includes additional online training, so its improvement over
the original frozen checkpoint is not a matched-training architecture ablation;
that comparison remains the earlier five-seed experiment.

## One boundary control, then stop

Reload the same trained checkpoint and replace **only** `state.memory` with the
original context prior. Keep all learned short/long dynamics and calibration.
Outage support returns to **180/180**, with 36/36 identities, zero extras, zero
missing forecasts, and exactly the same cached forecast NLL (**1.9248708455**).

This isolates the observed regression to online changes in the separate context
correction memory, rather than the eight-step forecast or its serialization.
That existing memory learns normalized velocity corrections at reappearance and
applies retrieved corrections to carried positions. In this run it changes while
learning ordinary noisy motion. Its maximum stored correction norm grows from
0.421 to 3.162. This is supporting evidence, not proof of the precise admission
or retrieval defect. Both relational tests still pass, so those tests alone were
insufficient to certify safe online integration.

Do not silently freeze context learning or promote the restored-memory diagnostic
as a fix. Keep the eight-step candidate; investigate the existing correction
memory's credit/admission and retrieval on this reproducible failure, then rerun
the same integration gates. This is a specific robustness problem, not a reason
to restart a representation or latent-mechanism search.

## Reproduction and scope

- `python scripts/run_context_integration.py`: trains, saves, reloads, evaluates;
  returns a failure status when any integration gate fails.
- `python scripts/run_context_memory_boundary.py`: the single memory-restoration
  control; requires the integration checkpoint.
- Checkpoint: `checkpoints/m1a5/context-integration.pt` (generated, not bundled).
  SHA256: `bb4cef3f0c3a7ab0c46a4fba1da8d0396def27c86d3394dc126128801ee16d68`.
- Results: `2026-09-27-context-integration-{results,transfer,recovery}.json` and
  `2026-09-27-context-memory-boundary-{results,recovery}.json` beside this report.

Standalone two/eight-entity latency and memory-growth benchmarking is deferred
because promotion is blocked. The replay's 27.3 samples/s ran concurrently with
tests and is not an isolated online performance claim. Append-all storage grows
with experience; no memory pruning or cheaper biological model is introduced.
M1B is not activated, and the current production baseline remains available.
