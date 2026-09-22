# STP learning under corrected target comparison

Use the existing executing-plans workflow inline. User approved the
[comparison amendment](2026-09-21-short-term-target-comparison-amendment.md).
No release dynamics or local-learning changes.

- [x] Test that comparison accepts internal changes but rejects injected/scored
  changes. Implement scripts/short_term_learning.py using the existing FramePrediction
  runner. Check saved preflight hashes/neutral controls and target reconstruction;
  preserve the historical failed decision and write a separate amended decision.
- [x] Train D/F from original crop for200trials on exact seed9060 schedule.
  Save weights/traces/stability/schedule hash. Commit runner before long runs.
- [x] Frozen initial/trained baseline/D/F15trials each2/3/4/6 seed9081; all-node
  targets recorded only for comparison/reconstruction, metrics scored solely L3.
  Verify every injected/scored target against baseline and reconstruct internal
  observations from actual fixed-weight feedforward arrivals. Save each condition.
- [x] Apply unchanged pass criteria and held3baseline-MSE comparison. Only qualifiers
  get30trials seed9082 and unit-release ablation through warmup/evaluation. No tuning.
- [x] Review, full tests, findings/metrics/hashes/ledger; commit/push valid state.

The two functions shared by runs and tests are compare_targets(candidate,reference,
injected,target), returning the descriptive count of changed internal entries, and
the existing observation rule; no target substitution or artificial teacher.

Review found unchecked cached-row reuse. Fixed before evaluation: validate raw
artifact, current weight and stimulus hashes, then recheck sensory/scored targets
and reconstruct internal observations even for cached rows. Five regression cases
cover valid reuse and changed weights/stimulus/artifacts/missing files; missing
helper failed first, all eight comparison tests now pass. Fresh training path had
no review blockers. Full fix verification:269passed4skipped.

D/F training finished in417.61/415.66s (concurrent jobs; includes compression,
excludes warmup/loading). Both11158frames, exact same stimulus hash and200trial
schedule; finite states, bounded weights and release states. Artifacts saved in
runs/short-term-v1. All 24 frozen conditions and verified resume completed. Neither candidate qualifies; no confirmation or ablation was triggered. See ../experiments/2026-09-21-short-term-learning-findings.md.
