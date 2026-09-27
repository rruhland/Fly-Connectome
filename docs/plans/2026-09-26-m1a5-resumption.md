# M1A.5 restart point after the bounded sensory/state batch

## Newer checkpoint: generic state and direct dynamics

**Superseding streaming checkpoint:** see
[streaming candidate findings](../experiments/2026-09-26-streaming-candidate-findings.md).
The 2x2, competence and entity-local comparisons have now finished. Retain the
statistical observation state; all point-forecast alternatives still fail at
least one acceptance gate. Stop that chain. The next planned investigation is
a learned **distribution of future spatial state**, with proper scoring and
calibrated controls; do not silently replace the failed point gate with a pass.
M1A.5 is not complete, and production remains unchanged.
Joint context-dependent covariance now retains both relations at 16/16 in all
rotations, and a causal streaming/checkpoint interface exists. However, the fresh
transfer gate rejects the stored-displacement forecaster even after broader
online experience. Do not call it complete. The next active run is
`scripts/run_generative_state_factorial.py`: a registered 2x2 comparison of local
linear temporal maps and statistical observation uncertainty. Preserve the
successful contextual memory; do not tune the rejected prototype forecaster.
Results, checkpoints and failed/provisional variants are retained. Production is
unchanged; continue autonomously until a real acceptance candidate is ready.

The older record below is preserved as experiment history. The current candidate
is the opt-in `FrameObservationState` with learned `PatchAssociation`, plus
separate horizon-specific `LocalMotionDynamics` readouts. It explicitly requires
current visible frames plus events. M1A.5 remains unaccepted and production is
unchanged. See [the consolidated findings](../experiments/2026-09-26-generic-state-dynamics-findings.md).

Verified: rotated spatial-context 16/16, unrelated appearance-context 14/16,
camera crossing/noise/outage identities 36/36, and direct eight-frame mean
forecast error 3.10 pixels versus 8.53 for the strongest fixed control. Constant
motion remains worse than its fixed control. Event-only noisy sensing still fails.
One-step diversity and linear-correlation branches are closed, not tuning targets.

Next: streaming interface with explicit observation/hypothesis distinction,
calibrated observable-outcome uncertainty, bounded state and timing, broader
camera transfer, measured-connectome contribution/ablation, and final forecast
regression assessment. No production promotion without the user's review.
The latest full suite passed 671 tests with four skips; the subsequently added
linear-dynamics test and both prefix-causality tests also passed.

## Status and preserved result

M1A.5 is **not complete** and nothing is promoted to production. The strongest
opt-in state is `EvidenceContextMemory(representation='surface')` with the
original causal event-only observer teacher. It learns the benchmark's context
association, passes all clean localization/regression gates, and improves clean
continuing reacquisition from 10/16 to 16/16 while suppressing all 16 vanished
targets. Familiar/heavy-noise context localization remains 5/16 and 3/16, below
the registered 12/16 and 8/16 minimum.

Current visible-image anchors, computed without hidden/background metadata,
rescue 16/16 under both noise levels. Sparse anchors every eight frames reach
9/16 and 7/16. This is a useful causal intervention: accurate appearance makes
the unchanged local association useful. It is not event-only acceptance and is
not a generic learned representation result. The context comparator and
position/velocity tracklets are still engineered and specialized.

## Completed bounded hypotheses; do not repeat or tune

1. Weighted event-density association beam: failed clean and noisy joint gate.
2. Reconstructed-surface association beam: clean-state improvement; noise fails.
3. Sensor-information intervention: dense visible appearance rescues the state;
   sparse images remain insufficient at the registered cadence.
4. Delayed polarity-return observer teacher: rejected; clean state collapses,
   no context credit acquired. Do not sweep return windows.
5. Three-level contrast-transition belief: no improvement on registered scores.
   Do not sweep its priors or thresholds.

Protocols, scripts, tests and result JSONs are under the corresponding
`2026-09-26-evidence-state`, `sensor-budget`, `polarity-return`, and
`contrast-belief` names. No hidden/future targets enter inference or training.
The sensor-budget intervention explicitly supplies extra visible images.

## Reassessment and next testable alternatives

Stop treating this as a context-plasticity or forecast-head failure. The current
benchmark local rule works when appearance is reliable. Before another state
architecture, compare two substantially different **observation-model** routes:

- **Joint temporal reconstruction:** infer a spatial visual surface from a
  short event history plus an explicit corruption model, retaining uncertainty
  about missed changes. Learn local patch statistics from observed data, with
  no object templates, motion labels, hidden masks or backprop. Test against
  both current event-only observation and its frozen/unlearned counterpart.
  Unlike a scalar event gate, inference must use joint evidence across time.
- **Declared hybrid sensing:** retain sparse visible images as real sensor
  input and use them for online self-supervision of reconstruction between
  arrivals. No every-frame renderer teacher. Compare under the same fixed
  eight-frame sensor cadence and include corrupted images. This requires an
  explicit sensor contract before production promotion; experiments are already
  authorized. Do not confuse success here with event-only success.

Register one decisive configuration per route. First score reconstruction and
the unchanged state together: a prettier image or improved event precision is
insufficient if clean/noisy state or onset/static-cue retention collapses. No
post-result filter/window/cadence sweeps. Use fresh held-out noise seeds for any
new acceptance claim; this small repeated suite is now development evidence.

If one route clears the fixed screen, broaden **before promotion**: rotation,
unrelated visual contextual relations, independently moving entities with actual
identity scoring, continuous contrast/illumination changes, camera-domain
transfer, uncertainty calibration and streaming throughput. Then freeze the
accepted state and attach a separate local/no-backprop forecast learner against
persistence/tracking/sensory controls. Run measured T4/T5 ablation to establish
the connectome's contribution rather than presuming it. These acceptance items
remain outstanding; the positive dense-image control does not waive them.

## Verification and execution

At the end of the implementation batch: 653 tests passed, 4 skipped. A separate
read-only reviewer found no blocking code or target-leakage issue. The reviewer
did not independently rerun experiments or establish broad generalization.
All new mechanisms are in `scripts/`; production architecture is unchanged.
