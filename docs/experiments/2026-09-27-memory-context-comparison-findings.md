# Predictive memory and temporal context comparison

This is an opt-in experiment, not a production promotion. The approved visual
checkpoint, observer, association, context memory and predictive runtime are unchanged.
See the [protocol](2026-09-27-memory-context-comparison-protocol.md) and
[raw results](2026-09-27-memory-context-comparison-results.json).

## Five-seed results

All reservoir curves, including fresh controls and return-A learning, reproduce the
preceding audit exactly. Run completed in 955.30 seconds. Each segment has 2,000
samples; lower NLL is better. Mean NLL combines horizons 4/8, then seeds.

| Arm | Changed-B final NLL | A before B | A immediately after B | Mean A degradation | Retention passes |
|---|---:|---:|---:|---:|---:|
| Reservoir | 3.221 | 3.939 | 4.135 | +0.196 | 4/5 |
| Reservoir + recent 512 | 3.088 | 3.869 | 4.250 | +0.381 | 0/5 |
| Append all new examples | 3.202 | 3.921 | 4.046 | +0.125 | 4/5 |

Changed-B samples to the registered target:

- Reservoir: `[1000,1500,1000,1000,1000]`; fresh: `[1000,1000,1000,1000,1000]`.
- Recent+stable: `[500,500,500,500,500]`; fresh: the same. This is faster adaptation
  from the mechanism, **not** a demonstrated acquisition benefit from prior A learning.
- Append: `[1000,1500,1000,1000,1000]`; fresh: `[1000,1000,1000,1000,1000]`.

Append retention deltas are `[0.0694,0.2286,0.1552,0.1112,0.0595]` nat. It repairs
the previously worst seed (0.3474 -> 0.1112), but seed 1 fails (0.2286). It improves
final B NLL in all five seeds relative to the reservoir, and improves absolute
post-B A NLL in four. This is a useful improvement, not an all-seed retention pass.

Shared-dynamics B remains beneficial: final online/fresh NLL is 3.704/3.991 for
reservoir, 3.656/3.916 for recent+stable, and 3.656/3.970 for append. All arms improve
A after shared B in all seeds. The acquisition-speed target remains censored for
most seeds. After 2,000 return-A training samples following changed B, mean A NLL
is 3.965/3.888/3.949 respectively. Immediate retention loss is not irreversible
loss of the ability to relearn.

After A and B, append stores about 4,648/3,907/2,339 examples for horizons 1/4/8
(seed 0), versus 2,048 per horizon for reservoir and 2,560 including duplicates
for recent+stable. No capacity-matched arm isolates pure recency from additional
capacity; claims are about these complete policies, not that mechanism in isolation.

## Observed-history collision screen

Counts below sum seed-specific key groups; they are not independent statistical
trials. Same eligible origin samples are used for four/eight-step raw histories.
Overlap is sparse, especially for longer histories.

| Raw history | Horizon | Shared A/B keys | Mean-offset gaps >2 pixels | Largest gap |
|---|---:|---:|---:|---:|
| 4 steps | 4 | 119 | 18 | 4.00 px |
| 4 steps | 8 | 34 | 16 | 10.00 px |
| 8 steps | 4 | 11 | 1 | 2.24 px |
| 8 steps | 8 | 3 | 0 | 1.41 px |

There are 2,998 eligible A samples at horizon 4 and 998 at horizon 8. Four-step
shared keys cover only 146/37 A samples respectively; eight-step keys cover 11/3.
Thus exact collisions demonstrate some ambiguity but cannot explain the whole
aggregate score degradation. Eight steps do not remove every observed conflict.

For the production normalized-four-step key, the respective shared/conflicting
group counts are 319/24 and 70/21. Its largest gaps are 3.33/7.77 **normalized
displacement units**, not pixel errors. Rotation/scale pooling is intentional;
these counts alone do not establish whether that invariance should be changed.

## What this comparison establishes

Adding a recent-experience bank speeds adaptation to changed dynamics, but increases
old-domain degradation. Preserving the existing prior and every new endpoint is a
useful alternative, but it does not eliminate interference. In particular, retention
can fail in the append-all arm even though no examples are deleted during the audit.
New examples change the nearest-neighbor set and its predictive mixture. Replacement
is therefore not necessary for interference; this does not prove that replacement
never contributes or that all interference is caused by insufficient history.

Some indistinguishable four-step observed histories have different future offsets.
That makes a deterministic answer from those histories insufficient in those cases;
it does not make probabilistic prediction impossible. A correct distribution may
need multiple modes. The diagnostic does not establish how much of aggregate NLL
degradation those exact collisions explain. Eight-step keys have much less overlap,
but uniqueness in finite data is not evidence of learned generalization. Longer
history must be tested as a predictive input, not accepted from a grouping statistic.

## Decision

Do not promote recent+stable pooling as the retention fix. Preserve its fast-adaptation
result as a useful tradeoff. Keep append-all as a promising experimental control:
do not reject it for memory/compute cost, but do not label it a complete continual
learning solution either.

The next architectural experiment should change the information used for retrieval,
not sweep bank sizes or replay ratios. Test an eight-step observed-motion context
alongside the current four-step model, with four-step fallback when longer history
is unavailable. Use the same normalization and outcome learning initially so the
effect of added history is identifiable. Generate matched training histories from
actual camera observations; do not invent eight-step keys for the existing four-step
checkpoint. Compare on held-out shape/position/speed streams and A -> B -> A, with
forecast availability, immediate retention and return-learning measured separately.
Keep the current observer and generic interface. No new latent neural world model,
game-specific state, domain label, or backpropagation is justified by these results.

This recommendation is an experiment, not approval to change the production
architecture. The remaining continual-learning gate is still open.

## Verification and limits

Full suite: **710 passed, 4 skipped**. Tests check unchanged stable admission,
causally issued calibration credit, bounded recent memory, no deletion in append
memory, normalized mixture weights and a constructed observed-history collision.
Independent review found no causal/arm-construction defect and requested the now
documented distinction between normalized and pixel units.

Only generic synthetic single-mover dynamics are tested here. Prior context-relation
retention results are not a guarantee of indefinite context memory. Runtime includes
probe evaluation and, for part of this run, concurrent test verification; it is not
a live-system latency benchmark. No candidate is rejected merely for its compute cost.
