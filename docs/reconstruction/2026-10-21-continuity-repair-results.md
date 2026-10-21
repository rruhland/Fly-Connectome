# Visual continuity repaired; fixed SARSA screen remains negative

The generic repair removes a demonstrated fabricated-motion defect. It does not
solve association ambiguity or establish reward learning. The corrected 24k-camera
SARSA endpoint matches constant up on first returns and fails its unchanged
development screen. Continuous iteration remains authorized; the next experiment
directly optimizes hit return in the existing frozen linear policy class.

## Correctness evidence

Exact saved-action replay showed one beam slot alternating between distant
objects under a single public ID. The selected branch's own velocity was modest,
but encoder differences and forecast histories crossed lineages. The repair
preserves the beam, revises routing IDs when the winning parent endpoint/time
differs, clears slot context and retires old forecast credits. Ordinary occlusion
continues under the same ID. Model bytes and transport layout are unchanged;
the service is versioned `+selected-lineage-continuity-v1`.

| Saved failure clip | Before | After |
| --- | --- | --- |
| Independent: max encoded speed / raw norm² | 42.76 / 20,611 | .8735 / 237.83 |
| Sustained: max encoded speed / raw norm² | 43.37 / 21,300 | .8735 / 236.91 |
| Independent: h8 forecast availability | 39.1% | 25.6% |
| Sustained: h8 forecast availability | 48.6% | 19.4% |

Current ball/paddle coverage is unchanged on both failures and the two ordinary
1024-frame clips. Forecasts retained on common frames have **exactly unchanged
errors**: the repair abstains on invalid histories. Lower conditional mean errors
are selection effects, not an accuracy improvement on retained forecasts. The
ordinary clips' ball forecast metrics are unchanged. Exact fixed-T norms/cap
activation remain usable without refitting. Generic continuity, occlusion,
context/credit isolation, routing invariance and recovery gates passed among
44 focused tests. Independent evidence: [fixed-frame validation](/workspace/m1b-experiments/continuity-validation-20261021/after-summary.json)
and [paired forecast check](/workspace/m1b-experiments/continuity-validation-20261021/paired-forecast-summary.json).

## Reward-learning outcome

The [prospective protocol](2026-10-21-continuity-repair-protocol.md) retained seed
736187003, .65 opponent, hit-only scalar reward, frozen exact DCT/cap256,
alpha .001/lambda .995/epsilon .2, hold1/block1 and one SARSA update per camera.
Only visual continuity semantics changed. Training finished 24,000 cameras with
55 hit rewards and finite weights; 144 rallies ended in losses, with no timeout
or win. Hits per successive 4k-camera windows were 8/11/9/8/9/10. No clear growing
contact curve follows. Historical learning comparisons are nonpaired.

Frozen greedy evaluation used 32 new shared ten-second serves per policy:

| Policy | First incoming returns | All returns/opportunities |
| --- | --- | --- |
| Corrected SARSA | 11/32 | 15/45 |
| Own zero initialization / constant down | 6/32 | 6/38 |
| Constant up | 11/32 | 14/45 |
| No-op | 6/32 | 7/38 |
| Random hold1 | 6/32 | 7/38 |
| Random hold10 | 6/32 | 8/38 |

Against the strongest control, first-return lift is **0 pp** (paired 95% interval
−25 to +18.75); all-opportunity lift is **+2.22 pp** (−15.49 to +16.12).
Neither meets the ≥20 pp/both-lower-bounds-positive screen. The learner produced
three rallies with at least two returns, maximum streak three, and .1466 returns
per physical second; constant up also produced three such rallies. The opponent
returned 128/128 evaluated balls, including 56/56 after own returns. No natural
wins occurred. This is one training seed and a finite exposure, not an
asymptotic impossibility result, visual-necessity claim or completed M1B.

## Integrity, cost and next decision

Source **570222e**, [full artifact](/workspace/m1b-experiments/pong-continuity-repaired-sarsa-20261021-r1/summary.json):
24,000 training + 26,467 evaluation cameras, **1,261.56 seconds shared wall**.
Durable heartbeat p50/p95 was 29.83/113.00 ms, excluding restart/chunk-boundary
pairs. This does not meet the 20 ms p95 target. Raw training norm² max 384.41 is
not evidence that association is globally correct; actual policy norms were
bounded by 256 plus roundoff. Final streams remain untouched.

The [independent parent audit](/workspace/m1b-experiments/continuity-repaired-parent-audit-20261021/summary.json)
passed all 193 jobs, 50,467 raw physics/reward rows, 24k exact TD equations and 32
paired first trajectories in 37.92 seconds; both bootstrap results matched
independently. [Zero-work recovery](/workspace/m1b-experiments/pong-continuity-repaired-sarsa-20261021-r1/completion-verification.json)
added zero observations/replay, preserved every training record/state/chain and
verified all 193 completion pins before shared source changes (2.40 seconds).

Diagnostic computation is separate: rejected fused-hash prototypes reexecuted
1,152 old cameras; root-cause tracing reexecuted 23,840 physics/248 visual frames;
repair validation used 31,925 physics/4,670 visual reexecutions. A separate
[GC investigation](/workspace/m1b-experiments/gc-profile-20261021/RESULTS.md)
reexecuted 2,176 old cameras, no novel candidate data. Closure cleanup was slower
and rejected. One imports-only freeze per spawned worker showed 18.6% whole-fixture
reduction and stable sequential-session memory, with enabled GC and exact records/
state/chains. Its measured gain is provisional and its runtime mode will be
explicitly pinned in the next experiment; codec/integrity guarantees stay intact.

The next authorized bounded experiment is six-serve mirrored **hit-return policy
search**, using the existing policy class and no critic, teacher or backprop.
It tests direct greedy-policy learning rather than Bellman-value fitting; high
search dimension, sparse paired contrasts and constant-policy degeneracy remain
explicit risks. Preserve all negative evidence and the broader
[M1B completion checklist](2026-10-21-continuous-m1b-plan.md).
