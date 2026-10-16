# First bounded Pong development comparison

This protocol is declared before any Pong competence training. The 256 profiled
diagnostic Pong transitions (seeds 99111/99112) are development diagnostics, not
competence evidence. The separate toy pilot retains its exact original allocation.
Current user authorization allows development experiments; earlier handoff pauses
and toy-only scope no longer govern this continuation.

## Allocation and selection

Use three dispersed training seeds: 172900127, 736187003, 1597334677. Pair each seed
across lambda .8 and .995. Keep alpha=.0001, gamma_per_second=.95, epsilon=.2,
zero initial three-action weights, the current unscaled generic encoder, frozen
vision, 64x64 images, existing 2/2/3/2/3 physics-tick decision cadence, and score-only
terminal rewards. No backprop, shaping, privileged policy inputs, or forecast
benefit claim is permitted. The frozen prior's long dynamics banks are empty, so
its declared long-context mechanism falls back to four steps.

Each arm receives exactly 6000 new learning transitions per seed (36000 total),
with checkpoint weights saved at 0, 2000, and 6000. Each natural score ends a rally;
2400 physics ticks (20 seconds) without a score truncates it. Truncation uses the
actual final camera observation for Sarsa bootstrapping and resets episode traces.
Training allocation does not stop on success. Each execution chunk has <=200 new
transitions and <=180 seconds, including save reserve; restarts use durable replay.

Fresh development evaluation uses seed 904271833 and exactly 12 rallies at each
checkpoint, for every training seed/arm. The zero checkpoint is identical across
all arms and is evaluated once. Uniform random (independent action seed 447201113)
and no-op baselines are also evaluated once on this same 12-serve sequence.
The initial frozen zero-weight policy chooses action index 0; no-op chooses index 1.
Every policy receives the same physical dynamics, score rewards, explicit reset
semantics, max-rally duration, initial camera histories, and serve-seed sequence.
Baseline perception may be skipped only after matching full-vision action, reward,
and reset traces; this optimization changes neither actions nor camera cadence.
Report its timing separately from learned-policy inference cost.

There are 12 learned checkpoint evaluations (2 nonzero checkpoints x 3 seeds x
2 arms), plus 3 shared baselines, totaling 180 development evaluation rallies.
At <=2400 ticks per rally, each consumes <=1200 camera transitions (at least two
physics ticks per transition), giving a conservative evaluation cap of 216000
transitions. This is a hard bound, not an expected cost. Training plus evaluation
has a 90-minute aggregate active wall-clock cap; retain partial allocations if
exhausted, and report actual camera observations, transitions, replay and durations.
Do not silently extend, replace failed seeds, or tune on the final holdout.

## Evidence and interpretation

Primary outcomes are score differential per attempted rally and wins divided by
all 12 attempted rallies, counting truncations as nonwins. Also report completed
rally fraction, win fraction among completed rallies, total return per physical
second, own hits, truncations, per-rally durations and raw outcomes. Report every
training seed and checkpoint, not only the selected candidate. Compare against
the strongest shared simple baseline. Occasional wins, nonzero weights, surviving
until truncation, or beating only the initial down-driving policy do not establish
competence. The present .02-second visual cadence is nominal: actual observations
follow 1/60 or 1/40 seconds; learner discounts use actual elapsed time.

This first comparison determines whether longer traces produce acquisition and
whether a larger finite training allocation is warranted. It is development,
not final acceptance. Feature, trace or action-persistence changes after observing
these results require a new labeled development protocol and preservation of
this control. No large architecture rewrite is included.

## Reserved final holdout

Seeds 308713321, 1203078761, and 1983461927 remain untouched. Before running them,
freeze the chosen learner settings, training budget/checkpoint, evaluation policy,
and acceptance screen with the parent. Proposed screen: >=60% aggregate wins,
>=50% for every independent training seed, and >=20 percentage points improvement
over the strongest simple baseline on the same fresh serves, with <=5% truncation.
Use 20 rallies per evaluation seed, retaining all independent trained policies.
Also require improvement from zero checkpoint and positive return per physical
time. These are prospective proposed criteria, not permission to select the final
checkpoint on those seeds or a claim that the current system meets them.

## Recoverability and reproducibility

Pin Git/source hashes, driver hash, model SHA256, exact runtime/dependency versions,
torch thread count, all seeds/configuration, learned weights and complete records.
Use one writer per SQLite journal. Preserve per-transition complete-state mutation
checks and fail-closed replay when optimizing digest/storage overhead. Demonstrate
fresh-process replay and frozen-evaluator/global-RNG isolation before launching.
