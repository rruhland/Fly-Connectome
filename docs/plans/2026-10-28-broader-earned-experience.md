# Broader own-reward experience for sustained returns

## Why this is the next change

Joint output fitting now integrates several rewarded trajectories while retaining
all practiced returns. However, two updates improved the same training bank from
493 to 523 to 546 returns without a convincing fresh-game improvement. More small
replay batches on those same contexts are not the chosen continuation.

Across the two existing 32-game comparisons, 40069 versus eb earned 151 versus
145 returns, 60 versus 57 first returns, and 42 versus 39 second-return games.
Later success was 91/142 versus 88/136, essentially unchanged. This pooled result
is descriptive after an interim decision, not an untouched qualification. Use
40069 as a practical research baseline; do not claim a better later-return rate.
Keep 4c as a complementary source of the learner's own successful experience,
although its latest fresh comparison did not justify replacing 40069.

## Authorized collection

Collect 64 new TRAIN starts, balanced 32 per serve phase, selected before outcomes
and excluding every prior and reserved schedule. Run the frozen 40069 and 4c
policies on each start: 128 games under one 600-second collection clock. Use the
existing camera, vision, transport, physics, opponent and native evaluator.

A minimal manifest-driven collector records model paths/hashes, schedule and
exclusion proof, source pins, complete receipts, actual raw features and camera
costs. Do not implement another learner or cache for collection. Review code and
focused checks before launch. Never silently retry an incomplete game or reset
the allocation. If bounded work is partial, preserve and report it.

These are deliberately training contexts. Prior fresh sets 62993 and 63009 remain
evaluation-only. No policy fitting or automatic promotion occurs during collection.

## Learning follow-through

Use genuinely better model-earned trajectories as candidates for the existing
joint output learner. Validate actual rewards and compatibility with complete
earned memory before selecting a diverse batch. The frozen vision/map and no-
backpropagation constraint remain. If there are no usable witnesses, report that
and reconsider; do not silently launch another exploration scheme.

Selective, reward-verified action pruning may reduce incidental exploratory
choices in selected new witnesses. It is complementary to broader context
coverage, not a substitute for it. Earlier causal-credit code already exists;
combining it with joint retention would not be a novel credit algorithm.

Parent also proposed an efficiency option for a bounded learning block: use exact
complete-prefix certificates to preserve known returns between updates, execute
new target games, and run full-bank/fresh qualification at the end of the block.
This needs implementation review first. Certified lower bounds must remain
distinct from newly measured full-game outcomes. It could avoid repeated known-
prefix replay without the complexity and storage cost of production snapshots.

## Research interpretation

The existing [SIL discussion](2026-10-25-practical-interception.md) already warned
about overusing early successes. Parent rechecked the
[primary paper](https://proceedings.mlr.press/v80/oh18b.html); this is not a new
outside idea. Our solver is not its actor-critic algorithm, and an earned return
does not establish the causal value of every preceding action. The new collection
tests coverage while preserving that distinction.
