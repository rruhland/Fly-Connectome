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

## Collection outcome

Source agenta3c2aee/main41d1ce6,29focused checks passed. Root
`pong-rally-new-TRAIN-repertoire-20261028-r1`, manifest at sibling `.manifest.json`,
ROOT1943906127/TAG63025. All128games completed in209.534s of600;
63,910actual transition cameras+128initials, no errors or unknown work.
Sources, weights and manifest stayed frozen. No fitting or bank admission yet.

On the64new starts,40069 earned139returns/58first/38games≥2/81of133later;
4c earned140/58/38/82of131later. Fourteen starts improved under4c and16regressed.
Both policies missed the same six first returns. The useful new repertoire here
is therefore for later returns, not those opening failures.

A uniform40069bank combining previous192and these64would have256episodes,
662returns/233first/161games≥2/429of615later. Exact source/raw/action audits and
target compatibility review precede admission and learning. Existing fresh sets
remain evaluation-only.

Parent's pending block proposal: up to three updates, up to three compatible new
targets each, one90sfit per update with no fit retry, and one1200s execution clock
covering broader metric preparation, current-target checks, small correctness
sentinels, final full256 and an unused fresh comparison versus40069. A later fit
failure must preserve earlier verified checkpoints. Concrete target and code
review is still required; this paragraph does not authorize launching that block.

## Audited admission and block implementation selected

`receipt-audit.json` passed all128receipts/63,910exact deployed action/Q rows,
raw shapes and source/weight/map/cap checks. Parent authorized adding ONLY the64
audited40069scaffold receipts to the old192uniform40069bank, making256contexts.
The alternative4c receipts are own-reward experience; old fresh sets remain out.
Current complete earned memory is100,069rows across233positive cases.

`learning-target-screen.json` found12of14better4c prefixes compatible with that
memory. Exclude1163974156 and3016544790 because of exact raw-action conflicts;
do not reopen them during this block. The first three selected targets are:

| Seed | Phase | Current → target | New / replaced prefix rows |
| --- | --- | --- | --- |
| 1289501189 | left | 1 → 2 | 214 / 41 |
| 58405525 | right | 1 → 2 | 322 / 140 |
| 3654232341 | left | 2 → 3 | 362 / 200 |

First contract100,586rows/233cases, combined raw-action conflicts zero.
Remaining fixed pool/order:3143732030,3614102378,1482281654,3232760599,
1881744746,3238414060,2814548503,1601498681,403762985. Before later target
replacement, measure its current-policy outcome. Skip a source already matched
or exceeded; otherwise use the source's next earned-return prefix. Recheck
compatibility against evolving memory. No new exploration or pool expansion.

Parent approved implementation of at most3updates with at most3targets each,
one90sfit per update and a single1200s execution clock. Rebuild the score metric
once using the unchanged equal-episode earned-prefix formula,1%ridge and
optimizer-only scaling. Vision, map assets and deployed scaling stay fixed.

Use actual target games and a fixed8game correctness sentinel (including the
original104562292five-hit skill) between fits. Add full newly earned prefixes
from these games to memory. Explicit per-case sourceW/bundle/Q provenance is
required; never relabel an old receipt as current-policy native execution.
Known-return lower bounds and measured full-game outcomes must remain separate.
No extra fit based on a noisy sentinel aggregate. A failed fit stops updates;
qualify the last verified candidate if time permits. Full256 and unused fresh32
versus40069 run once at the end. Review implementation and focused tests before
launch. No action-pruning or intervention-label changes in this block.

## Assumptions to challenge after the block

The user's goal is reliable learned returns, not exact preservation of every
historical action. Complete-prefix locks give efficient retention guarantees, but
may also preserve incidental behavior and force complicated decision boundaries.
If broader experience again improves only practiced games, reconsider that
constraint alongside causal credit and on-policy coverage before blaming feature
capacity or launching another identical block. Earlier linear-only soft-retention
failures do not establish failure for the joint learner. No such ablation is part
of the current block.

## Block launched

Source agentbeb99fc/maina8a0a84 reviewed;32focused checks passed. Root
`pong-rally-joint-learning-block-20261028-r1`; manifest SHA
`5759fd6f36eb4c536a6fcae7c9c4a09ec41433aca8130ada76e37fb83f32cfd3`.
Single1200s clock, implementation900s separately. Uniform256bank is revisionr7;
fresh ROOT1943906127/TAG63041 is reserved before fitting. The fixed8controls
exclude the target pool and cover old/new contexts, phase and return depth,
including104562292. Later rounds precheck all remaining pool cases before
selection, so newly observed earned prefixes cannot invalidate a previously
selected target silently. No active code changes during execution.

Preparation47.20s, including28.98s for the broader metric. All three updates were
accepted by218.70s; fits20.62s/27.59s/28.55s verified100,586/101,837/103,318rows.
Actual target returns were1/1/2→3/4/3, then1/1/2→3/4/5, then2/2/3→3/3/4.
Correctness controls28→28,28→28,28→29; the last gain was9291157632→3.
All known per-case minima passed; original104five-hit behavior retained.
Memory lower bounds662→668→676→680returns (233first,165games≥2,447later).
These are preserved minimum counts, not a complete current-policy measurement.
Full256 completed:662→686returns,233first unchanged,161→167games≥2,
later429/615→453/636 (69.76%→71.23%). Thirteen games gained and none lost
returns; all evolving earned minima passed. Full evaluation used40,494new
transition cameras+69initials and reused187exact episodes.
Cloud interruption stopped fresh qualification after58of64saved games; no
training process remained on recovery. The original allocation is not reset.
Only missing frozen qualification games may run under a separately recorded
180s recovery allocation. No training or schedule changes are authorized by
that recovery; unsaved interrupted work remains unknown.
Candidate bundle SHA
`bfb0d2824c7eb943434f31ff5772bb1ceee48add957220bd0374f6c83e84d1f7`;
W SHA `f90abda40c83aa4a9fa9df25efdbf04db66df6f13c03519cb857f828def048a6`.


## Completed block and interrupted qualification recovery

Full TRAIN256 improved 662 → 686 returns with 13 improved games and zero losses.
First returns remained 233; games with at least two returns rose 161 → 167;
later returns rose 429/615 → 453/636. The three accepted updates retained every
measured earned minimum. This establishes trajectory integration on practiced
starts, not reliable general play.

The cloud outage interrupted fresh TAG63041 after 58/64 saved games. The original
1200-second deadline expired during the outage. A separately authorized 180-second
frozen-evaluation extension completed only the six missing games in 13.575 seconds,
using 3,690 cameras plus six initial observations. The prior 58 receipts were
unchanged. No fitting, new seeds, or training restart occurred. Total documented
block work is 88,284 transition cameras plus 150 initials; unsaved interrupted
work is unknown, not zero.

Fresh32 decisively rejected the candidate: 40069 → bfb had 90 → 69 returns,
30 → 30 first returns, 21 → 19 games with at least two returns, and later success
60/82 → 39/66 (73.17% → 59.09%). Zero games gained, eight lost returns. Paired
mean return difference was −0.65625/game, bootstrap 95% interval
[−1.09375, −0.28125]. Later-rate differences also reflect changed exposure.
Artifacts: `pong-rally-joint-learning-block-20261028-r1/fresh/recovered-metrics.json`
and `recovery-extension.json`.

Do not promote bfb or repeat another identical complete-trajectory replay block.
Keep 40069 as the practical research baseline. The fresh set stays evaluation-only.
Next direction must reconsider incidental action retention and reward credit,
not claim that more practiced returns alone satisfy the user's goal. Existing
linear soft-retention and causal-credit failures must inform the next design;
no architecture change is authorized by this decision.
