# Review: one fixed-partition value-learning pilot

**Tree controller: proposed major runtime change; user approval required before implementation/native use.** Data collection below is separately parent-authorized and does not implement trees. No parameter sweep.

## Choice and limitation

Prefer local fitted Q iteration before a full predictive world model. It learns future own-hit value from real transitions, including failures, without recursively generating anonymous visual states. A cheap world model must additionally learn rare hit/terminal events and a useful long-horizon value tail. Actions can matter40–80cameras before reward: a short planner without that tail may see only zeros; a longer rollout compounds state/model error. The failed LSPI critic cannot supply a validated tail.

LSPI fitted existing-runtime weights in4.613s, but8543/11776chosen actions had behavior probability<.01. Actual8-game play regressed23→2returns,7→1first,16→1later (7.42s,1097cameras+8initials). Reject that endpoint. Its sample retained only29reward rows; all256saved TRAIN executions contain494actual hit rows in118257frames, from64unique starts. This rejects the candidate, not value learning overall.

## Authorized coverage collection, before fitting

New `run_rally_value_collection.py` used frozen40069 and existing session APIs.32fixed admitted TRAIN starts,16perphase, were drawn uniformly with seed2026102884 before outcomes; independent declared action RNGs. Per camera, epsilon=.2 gave probability .8666667 for40069greedy and .0666667 for each alternative. Primitive cadence, vision, map and cap stayed fixed. Four workers; one120s clock;625transitions/start gave total≤20000. Unused quota was not redistributed. Saved actual N+1raw observations, actions, exact behavior probabilities/scores, hits, physics ticks/elapsed, and natural-terminal/environment-timeout flags, including partial traces. Hit accounting sums actual transitions, including partial traces.

**Completed:**39.099s,11871transition cameras+32initials,49own hits,23natural terminations and9quota partials, all32starts executed. Exact zero-native audit passed all11871pre-action scores/actions/probabilities and11903saved observation rows;1571nongreedy choices (13.234%, expected13.333%). Source/model/map/cap stayed fixed. The combined eligibility index contains118231valid old tuples plus11871new tuples,543actual reward rows,253natural-terminal rows and90unique starts (26added). No fit or tree implementation was performed. Artifacts: `/workspace/m1b-experiments/pong-value-collection-20261028-r1/{summary,final-audit,eligible-training-transitions}.json`.

Collection improves immediate action coverage but does not guarantee long corrective bursts: independent exploration is deliberately a small, defined acquisition test. Preserve all successes and failures. If it remains weak, do not claim coverage solved or automatically add a held-action sweep.

## One input and one learner

Use711existing legal coordinates:198spatial DCT, existing capped frame-present raw coordinate4384, fixed512TensorSketch. Freeze40069vision, cap, map and saved198×4389basis. No roles, semantic IDs, physics state, seed/phase input, expert geometry or new representation. This isolates value learning from the continuous-residual pilot. DCT retains smooth spatial variation and TensorSketch existing interactions; axis partitions may still give weak locality. Do not claim continuous entity neighborhoods from this input.

Build32fixed state trees from a4096-row geometry cache, balanced equally between old TRAIN executions and new collection, with episode/frame-uniform sampling and seed2026102882, before using rewards/actions. At each node stop at depth10 or<32cache rows. Otherwise draw32feature coordinates without replacement; use the first with positive range and one uniform threshold. Split only if both children have≥16rows; otherwise stop. No threshold retry or outcome-dependent topology selection.

Each leaf has three executed-action values. Cache current/next leaf indices. Initialize zero; perform256fixed sweeps of weighted Bellman averaging using actual `hits + .95**elapsed_seconds * max_supported_next_action Q(next)`. Update each leaf/action by the weighted mean target from its actual executed-action transitions. Fixed partitions and nonnegative normalized weights form an empirical averaging operator; no forest refits, BP, matrix inversion, actor gradient or old-logit value labels.256sweeps provide about5.12s lookahead in the worst case of one-camera reward propagation, beyond40–80camera correction horizons, without claiming full20s convergence. Select final endpoint only.

## Explicit support and fallback

For every leaf/action retain count, weight and effective sample size `sum(w)^2/sum(w*w)`. A tree/action cell is eligible only at count≥4 AND ESS≥4. A query action is supported only if≥24of32trees have eligible cells; average eligible predictions. These thresholds and masks are fixed from the data before value learning.

At deployment select the highest supported action value; ties use lowest existing action index. If NONE is supported, use frozen40069. No residual blend. At a next state with no supported action, use an explicitly labeled zero LOWER BOUND on remaining nonnegative hit reward. This is a conservative coverage cutoff, not an observed terminal or fabricated negative reward; count it separately. Actual behavior there still uses40069. This may undervalue actions entering uncovered regions.

Counts are not proof of nearby counterfactual coverage. Report learned-action/fallback fractions, leaf ESS/counts, coverage-cutoff backups, behavior probabilities and greedy changes with actual returns. If the learner makes essentially no supported decisions, report acquisition failure rather than crediting baseline fallback.

## All rewards, proper weights and time

Fit only admitted old TRAIN/training executions plus new collection. Preserve eligibility for every118257old transition and every new transition; subsample only for the bounded fit. Retain ALL actual hit and natural-terminal rows. Allocate≤12000tuples equally by old/new source, then equally by execution; uniformly sample remaining rows within execution. If protected rows exceed a source quota, expand that source deterministically and shrink the other source's unprotected quota; stop if protected rows alone exceed12000. Freeze exact integer allocation before fitting. Use inverse inclusion probabilities, source population weight1/2, equal execution weights within source and equal valid-transition weights within execution. Protected rows have inclusion probability1. Never reward every prefix action for a later hit.

Pair pre-action state with the same execution's actual successor. Natural terminal gets zero bootstrap. Drop an old final timeout row whose successor was not saved; new collection saves that observation and may bootstrap it. Reconstruct old camera12/5tick durations at1/120s and verify episode totals, allowing a shortened terminal interval. Reward is own hits, never raw score. No fresh outcomes enter fitting.

## Bounded tree comparison after user approval

Tree implementation≤15minutes separately. One600s experimental clock for data preparation, fit and qualification; fit≤120s, one BLAS/OpenMP thread. Collection has its separately reported authorized120s/20k allocation above. Cache integer leaves and sampled features, not dense120k×711buffers. Report measured memory/inference cost; estimates remain unmeasured. No restart or automatic extension.

After completed finite learning and deployed-tree parity checks, run the same predeclared8TRAINgames that exposed LSPI. Clear regression stops. If useful, complete fixed32TRAINgames reusing those8, then conditionally32genuinely unused paired fresh starts versus40069. Fresh requires total/later gains, retained first/second counts and no lower later-return fraction; mixed outcomes return for review. At most96new20s evaluation games, nominal96000transition cameras plus initials;600s dominates. No promotion or reliable-match claim. Failure stops this family, without depth, input, support or reward-weight sweeps.

[Ernst/Geurts/Wehenkel, JMLR2005](https://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf) motivates fixed randomized averaging partitions and fitted Bellman regression; this is a bounded adaptation, not an inherited guarantee for visual aliasing. [Efficient Local Planning](https://arxiv.org/abs/2108.05533) motivates real local action evidence; its simulator and strong all-policy linear-Q assumptions do not certify our critic. No tree implementation/native work has begun.
