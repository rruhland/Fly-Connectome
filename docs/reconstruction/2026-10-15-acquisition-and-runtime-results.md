# Reconstruction acquisition and execution evidence

## Current result index

**Continuous iteration was authorized on 2026-10-21**, superseding historical
stop-after-negative restrictions. Read the [completion checklist](2026-10-21-continuous-m1b-plan.md).
Latest: [frozen later-camera feedback](2026-10-21-camera-feedback-results.md), source415f066.
Across both fixed policies, live first-return lift12.50pp versus held and missing
images; aggregate paired CIs[4.69,20.31]/[3.13,21.88] pass. Seed1 missing contrast
individually inconclusive; fresh live10/32 and15/32 lower than prior cohorts.
29,724eval cameras/192jobs/272.00s, allphysics/onset/stats audited, no training.
Later visual-update benefit is not explicit forecast benefit or score mastery.
Next approved design is third full seed plus state-only comparison in old-core worktree.

Previous: [independent whole-procedure replica](2026-10-21-first-return-replication-results.md),
source886d05d. Two independently trained seeds now pass the unchanged return screen.
Replica initial6/64 → phase0 37/64 → final39/64; final64/122 all returns versus
strongest control22/85. Primary lifts35.94pp [15.63,46.88] and26.58pp [13.81,33.65].
199,771 training/95,918 eval cameras,43.20min; all295,689 rows/ten updates/phase
boundaries/statistics independently audited. Replica's incremental second-stage
benefit is inconclusive. [Two-seed chart](/workspace/m1b-experiments/pong-first-return-replication-20261021-r1/replication-summary.png).
No score wins/forecast-benefit claim or final-stream use. Broader M1B remains open.

Previous: [first-return curriculum](2026-10-21-first-return-curriculum-results.md),
source36b05e7, passed its first-seed screen:14/64 →29/64 →41/64 first returns,
final63/114 all returns; primary lifts42.19pp and30.87pp. Both stage comparisons
improved in that seed.199,676 training/92,013 eval cameras,43.34min, all291,689
raw rows/updates/statistics audited. Earlier negative allocations remain below.

Previous: [same-class continuation](2026-10-21-direct-hit-continuation-results.md),
sourceaea715b. Seven new updates,159,842 training-phase cameras including anchors,
103,900 evaluation cameras. First35/64 versus owninit32/64 andtheta9 37/64; all
48/105 versus42/103 and48/107. Primary and extra-learning screens failed. All263,742
rows/optimizer/anchors/statistics audited; negative retained.

Previous: [fresh frozen-policy confirmation](2026-10-21-frozen-hit-confirmation-results.md).
Theta9 returns 34/64 first balls and 48/106 overall versus original theta0 27/64
and 34/95. Primary lifts +10.94/+9.49 pp have intervals crossing zero and fail the
unchanged screen. Secondary blanking drops performance by +31.25/+25.03 pp,
with positive intervals: whole-input dependence, not reward-acquisition proof.
All 83,814 evaluation rows/512 jobs independently audited; no new training.
Next is a separately registered same-class continuation, not final holdout.

Previous: [direct hit-return policy search](2026-10-21-direct-hit-policy-search-results.md).
Nine completed updates used 145,561 of 159,873 training cameras. Fresh endpoint
returns 16/32 first balls and 21/51 overall versus own initial 8/32 and 8/40,
strongest simple down 9/32 and 10/42. First lift 21.875 pp (95% interval −3.125 to 31.25),
all lift 17.367 pp (1.772 to 25.359): positive point estimates, unchanged screen failed.
No replicated acquisition or final holdout. Restart recovery preserved clock/data;
99 existing steps were replayed. Candidate workers exited; independent physics,
optimizer and bootstrap audit passed. A larger fresh frozen-policy DEV confirmation
is the next reviewed design; the original failed outcome remains unchanged.

Previous: [selected-lineage repair and corrected SARSA](2026-10-21-continuity-repair-results.md).
The repair removes fabricated cross-hypothesis velocity/forecast histories while
preserving ordinary coverage; retained forecasts have identical errors, with
invalid ones withheld. The fixed24k reward learner remains negative:11/32 first
returns and15/45 overall versus constant up11/32 and14/45. No genuine acquisition.
The subsequent direct-hit search is reported above.

Previous investigation: [camera-rate sustained exploration](2026-10-16-sustained-exploration-results.md).
Two fresh 24k-camera hit-only learners kept identical SARSA update counts and about
20% exploratory occupancy. Sustained exploration increased training contacts
46→60 but yielded fewer fresh returns: first 3/24 versus independent 6/24, overall
4/28 versus 8/32. Strongest random control achieved 10/24 and 12/35. Neither
qualified; direct-arm uncertainty intervals cross zero. No acquisition or
automatic further allocation follows.

Previous investigation: [reliable-opponent return learning](2026-10-16-reliable-return-learning-results.md).
Two hit-only DCT-SARSA policies trained for 12k cameras each against a .65 opponent.
Fresh hold1 returned 3/27 incoming balls and 3/24 first balls; hold10 returned
13/36 and 9/24. Strongest simple controls achieved 10/33 and 7/24. Hold10's small
lifts had paired uncertainty intervals crossing zero; neither arm qualified.
The opponent returned all 105 evaluation balls and there were no natural wins.
No replication or further experiment is automatically allocated.

Previous investigation: [reward-credit and vision results](2026-10-16-reward-and-vision-results.md).
Three fixed12k DCT-SARSA arms yielded8/24score-only,9/24hit-shaped and9/24survival-
shaped fresh wins, all below simple controls. Independent vision localization and
ball forecasting were substantially functional on two tested trajectories; no
visual-necessity or Bellman-Q adequacy claim follows. The staged90k score-only
DCT study was NOT launched. No further experiment is automatically allocated.

**Historical negatives below remain unchanged.** The new single-seed return result
is reported above; independent replication remains open. The original .45 opponent
has a likely scoring ceiling after own returns; it is paused as a learning target.
The separately labeled .25 opponent is controllably scoreable, but its first raw
6000-transition screen only matched no-op at 5/12 wins. Both subsequent uncapped
longer-study arms failed numerically before their fixed endpoint. The new raw
arm with a generic input-norm cap finished stably but failed its acquisition
screen (9/24 wins versus strongest control10/24). Capped block-RMS also failed,
8/24 with no own-hit wins; frozen epsilon.2 matched a bias-only control at8/24.
The step-size/control-cadence pair also failed: lower alpha9/24 and hold4 11/24,
versus strongest control10/24. No arm qualified or received replication.
Score-only smooth policy search also failed its fixed gate:12/24 wins (four
own-hit wins), versus its initial10/24 (two own-hit wins), after seven updates.
The fixed continuation also failed after14additional updates:10/24(2own-hit),
matching original initial10/24(2own-hit). No candidate received replication.
The stationary→.25 curriculum also failed its fixed screen:13/24(5own-hit),
versus original initial10/24(2own-hit), below the unchanged14total/6own-hit gate.
This positive point estimate on one reused development allocation is not a
replicated acquisition result; no checkpoint or threshold was changed.
All final holdout seeds remain unused. No .25 result is an original-opponent
success claim.

| Experiment | Registration / interpretation | Observed endpoint |
| --- | --- | --- |
| Original toy pilot | [Registered toy protocol](2026-10-15-toy-preregistration.md); exact allocation | Immediate all 20/20; delayed all 10/20 |
| Original .45 Pong | [Protocol](2026-10-15-pong-development-protocol.md) | All 12 trained checkpoints 0/12 wins |
| Fixed block conditioning | [Protocol](2026-10-15-fixed-block-conditioning-protocol.md) | Every raw/scaled delayed seed 10/20 |
| Fixed 200-episode exposure | [Protocol](2026-10-15-conditioning-exposure-protocol.md) | Every raw/scaled delayed seed 10/20 |
| Larger-alpha .45 Pong | [Protocol](2026-10-15-pong-alpha-screen-protocol.md) | Both checkpoints 0/12; stopped |
| Scoring feasibility | [Methods and results](2026-10-15-pong-scoring-feasibility.md) | 429,614 relaxed own returns, zero return-to-score wins at .45; legal .25 controls can score |
| Raw .25 Pong | [Protocol](2026-10-15-winnable-pong-screen-protocol.md) | 2000: 3/12; 6000: 5/12, matching no-op; stopped |
| Longer .25 exposure/sharing | [Protocol](2026-10-15-pong-exposure-sharing-protocol.md) | Raw failed at 20,773; divergent sharing stopped at 23,400; no endpoint evaluation |
| Fixed radial-cap .25 Pong | [Protocol](2026-10-16-radial-cap-pong-protocol.md) | Stable30k endpoint,9/24wins(1own-hit), below strongest control10/24(2own-hit) |
| Capped block-RMS .25 Pong | [Protocol](2026-10-16-capped-block-rms-pong-protocol.md) | Stable30k endpoint,8/24wins, all natural; no acquisition |
| Frozen epsilon behavior | Diagnostic only, `frozen-epsilon-diagnostic-20261016` | State-dependent and mean-Q bias-only policies both8/24, no own-hit wins |
| Lower alpha versus hold4 .25 | [Protocol](2026-10-16-alpha-hold-pong-protocol.md) | Lower alpha9/24(1own-hit); hold4 11/24(3own-hit); neither qualifies |
| Score-only smooth policy search .25 | [Protocol](2026-10-16-score-only-policy-search-protocol.md) | Seven updates/59,903 training cameras;12/24(4own-hit) versus initial10/24(2own-hit); failed14/6 gate |
| Fixed search continuation .25 | [Protocol](2026-10-16-score-only-policy-search-continuation-protocol.md) | Fourteen additional updates/119,831 NEW training cameras;10/24(2own-hit), matching original initial; failed |
| Stationary→.25 score-only curriculum | [Protocol](2026-10-16-stationary-curriculum-protocol.md) | 5stationary+7target updates/89,712training cameras;13/24(5own-hit), below14/6 gate |
| DCT reward-credit diagnostic | [Protocol](2026-10-16-reward-credit-diagnostic-protocol.md); [combined results](2026-10-16-reward-and-vision-results.md) | 12k each: score8/24(0own), hit9/24(1own), survival9/24(1own), all below controls |
| Frozen vision and immediate feedback | [Combined methods/results](2026-10-16-reward-and-vision-results.md) | Observed/predicted point coverage99.85–99.90%within3px; ball forecasts beat matched baselines; cached immediate mapping83–86%, no reward acquisition |
| Reliable-opponent hit-only hold1 versus hold10 | [Protocol](2026-10-16-reliable-return-learning-protocol.md); [results](2026-10-16-reliable-return-learning-results.md) | First returns 3/24 versus 9/24; overall 3/27 versus 13/36; strongest controls 7/24 and 10/33; neither qualifies |
| Camera-rate independent versus sustained exploration | [Protocol](2026-10-16-sustained-exploration-protocol.md); [results](2026-10-16-sustained-exploration-results.md) | 24k updates each; first returns 6/24 versus 3/24, overall 8/32 versus 4/28; strongest control 10/24 and 12/35; neither qualifies |
| Corrected-vision hit-only SARSA | [Protocol](2026-10-21-continuity-repair-protocol.md); [results](2026-10-21-continuity-repair-results.md) | 24k cameras; first11/32 and all15/45 versus up11/32 and14/45; no acquisition |
| Corrected-vision direct hit search | [Protocol](2026-10-21-direct-hit-policy-search-protocol.md); [results](2026-10-21-direct-hit-policy-search-results.md) | Nine updates/159,873 cameras; first16/32 and all21/51 versus strongest control9/32 and10/42; positive point estimates, screen failed |
| Frozen hit-policy confirmation | [Protocol](2026-10-21-frozen-hit-confirmation-protocol.md); [results](2026-10-21-frozen-hit-confirmation-results.md) | Fresh 64: theta9 first34/64/all48/106 versus initial27/64/34/95; primary screen failed, secondary visual blanking reduces returns |
| Cached delayed-template credit | [Diagnostic protocol](/workspace/m1b-experiments/cached-delayed-toy-20261016/PROTOCOL.md) | Fixed RMS/alpha.001 reaches both templates in all seeds by1000episodes; raw fails through2000 |
| Oracle diagnostic closed-loop capacity | [NEW diagnostic protocol](/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-closed-loop-capacity-20261016/PROTOCOL.md) | Camera-only oracle-fitted linear readout8/12(5own-hit); legal teacher8/12(5own-hit); NEVER reward acquisition |
| Restricted-class oracle capacity | [NEW diagnostic protocol](/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-restricted-capacity-20261016/PROTOCOL.md) | Frozen search class6/12(3own-hit),4loss/2timeout; capacity witness only, no learning |
| Fixed centering mechanism | [Diagnostic protocol](/workspace/m1b-experiments/cached-centered-toy-20261016/PROTOCOL.md) | No arm meets the200episode/stability target; some reverse both choices |

All named artifact directories are under `/workspace/m1b-experiments/`; they retain
complete source snapshots rather than relying on dirty-source hashes alone.

## Scope and runtime

These are new observations from the reconstructed learner, not imported historical
success claims. Runtime: `/workspace/Fly-Connectome/.venv/bin/python`, Python
3.12.14, Torch 2.14.1+cpu, NumPy 2.5.4; explicit `PYTHONPATH` selects this worktree.
No backprop was used. Later reward diagnostics explicitly changed only the
registered scalar objective; their scope is labeled separately. The pinned frozen visual prior has empty
long-context banks and therefore uses its four-step fallback.

## Exact registered toy pilot

The unchanged fixed pilot completed all 36 jobs: **2400 learning + 2160 evaluation
= 4560 new transitions**, zero replay, 205.05 seconds aggregate wall time. All
three immediate lambda pairs had identical full training records and final weights.

| Task | Trained lambda .8, per seed | Trained lambda 0, per seed | Initial frozen | Random, per seed |
| --- | --- | --- | --- | --- |
| Immediate | 20/20, 20/20, 20/20 | 20/20, 20/20, 20/20 | 10/20 each | 10/20, 11/20, 7/20 |
| Eight-step delayed | 10/20, 10/20, 10/20 | 10/20, 10/20, 10/20 | 10/20 each | 10/20, 9/20, 13/20 |

Every trained immediate policy was correct on 10/10 left and 10/10 right scenes.
Every delayed policy selected only one side: one side 10/10, the other 0/10. The
immediate acquisition screen passes; delayed acquisition and the exploratory
trace-benefit screens fail at this fixed allocation. Immediate training used 200
episodes per arm/seed, delayed only 25 because allocation matched transitions.
This does not prove delayed learning or the representation incapable.

This is acquisition of **two fixed visual templates**, not visual generalization
or Pong learning. Evaluation uses fresh camera/perception histories and copies only
the learned weights. Repeating a template does not create independent visual
generalization trials. The source checkpoint and all original pilot artifacts are
preserved at `/workspace/m1b-experiments/registered-toy-20261015-r1/`, including the
original source archive and exact runtime dependency list.

Post-pilot first-choice Q decomposition uses the same two templates and existing
weights; it adds no learning or new independent competence trials. Immediate
right-minus-left Q gaps have opposite signs across the templates in every seed.
Delayed visual contrast changes those gaps by roughly 6e-6 to 3.3e-5, while common
metadata contributions have magnitude roughly .0006 to .006. Those learned biases
explain one-sided greedy evaluation at this allocation. The raw decomposition is
retained in `first-choice-q-decomposition.json` beside the original pilot summary.

## Execution and recovery

The first 128-transition Pong profiles found median feature norm squared about
77.5, three entities, nine forecasts and 105 active features. With cProfile enabled,
direct p50/p95 step latency was 14.7/23.8 ms; durable p50/p95 was 42.8/125.5 ms.
Complete-state fingerprinting consumed 4.38 of 6.61 profiled durable seconds.
These were separate diagnostic seeds, not matched competence comparisons.

The replacement digest preserves complete tensor bytes, array shape/dtype,
explicit object aliases/cycles, RNGs and process-local cache-version normalization.
It encodes the canonical primitive tree as binary with pickle memo disabled,
avoiding hexadecimal expansion and JSON. Its version is pinned in the journal;
old digest formats fail closed and require the original source to reopen.

An isolated retained-state benchmark reduced median fingerprint time from about
15–16 ms to 8.61 ms. A matched 128-transition end-to-end comparison produced
identical complete states and records: legacy 5.17 seconds versus binary 4.26
seconds including startup/save. Some concurrent diagnostic CPU load was present,
so this is measured execution evidence rather than a hardware-independent guarantee.
Three separate 200-transition workers completed in 8.30 seconds together; one
200-transition worker under short overlapping diagnostic load took 7.56 seconds.
This supports three workers for independent seeds while keeping one writer per DB.

The fresh-process crash proof passed: 16 uninterrupted versus 9 + crash + replay
2 + 7, **32 new + 2 replay transitions**, 6.21 seconds. The full sample sequences,
complete final state, nonzero weights/traces, pending action and immutable bank
matched. Hash mutation tests include NumPy writes that bypass Torch version
counters. Fresh Pong evaluator tests cover first-serve/reset semantics, global RNG
and training isolation, forbidden learner calls, frozen-bank mutation detection,
baseline perception skipping and durable replay across a rally boundary.

A separate fresh-process Pong evaluator proof also passed: 32 new + 2 replay
transitions, 8.83 seconds, identical full records and complete state after crash
and restart. Its deliberately forced terminal-score fixture checks boundaries
only; its positive score is not competence evidence. The proof source is retained
beside `/workspace/m1b-experiments/pong-eval-fresh-process-proof-20261015/report.json`.

The final focused gate before Pong development was **69 passed in 14.68 seconds**.
Named runtime artifacts live under `/workspace/m1b-experiments/`; unit fixtures are
correctness evidence and are not added to competence observation counts.

A later isolated warm-session benchmark retained one verified writer across five
200-camera chunks instead of closing/recovering at each chunk. Full per-camera
SQLite commits, heartbeats,100camera/10second snapshot bounds and full snapshots
at every chunk remain; actual opens still perform complete historical integrity
checks. Cold1000camera continuation took36.49seconds including imports/final saves,
versus32.16warm (11.85% reduction in one sequential pair under changing concurrent
load). Both produced identical18,107record/state/chain rows, complete final graph,
weights and traces. Parent comparison also matched every row to the original
registered trajectory:2000diagnostic camera executions supplied zero novel
candidate data. Each mode replayed seven preexisting samples once. Cold used five
opens/stops; warm one open, five chunk snapshots and one stop. Sixteen focused
worker/durability/driver tests passed, including stale-writer poisoning and failed
checkpoint recovery. The worker path is opt-in via `--warm-session`; old archives
remain usable and control source bytes were unchanged. Full artifacts and the
prospective protocol are at
`/workspace/m1b-experiments/warm-session-benchmark-20261016/`.

## Pong development

The exact [development protocol](2026-10-15-pong-development-protocol.md) completed
from a complete source snapshot, with 36,000 learning transitions allocated across
three dispersed seeds and lambda .8 versus .995. The latter is a prospective
longer-credit comparison; alpha, reward, architecture and visual inputs are fixed.
Fresh evaluation compares 2000/6000 checkpoints against shared initial, random and
no-op baselines on identical serves. It used 34,025 fresh evaluation transitions
for 180 attempted rallies and finished in 1047.74 seconds (17.46 minutes), with
zero replay. All 12 learned checkpoint evaluations and three baselines won 0/12
rallies. Every learned checkpoint except one lost all 12; seed 736187003/lambda
.995 at 6000 transitions had 10 losses and 2 truncations, with 19 hits. Shared
initial/random/no-op baselines had 4/7/8 hits and 12 losses each. Surviving longer
or making more contacts does not establish score acquisition. **This control did
not demonstrate Pong learning.** All three reserved final seeds remain unused.

Raw reports, weights, full per-transition journals, runtime/source manifests and
the complete source snapshot are retained at
`/workspace/m1b-experiments/pong-development-20261015-r1/`. The separately declared
[conditioning mechanism gate](2026-10-15-fixed-block-conditioning-protocol.md)
investigates one measured failure mode before further Pong allocation.

## Conditioning, exposure and step size

The separate fixed-block conditioning gate used 512 score-free calibration
transitions, 1200 learning transitions and 1920 fresh evaluation transitions;
all six raw/scaled policies remained 10/20. It finished in 69.42 seconds with
zero replay. The fitted bounded, uncentered scales preserved actual mean total
input energy. Exact credit replay showed a 48–88-fold increase in first-choice
discrimination, but a common action bias still prevented correct choices on both
templates. More than 99% of this discrimination came from terminal reward times
the Dutch trace. A separate balanced 200-episode exposure allocation then used
9600 learning plus 960 fresh evaluation transitions in 121.78 seconds. Every
raw/scaled seed remained 10/20. These are failures at their declared allocations;
the conditioning hypothesis did not establish acquisition.

The [isolated Pong step-size screen](2026-10-15-pong-alpha-screen-protocol.md)
increased alpha to .001, retaining lambda .995 and raw inputs. Its first seed
completed 6000 learning and 4213 fresh evaluation transitions in 302.45 seconds,
with zero replay. Both checkpoints won 0/12 and lost all 12 rallies, with six
hits at 2000 versus four at 6000. The declared stop rule prevented replication
or extension. Weights stayed finite (maximum absolute weight .099); maximum
actual input squared norm was about 700, far above its median around 79.
The larger step size did not establish Pong acquisition.

A journal-only audit found all seven winning training rallies across the original
six arms and this screen had **zero own paddle hits**. Every win lasted 124–125
physics ticks, about 1.04 seconds. Positive feedback so far has come entirely
from uncontrollable serve/opponent behavior, with no observed return-to-win
example. This motivates examining reward sparsity and scoring feasibility before
spending on further standard-opponent optimization. Neither hits nor longer
survival will be relabeled as score acquisition. Final holdout seeds remain unused.

All three experiments retain complete source snapshots, journals, runtime/config
manifests, weights, fit artifacts and per-process costs under their named
`/workspace/m1b-experiments/` directories. The win audit is
`/workspace/m1b-experiments/training-win-ownhit-audit-20261015.json`.

## Explicit .25-opponent screen and longer-study failure

The separate winnable-opponent screen used 6000 learning plus 8801 fresh evaluation
transitions, zero replay, and 351.15 seconds. At 2000 transitions it won 3/12,
all natural serves; at 6000 it won 5/12, comprising three natural plus two own-hit
wins. No-op also won 5/12 with the same split. Initial/down and constant-up each
won 4/12 (three natural plus one own-hit), random won 3/12 (all natural), and all
allocations had zero truncations. The declared acquisition screen failed, so no
replication or extension occurred under that protocol. Training's 70 rallies had
24 wins (20 natural, four own-hit), 46 losses and ten contacts. Exact replay of
all 6000 records and final weights passed in 67.03 seconds; this adds no new data.

The 6000 readout is state-dependent without outperforming a simple baseline.
On a separate score-free calibration trajectory, its mean action-bias spread was
.0748 versus median state-dependent spread .1646; at 2000 those values were .296
and .0841. Thus large common input energy alone is insufficient reason to apply
block balancing. Fine spatial support remained sparse: current/future active
feature medians were four/two visits per action, with 40%/44% missing at least one
action. This motivates a fixed-endpoint longer-exposure comparison with generic
neighbor credit sharing, rather than more short modifier screens.

Its separately registered 1024-transition calibration used no learner or score
inputs, took 47.80 seconds and observed 1034 images including resets. Sharing
compensation .9847227473/.9573019387 preserves actual mean current/future energy.
The [longer paired study](2026-10-15-pong-exposure-sharing-protocol.md) started both
arms from zero and fixed 30,000 as the evaluated endpoint. **Neither reached it.**
Raw committed 20,773 transitions before nonfinite weights. Sharing was stopped
at durable transition 23,400 after finite numerical divergence; this resource
stop is an explicit post hoc protocol deviation, preserved in its amendment.
There was no trained endpoint evaluation or qualifying acquisition result.
The allocation used 44,173 learning and 11,551 baseline-evaluation transitions
in 857.29 seconds. The shared 24-rally controls won 9/10/9/10 for initial/up/
random/no-op, with one/two/one/two own-hit-followed wins and no truncations.

Exact replay reproduced every training record, checkpoint weights and captured
complete-state hashes before and through both failure windows. Visual public-ID
hypothesis switches alternated between a moving point and a vertical object,
producing increasingly large finite velocity differences. Current/relative-motion
input energy exceeded 2000 while weights and Q were still small; expanding Dutch
traces and weights followed. Average-energy matching did not bound these rare
inputs. The replay retains full structured features, entity histories, images,
weights and traces under `pong-exposure-sharing-20261015-r1/numerical-replay-*`.
This is a numerical failure, not an informative fixed-30k acquisition endpoint.

## Fixed-cap robustness study and isolated capacity diagnostic

The [new radial-cap protocol](2026-10-16-radial-cap-pong-protocol.md) fixes a generic
policy-input squared-norm bound of 256, identity below that bound, after raw
transport. All 1024 independently calibrated inputs are unchanged (maximum
243.11); no reward or entity role fits the bound. Positive global radial scaling
preserves greedy argmax for fixed weights but changes TD values and learning.
Captured-sequence regression reproduced all uncapped weight prefixes exactly;
with the cap, maximum absolute weights remained .154/.0733 instead of the
uncapped raw/sharing windows' 5.60e70/1.48e6. This is stability evidence only.
Finite guards now reject nonfinite proposed weights, traces or action values
before another journal sample commits. A failed live writer must be discarded;
recovery restores the last valid journal state. The focused gate passed 48 tests.

The separately registered fresh raw+cap arm completed exactly30,000 decisions on
the same .25 physics, alpha .001 and lambda .995, with finite weights throughout.
Fresh final evaluation won **9/24** (eight natural serves, one own-hit-followed),
lost15 and truncated0, with four contacts. Initial also won9; strongest up/no-op
won10 with two own-hit wins. Win fraction was.375, score per rally−.25, and
physical-time return−.11869 over50.55seconds. **The fixed endpoint failed the
14total/6own-hit development gate and did not demonstrate Pong acquisition.**
The run used30,000learning+2534fresh evaluation transitions, zero replay, and
1215.19seconds. Stability is an improvement, not a competence result.

Training's286 completed rallies had97wins(71natural+26own-hit),189losses and
59contacts, with no truncation. Final maximum absolute weight was.0944, maximum
historical Q spread7.54 and actual input norm²≤256 within rounding. On the
existing calibration cache, final common action-bias spread.594 exceeded median
state variation.234; actual frozen evaluation selected down on95.2% of decisions.
Training's last5000decisions had15.48action switches/second and median one-camera
runs, while frozen evaluation had1.68switches/second with strong directional bias.
These are descriptive mechanism diagnostics, not acquisition substitutes.
Source, runtime, cap fit/config, weights and full journals are preserved under
`pong-raw-cap-20261016-r1`. Exact cross-source replay verified all11,551 reused
control action/physics records; final holdout remains untouched.

An isolated **diagnostic-only** ridge readout of a privileged legal PD teacher
used existing generic feature caches and physics replay, adding no learner data
or new policy observations. No diagnostic coefficients were saved or supplied
to any candidate. It matched teacher actions on 99.0% of fit rows, 86.0% of
excluded rallies (majority baseline 60.1%) and 92.0% of an independent calibration
trajectory (majority 57.7%). LSMR hit its declared 2000-iteration limit. Agreement
fell to 74.4%/87.2% on the two numerical-failure windows, where constant-up was
100%. This indicates substantial tracking information is linearly accessible,
with conditioning/distribution limitations; it is neither closed-loop control
evidence nor reinforcement-learning acquisition. The 33.45-second audit and
complete diagnostic specification are retained under
`ORACLE-DIAGNOSTIC-readout-capacity-20261016`.

A second, separately [registered capped block-RMS arm](2026-10-16-capped-block-rms-pong-protocol.md)
was launched before the raw endpoint. It uses only actual mean raw block energies
from the same score-free 1024-row calibration: context is scaled8, metadata.534,
current1.644, future1.696 and relations1.328. Pre-cap mean total energy is matched;
three calibration rows exceed256 and capping reduces mean energy by about.18%.
No diagnostic teacher coefficients enter the candidate. This tests conditioning,
not a claim that metadata bias remains dominant throughout training: by raw+cap
15k, median state-dependent Q spread .264 exceeded common action-bias spread .145.
The raw and scaled protocols retain separate fixed30k endpoints and the same
24-rally gate. Among qualifying arms choose most total wins, ties raw; no early
checkpoint is eligible. Shared control/vision/worker source is unchanged, with
separate source/fit/config archives. Ten composition/recovery tests passed before
launch, including frozen-base mutation detection and exact durable replay.

That arm completed30k finitely but won **8/24**, all eight natural serves, with
16losses,0truncations and two contacts. Score/rally was−.3333 and return/physical
second−.18583 over43.05seconds. The allocation was30,000learning+2162fresh
evaluation transitions in1221.49seconds, zero replay. Neither capped arm qualified;
no seed replication or final holdout was used. Artifacts remain under
`pong-capped-block-rms-20261016-r1`; raw remains the simpler base, without treating
its one-win difference as evidence of superiority.

The bounded frozen epsilon diagnostic then used the raw30k weights with training
epsilon.2, versus the same readout's calibration-mean-Q state-independent bias,
with exact matched SARSA exploration/tie semantics. Both policies won8/24, all
natural, with16losses,0truncations and two contacts. It used4289fresh evaluation
camera transitions and95.28seconds shared wall, no updates or oracle inputs.
This weakens a frozen-exploration mismatch explanation at this allocation.
Full per-camera journals/readout/RNG/source pins are retained under
`frozen-epsilon-diagnostic-20261016`. It is diagnostic, not acquisition acceptance.

The next [paired alpha/hold protocol](2026-10-16-alpha-hold-pong-protocol.md) tests
raw-cap alpha.0001/hold1 versus alpha.001/hold4 at matched30k-camera budgets.
Every camera still updates vision and commits durably; only action/learner
boundaries change. Preparation passed51focused tests, a400-transition hold1
source-equivalence proof with exact records/weights/source-normalized complete
state, and an80new+1replay fresh-process mid-hold proof. Both arms completed finitely. Lower alpha won9/24(8natural+1own-hit), lost15,
and made four contacts; hold4 won11/24(8natural+3own-hit), lost13 and made eight
contacts. Both had zero truncations. Their score/rally values were−.25/−.08333,
and physical-time returns−.12026/−.03316 over49.8917/60.3167seconds. The strongest
matched control won10/24(2own-hit). Neither arm reached the registered14total/
6own-hit gate; no arm was selected or replicated. The pair committed60,000learning
and7983evaluation cameras in1334.73seconds shared wall. Hold4 executed7612macro
choices and7611updates, ending with a durable partial three-camera hold. More
contacts and a one-win baseline lift do not establish acquisition.

## Cached delayed-credit mechanism diagnostics

A separately registered cache equivalence diagnostic reproduced generic encoded
features exactly across repeated resets:34unique histories and74repeat checks.
Raw and block-RMS live versus cached128-update trajectories each matched every
record and final weight exactly. The verification used396new images; subsequent
factorial runs reused those features and the task's unchanged terminal reward.
The24-arm comparison used384,000cached updates, three seeds, raw versus the
existing fixed RMS transform, alpha.0001/.001 and lambda.8/.995, ending at2000
episodes. It took66.67seconds including equivalence. All raw arms still chose
only one template at the endpoint. RMS/alpha.001 learned both templates in every
seed by1000episodes with lambda.8 and by500 with lambda.995, retaining both through
2000. RMS/alpha.0001 succeeded on only two seeds per lambda. This demonstrates
operational delayed credit and a conditioning/exposure limitation on two fixed
visual templates; it is not Pong or visual-generalization evidence. Earlier
25/200episode failures remain failures at their registered allocations.

A further fixed180second diagnostic tested score-free global centering in raw
and RMS bases at both step sizes, three seeds, with the same2000episode endpoint.
The transform retains the frame-present coordinate and subtracts its calibrated
mean multiple from each other coordinate, followed by a fixed mean-energy match.
It is invertible, preserves the all-zero input and the linear Q-function class,
but individual missing coordinates can become nonzero when a frame is present.
No arm met the preregistered all-seed acquisition-by200episodes and retained-at-
1000/2000 target. Centered raw/alpha.001 learned both templates in only one seed
at2000; centered RMS/alpha.0001 reversed both choices in all seeds at2000.
The224,000cached updates (including32,000baseline replay) took43.76seconds.
Centering over mostly post-choice scenes inserted large common components into
the sparse first-choice scene: centered RMS initial diagonal8.32 versus cross7.83;
centered raw initial Gram conditioning worsened to~2752 from~725. Preserved global
energy and invertibility did not ensure helpful initial-decision TD geometry.
**No centering promotion or additional toy sweep follows this negative result.**
Complete protocols, cache/fit artifacts, all arm outcomes and source pins remain
under `cached-delayed-toy-20261016` and `cached-centered-toy-20261016`.

## Quarantined closed-loop capacity witness

A NEW [oracle-diagnostic protocol](/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-closed-loop-capacity-20261016/PROTOCOL.md)
explicitly deployed a fixed diagnostic fit, after the earlier representability
audit that prohibited policy deployment. It refit the exact prior4715rows,
score-free column RMS, ridge.001 and2000iteration LSMR recipe; fit/excluded-rally
agreement matched99.003%/85.992% exactly. A scalar drive coefficient was mapped
to three existing linear score heads using the frame-present bias coordinate:
down=−drive−.1*b, no-op=0, up=drive−.1*b. Positive radial capping preserves that
argmax/deadzone. Inference received only existing camera-derived generic features;
privileged annotations were computed separately for diagnosis and never supplied
to the fitted policy. Coefficients were never exported or given to a candidate;
only hashes, source and fit recipe are retained.

On the same12DEV serves, the camera-only readout won8(3natural+5own-hit), lost4,
truncated0 and made19contacts; score/rally+.3333, return/physicalsecond+.05065 over
78.975seconds. Every one of its nine nonnatural rallies had an own hit. Legal PD
teacher won8(3natural+5own-hit), lost0, truncated4 and made44contacts over155.558
seconds. Equal win counts do not establish teacher equivalence: the camera-only
fit lost four rallies that the teacher survived. Online teacher-action agreement
was only62.13%, despite the positive closed-loop result. Solver hit its fixed
iteration limit. The complete diagnostic used3953camera transitions/3965images,
7780teacher physics-camera events and105.85seconds, within180seconds; zero reward-
learning updates, no candidate selection, no final seeds. It is a positive capacity
witness on these serves, **never reward-only acquisition**. It supports further
investigation of learning/credit geometry without proving linear Bellman-Q
representability or universal observability. All outcomes/records remain in
`ORACLE-DIAGNOSTIC-closed-loop-capacity-20261016`.

## Post-return representation reliability

A separate ordered-journal audit found severe raw transport outliers concentrated
after own returns. Raw-cap inputs before any prior own hit:20,657, with none above
norm²256; after a prior hit:9343, with226above256 and111above1000. RMS-cap counts
were20,134before(noneabove256),9866after(96above256,noneabove1000). These are raw
transport norms, not transformed cap activation counts; longer rallies also permit
more tracker history, so the association is diagnostic rather than causal proof.
Captured state shows public-slot best branches switching between a point and a
vertical object. Finite differences across those branches invent large velocity
although each branch's own velocity is modest; position and forecasts also switch.
The norm cap bounds numerical damage without restoring identity. The audit with
ordered-record hashes is preserved at
`/workspace/m1b-experiments/post-return-feature-audit-20261016.json`.

## Score-only policy search and fixed continuation

The separately [registered smooth policy search](2026-10-16-score-only-policy-search-protocol.md)
retains frozen perception/raw transport and ordinary linear policy inference,
while replacing Bellman-Q learning in this isolated arm with mirrored terminal-
score policy search. Generic lowest3×3DCT modes across every spatial plane plus
nongrid coordinates, score-free centering and regularized support projection give
183supported modes/366freeactioncontrasts; this restricted class is smaller than
the witness class. No oracle weights/labels choose its basis or parameters.
Eight mirrored direction pairs share four fresh-phase training serves per
complete generation, rotating only on a declared independent training stream.
Partial generations never update. Perturbations have calibration Q-RMS.25,
initial policy.10, and clipped top4update.10. Fitness is terminal score only.
The last full generation at8generations/60kcamera/24minute training bound is
frozen, with at least two full generations required;30minutes overall reserves
six for endpoint and initial-policy evaluation. Three persistent spawned workers
use full per-camera journals and warm chunks. Required gates passed21focused
tests, then7search tests after endpoint-freeze handling, a16new+1replay
fresh-process proof and24camera/27image worker smoke. Artifacts/source snapshots:
`/workspace/m1b-experiments/pong-score-policy-search-20261016-r1/`. No reward-learning
success is claimed. Its fixed endpoint completed:12/24wins(8natural+4own-hit),
12loss/0timeouts/sevenhits, versus original restricted initial10/24(8natural+2own),
14loss/0timeout/sixhits. Both total and own-hit gates failed. Total score/rally0
and physical return0. Seven complete updates used56,089training cameras;3814
additional cameras in incomplete generation7 were preserved but excluded. Counts:
59,903training-fitness+5543fresh-evaluation cameras,65,974images with resets,
zero replay,1127.75seconds shared wall. The3132.76summed persistent-task seconds
exclude pool startup and must not be conflated with earlier subprocess wall totals.
A read-only independent audit exactly reconstructed all448scored rollouts,
candidate arrays, seven updates and optimizer RNG from journal terminal scores.

A NEW quarantined witness used the EXACT rank183 restricted transform and prior
fixed4715row PD-label/LSMR recipe, without tuning or coefficient export. It won
6/12(3natural+3own-hit),4loss/2timeout/24hits. It used4903policy cameras/4915images,
129.78seconds, and replayed the teacher's7780physics camera events for reuse proof.
This supports usable restricted-class control, not mastery, a class ceiling or
reward acquisition. Its coefficients and labels never enter candidate learning.

The [completed fixed continuation](2026-10-16-score-only-policy-search-continuation-protocol.md)
starts from theta7, skips original partial-generation fitness, advances to its
postdraw RNG and starts fresh global-generation index8. Up to16additional updates,
120k NEW training cameras,34minutes training/40overall, four workers; same basis,
score-only rule, noise.25/step.10/hold1/.25physics. Last full update only; original
theta0 and simple controls define14/6 gate, theta7 is descriptive. Maximum23total
updates uses globalindices8..23 after the original0..6; index7 was not updated.
No intermediate DEV or final seeds. A later replication must reproduce the whole
staged procedure from zero, not selectively continue a promising intermediate.

The continuation completed14newupdates (21total/globalindex21),119,831NEWtraining
cameras+3486fresh evaluation,1642.78seconds sharedwall and5718.25summed persistent-
task seconds.119,495cameras contributed updates;336partial-population cameras did
not. Endpoint10/24(8natural+2own-hit),14loss/0timeout/10hits, score/rally−1/6,
return/physicalsecond−.057506 over69.5583seconds. It matched the original initial
10/24(2own-hit) and regressed from descriptive theta7's12/24(4own-hit);14/6 failed.
Independent cached audit reconstructed inherited theta7/skipped RNG, all896new
scored rollouts/directions/candidates/returns/updates and actual frozen readout
exactly,5.21seconds/zero observations. Endpoint calibration function-RMS.40079,
stateRMS.39784,biasRMS.04849; noise/policy RMS.62377 and meanlast-population action
disagreement25.35% do not support exploration collapse. No further same-rule .25
extension, replication or final evaluation is allocated. The next authorized
experiment was a separately prospective fixed stationary→.25 score-only curriculum;
it does not relabel either failed search.

The [stationary curriculum](2026-10-16-stationary-curriculum-protocol.md) completed
five stationary-opponent and seven .25-opponent updates from the original theta0,
using the same generic transform, optimizer, perturbations and terminal-score-only
fitness. Phase0 used24,360 of29,817observed training cameras; phase1 used58,102 of
59,895. Both phases froze at their fixed camera budgets, discarding5457 and1793
partial-population cameras respectively and advancing their postdraw RNG/index.
The endpoint won13/24(8natural+5own-hit),11losses,0timeouts,15hits, versus original
initial10/24(8natural+2own-hit). Score/rally+.083333 and return/physicalsecond
+.023859 over83.825seconds are descriptive; the unchanged14/6 gate FAILED.
No replication, final evaluation or threshold relaxation follows this near miss.
Its89,712training+4201fresh evaluation cameras/94,779images took1294.47seconds
shared wall and4316.29summed persistent-task seconds, with zero replay. Independent
cached reconstruction verified all768full rollouts, terminal-score returns,
directions/candidates/manual updates/RNG, phase timestamps/skipped definitions and
actual frozen endpoint exactly in3.66seconds, zero new observations. Endpoint
calibration function-RMS.33240 and noise/policy RMS.75210 do not support exploration
collapse. Artifacts: `/workspace/m1b-experiments/pong-stationary-curriculum-20261016-r1/`.
Optional frozen-DCT SARSA integration is being prepared in the isolated
`/workspace/Fly-Connectome-dct-sarsa` worktree, not allocated as a learning run.


Four workers reduced matched exact-replay wall by18.13% in the parent-owned ABBA
benchmark (4064old cameras, zero novel candidate data). Bulk tensor graph hashing
retains exact graph-v2 bytes, aliases and mutation scans, including NumPy writes;
768old-camera reexecutions and402synthetic scalar-reference checks matched before
integration. Narrow tests37+14 passed; fresh-process16diagnostic cameras+1replay
matched all records/state/chains and complete graph. Timing attribution and
uninstrumented comparisons are stored separately; instrumented microbenchmarks
are not deployment-speed evidence. Final seeds remain unused.

## Cumulative allocation accounting

The fourteen registered learning/evaluation allocations, including the completed
reward-credit diagnostic, committed **530,819 learning/training-fitness +96,995 evaluation transitions**,
plus512 toy calibration transitions. Separate Pong calibration used1024. The
frozen epsilon diagnostic added4289 evaluation camera transitions and95.28seconds
shared wall, with zero learning. Reused controls/calibration are counted only at
their original allocation; exact replay adds no independent competence evidence.

The allocation coordinator walls sum to11,450.88seconds (190.85minutes); their
subprocess walls are recorded separately in the machine-readable ledger. **Neither is CPU time
or real total elapsed time**: internal workers and some allocations overlapped.
These totals exclude separately retained physics feasibility, execution/recovery
proofs, numerical replay, representability audits and uncommitted failed attempts.
Transition counts are durable samples, not image totals; resets add images and
some simple controls skip perception. The machine-readable, SQLite-checked ledger
is `/workspace/m1b-experiments/cumulative-research-ledger-20261016.json` and states
these limits explicitly. The cached-template diagnostics and isolated warm-session execution benchmark
are separately counted diagnostic compute, not added to competence evidence.
