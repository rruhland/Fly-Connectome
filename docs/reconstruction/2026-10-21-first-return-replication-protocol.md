# Independent first-return curriculum replication

Prospective allocation of ONE independent from-scratch training seed, following
the audited [first curriculum result](2026-10-21-first-return-curriculum-results.md).
Its endpoint returned the first incoming ball on 41/64 serves, with 63/114 total
returns, and passed its unchanged development screen. That result remains one
seed. This replica retains every complete, negative, failed or partial outcome;
no seed replacement, best-checkpoint selection or outcome-dependent extension.
User authorization permits continuous iteration, not removal of failed replicas.

## Unchanged procedure

Use the entire [original registered procedure](2026-10-21-first-return-curriculum-protocol.md):
frozen repaired vision and unchanged model, empty long banks/four-step fallback,
exact existing generic DCT transform/fit/rank 183 and radial cap 256, opponent
speed .65 and hold1. No refit, feature, architecture, reward coefficient or
optimizer change. No backpropagation, teacher labels or privileged policy inputs.
Own-hit events are scalar training fitness only. Fresh score-free random
initialization has function RMS .10; no trained or oracle coefficients transfer.
Eight mirrored directions/top four, noise .25, update .10, function-RMS step cap
.10 remain fixed.

| Phase | Fitness | Shared serves / population | Camera cap | Update cap | Wall cap |
| --- | --- | --- | --- | --- | --- |
| First return | mean binary first own hit | 12 / 192 jobs | 120,000 | 16 | 24 min |
| Full rally | mean total own hits | 6 / 96 jobs | 80,000 | 16 | 16 min |

At least two complete updates per phase. Total training cap 200,000 new cameras
and 40 minutes; 60 minutes overall. Evaluation has 20 minutes from the durable
full-rally freeze, including a 180-second annotation/statistics reserve. No wall
extension. Four single-threaded spawned workers use the existing imports-only
GC freeze initializer once; normal garbage collection stays enabled.

First-return jobs retain the entire first-hit camera, then explicitly task-stop;
raw terminal score and terminal/truncation flags remain truthful. They are
single-episode training tasks, not scored wins. Full-rally training and all DEV
jobs retain the original ten-second physics limit and natural termination.

Phase carry derives this replica's own frozen theta, RNG and next global index.
Update counts can differ from the first seed. Never hardcode five first-stage
updates or global index six. Consumed incomplete/unexecuted draws are skipped;
partial populations never update. Complete pending populations reconcile before
budget checks on recovery; only state-committed populations count as used data.
Persisted failures, clocks and frozen boundaries remain fail-closed.

## Independent streams and registration

Training root **2093651519**, optimizer **837263411**, development root
**1179033779** with tag **29108**, random-action root **668291057**.
These roots were declared before expansion. Snapshot all actually declared prior
training/DEV/calibration IDs, including the first curriculum, and reject root
collisions. Training serves use `SeedSequence([2093651519,872631,global_index])`,
rejecting the exclusion snapshot, reserved final IDs and new DEV IDs.
Each population shares its twelve or six accepted serves across all sixteen
candidates; every serve starts a fresh world/camera phase.

DEV draws sequentially from `default_rng(SeedSequence([1179033779,29108]))`,
rejecting exclusions/reserved IDs and duplicates until 64 accepted. Random
control action seeds use
`SeedSequence([668291057,1179033779,serve_index]).generate_state(1)[0]`.
Final roots **308713321, 1203078761, 1983461927** remain unexpanded.

An immutable `seed-recipe.json` pins these roots/tag, this protocol path and the
first run's source commit plus manifest, summary, completion and parent-audit
hashes. The manifest pins the recipe bytes, selected and default protocol bytes,
source/runtime/model, helpers, exact transform/cap and seed exclusions. Phase
manifests own the training root. Default registration must still reproduce the
first run's initialization/directions/candidates/serve draws exactly. Old source
archives remain available for their own historical recovery contracts.

## Fixed final evaluation and interpretation

After both phases freeze, evaluate **64 new shared full-rally DEV serves** for
the same nine cohorts: final endpoint; its OWN from-scratch initial; constant
down/up/no-op; random hold1/hold10; first-stage endpoint; final endpoint blank
camera. 576 jobs, maximum 288,000 evaluation cameras. Learned policies are frozen
greedy hold1, with no updates or first-hit cutoff. Simple controls may use the
previously verified identical physics/action path without perception.

The final endpoint alone faces its own initial plus five simple controls.
Unchanged primary screen: BOTH first-incoming success and all-incoming return
fraction must have point lift at least **20 percentage points**, paired 95% CI
lower bound **above zero**, and zero invalid draws. The CI need not lie wholly
above 20 points. Use 10,000 paired serve-cluster draws with seed 193870291;
recompute the strongest control separately for each metric inside each draw.
All nine cohorts must match initial incoming geometry. Later opportunities are
policy-dependent; report raw counts, repeat-return rallies, hits/physical time,
losses/timeouts/scores, opponent reliability and all compute costs.

Prespecified descriptive secondary contrasts: phase0 minus own initial,
endpoint minus phase0, endpoint minus own initial, and live minus blank, for
both metrics. These do not change qualification or select a stage. Blank is OOD
whole-input dependence, not proof of ongoing tracking or forecast benefit.

A valid positive replica supports two independent training seeds; it does not
complete three-seed evidence, score winning, visual feedback necessity, forecast
ablation, the distinct visual task, or online-vision retention work in M1B.
No final streams are consumed here. A negative/partial replica is retained and
reviewed before another justified allocation, rather than silently replaced.
Parent independently audits raw physics, scalar fitness, both optimizer/RNG
lineages, task stopping, initial geometry and paired bootstrap calculations.
