# Fixed continuation of hit-return learning from theta9

The first search and its larger frozen confirmation failed their unchanged
acquisition screens. Theta9 uses visual input, but its lift over original theta0
is modest and uncertain. This new bounded experiment tests additional reward
learning in the **same class and rule**, with no teacher or backpropagation.
It does not revise either previous outcome or consume final streams.

## Lineage and learning allocation

Start from the exact theta9, transform and cap of
`pong-direct-hit-policy-search-20261021-r1`. Preserve frozen repaired vision,
empty long banks/four-step fallback, .65 opponent, hold1 and ten-second rallies.
Keep eight mirrored directions/top four, noise .25, normalized step .10 and
function-RMS step cap .10. Fitness remains mean total own hits over six shared
fresh-phase serves per generation; terminal scores do not enter fitness.

**Skip unused global draw g09** and its 14,312 observations. Use its retained
post-draw optimizer RNG and begin at global generation 10. Local continuation
update 0 is old theta9; after N new updates the cumulative learned-update count is
9+N. Global draw index is different because g09 was skipped. Never reuse that
partial population, old DEV results or diagnostic anchors as fitness.

Retain training seed 1563241907 and serve recipe
`SeedSequence([1563241907,872631,global_generation])`. Draw six unique IDs in
`[1,2147483647)`, rejecting frozen prior/calibration/reserved IDs and the new DEV
cohort. Direction draws consume only the inherited independent optimizer RNG.
Archive every input/output RNG state and candidate coefficient hash.

Stop at 16 complete new learning updates, **160,000 total new training-phase
camera transitions**, or 35 shared training minutes. Total training cost includes
both fitness and diagnostic anchors. Use four single-threaded spawned workers
with the validated imports-only GC initializer; normal GC remains enabled.
Freeze the last complete learning update, never the best. At least two new
complete updates are required for a valid new endpoint screen. Incomplete
fitness populations never update. Keep consumed draw/RNG and partial rows.
On recovery, reconcile a fully verified complete population before testing the
remaining camera/wall budget, including a crash between result and optimizer-state
writes. This performs no new observations and cannot double-apply an update.
Only state-committed results count as used fitness. Logged fitness failures persist
on restart and disqualify the endpoint; diagnostic failures remain explicit.

## Prospectively pinned progress diagnostics

At every even global generation (10,12,...), pin both the **pre-update current
center** and fixed theta9 readouts on the same six training serves. Store their
source local/cumulative update indices and hashes in the generation definition
BEFORE any fitness rollout. At initial identity, execute one six-job roster and
reference it as both diagnostics. Exclude anchors from fitness even if another
candidate happens to be identical.

Commit a completed 96-cell fitness update before executing its diagnostic roster.
Anchor rewards never affect optimization, checkpoint choice, stopping or future
cadence. Report paired current-center-minus-theta9 hit counts/means without
calling rotated population totals a learning curve. A partial diagnostic remains
partial and cannot roll back or select the last completed learning update.
Persist pending diagnostic phase so a restart cannot redraw/reapply learning.

Separate completed-population fitness rows, unused partial fitness rows and all
diagnostic rows. Their sum is the total training-phase camera budget. Do not call
diagnostic rows unused fitness. Keep all source/configuration/runtime/journals,
timings, failures and partial work; actual recovery retains every original clock.

## New frozen DEV evaluation

Before expansion, fix DEV `SeedSequence([1718230093,29106])`. Draw 64 unique fresh
serve IDs in `[1,2147483647)`, rejecting the pinned metadata-only prior/calibration/
reserved seed snapshot. Random action seed i is the first uint32 from
`SeedSequence([1880921331,1718230093,i])`. No old32/64 cohort enters training.

Nine cohorts share all 64 ten-second fresh-phase serves. PRIMARY: new endpoint,
original theta0 and five simple controls (down/up/no-op/random1/random10).
SECONDARY: inherited theta9 and new endpoint with the existing blank-camera
boundary. Blank and theta9 never enter the primary comparator maximum.

The primary screen remains at least 20 percentage-point lift on BOTH first-incoming
success and all-return/opportunity fraction, both paired 95% lower bounds above
zero and no invalid draws. Use 10,000 paired serve-cluster resamples, seed 193870291,
recomputing the six-control maximum inside each draw. Preserve first-trajectory
geometry checks across all nine cohorts. Report all scores, timeouts, durations,
return streaks, physical returns/second and opponent reliability.

Report new-minus-theta9 and full-minus-blank paired point effects and intervals
separately. Positive primary qualification alone does not demonstrate extra
continuation learning: that claim requires both new-minus-theta9 lower bounds
above zero and no invalid bootstrap draws. Blanking supports only whole-input dependence under OOD removal,
not object/forecast-specific benefit or ongoing tracking/feedback.

Evaluation allows 20 minutes from a durably recorded phase start, capped by 55
minutes overall from the original continuation manifest. Reserve 180 seconds for
annotations/statistics; maximum 288,000 evaluation transitions/576 jobs. No deadline
extension. Partial evaluation stays partial; work ending outside the overall cap
cannot be labeled within-allocation complete. Controls may use validated
perception skipping, not for an inference-speed claim.

Before launch, run narrow exact lineage/mapping, sidecar-contract, full-fitness
versus-anchor accounting, crash/recovery phase and partial-budget checks. Commit
and archive source/protocol before the manifest clock. Parent independently
reconstructs optimizer draws/updates, all physics and paired statistics. This
remains one trained lineage, not independent seed reproduction or completed M1B.
Final streams 308713321/1203078761/1983461927 remain unexpanded and unused.
