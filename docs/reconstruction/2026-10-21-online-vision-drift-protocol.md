# Persistent visual learning with fixed Pong policies

The forecast pair and both independent audits are complete. This allocation was
reviewed after a bounded actual-bank-growth runtime preflight. Staging does not
start its clock; start only after disk clearance and source/archive pins.

## Fixed allocation and seed recipe

The overall wall cap is **1,800 s**, including scoring: exposure **240 s**, frozen
probes **1,140 s**, prequential streams **240 s**, scoring reserve **180 s**.
Each phase starts at the preceding durable frozen boundary (exposure starts at
the original allocation timestamp). Absolute phase/overall deadlines never reset
on recovery. Stop earlier when complete; partial outcomes receive no extension.

Seed roots are exposure **1793074213**, probes **1360749281**, prequential
**927415613**, with expansion tag **29112**. In phase order, create
`default_rng(SeedSequence([phase_root, 29112]))`; draw positive 31-bit integers,
rejecting all frozen exclusions, reserved final roots, all three phase roots and
previous draws, until 64 seeds per phase are retained. Both policies share the
same schedule. Bootstrap seed **170953167**, 10,000 paired resamples.
Freeze exclusion metadata before expansion, including all 65 runtime-preflight
IDs and recorded implementation-fixture/archived-bridge IDs. No final streams
are expanded. Exactly two predetermined qualified policies are included.

The bank-growth preflight reached 2,000 learning transitions in 93.63 s; its last
200-camera mean heartbeat interval was 38.17 ms, and a separate 32-transition
frozen probe with those learned banks averaged 58.67 ms. Parent auditing ran
concurrently, so these are conservative allocation estimates, not isolated
optimization claims. Preflight images are separate diagnostic computation.

## Fixed system and question

Use both independently qualified original/replica FULL endpoints, regardless of
the third FULL/state-only results. Keep each readout, original transform, cap 256,
observer, camera cadence, legal actions and .65 opponent unchanged. No policy
updates, backpropagation, teacher labels or privileged policy inputs occur.
Only the existing visual context/dynamics/calibration learner is enabled.

This tests integration and retained skill after a specified exposure dose. It
does not establish acquisition with an adapting policy or cross-game continual
learning. Closed-loop exposure trajectories can differ between policies and
between learning modes; they are part of the treatment.

## Prospective finite sequence

1. For each fixed policy, export an owned scene-reset BEFORE snapshot of the
   common promoted visual model, then run exactly 2,000 transition cameras with
   visual learning enabled. Each actual initial image also learns and is counted
   separately. Preserve one visual memory across episodes; each episode uses its
   explicit fresh seed and fresh camera phase. There are 64 seeds per stream,
   ten-second episode caps, and no reward-dependent stopping/checkpoint selection.
2. Retain the exact AFTER snapshot at the dose boundary, including learning from
   the last image. Ending mid-rally is expected: completed dose and completed
   episode are separate fields. All raw final flags stay true to physics;
   a task-boundary stop is neither a score nor an artificial game termination.
3. On 64 new shared fresh serves, run frozen BEFORE and AFTER clones for BOTH
   policies: 256 single-rally jobs. No visual/policy updates occur in these probes.
   Every job starts with its own snapshot clone; probes never modify the snapshot.
4. Separately start each AFTER snapshot on a new ordered seed stream and run
   exactly 2,000 cameras with visual learning still enabled. These are two
   descriptive prequential streams, not independent episode trials. Do not discard
   the last partial episode's hits, time or learned updates.

Exposure, probe and prequential cohorts are disjoint from one another, all prior
actually expanded IDs, calibration IDs and reserved final roots. Both policies
share each prospectively declared cohort. Final streams remain unexpanded.
The precise seed-expansion recipe, exclusions, source/runtime, artifact bytes,
observer/readout/T/cap and all phase/overall wall caps are pinned before launch.
Durable phase start/deadline fields cannot reset on process restart. An incomplete
dose or probe remains partial; no seed replacement or wall extension is implied.
Final authoritative outcome pins preserve the original scoring-inclusive wall
eligibility on zero-work reopening; later verification cannot re-score against a
new wall clock. Partial/failed phases cannot become a complete study merely
because report files exist.

## Outcomes and interpretation

Primary frozen-probe quantities are AFTER minus BEFORE first-incoming return
probability and all-opportunity return fraction, averaging the two fixed policies
equally. Bootstrap the 64 shared SERVE clusters, retaining all four policy×snapshot
cells per cluster. Report each policy separately as well. This is inference for
this fixed policy set, not across-training-seed population inference.
There is no newly declared acquisition screen or formal noninferiority claim;
in particular, CI crossing zero does not demonstrate retention. No simple-control
margin from another cohort is reused as a current-cohort competence result.

Report absolute first/repeated returns, raw hits per physical second, losses,
timeouts and opponent reliability. Prequential/exposure summaries separately show
completed episodes and the right-censored final episode, including missing first
encounters. Sum raw hit records for whole-dose totals; inherited evaluator hits
sum completed rallies only and must not silently omit partial-rally contacts.
No IID episode confidence interval is used for a persistent learned stream.

Report actual learning-image counts (inherited, current and cumulative), short/
long/context bank sizes at segment boundaries, latency distributions, RSS and disk.
Long banks start empty and can change eight-step query eligibility after 32 samples;
this is an existing model transition, not a new predictor. Existing calibration
summaries are conditional observed-endpoint evidence, not hidden-state accuracy.
Do not add a new forecast-scoring system for this initial integration experiment.

## Integrity and resource contract

All mutable memory, private bank RNG/seen/capacity, calibration and pending scene
state remain in each per-camera durable graph hash/checkpoint. Scene resets retain
memory but retire unfinished scene credit; exported snapshots are full owned
copies with explicit reset semantics. Observer/policy/scaler/config stay pinned.
Default frozen operation remains unchanged. Dose-boundary snapshot export is
idempotent after a crash between the last camera commit and file export; a later
state may never fill a missing earlier checkpoint.

Validate interruption during real learned-bank growth and across an episode reset,
snapshot storage independence/tamper rejection, cumulative exposure provenance,
actual frozen-policy default parity, and zero-work completion recovery before
allocation. Count all preflight observations separately. Runtime preflight is
deferred until the active candidate workers exit; no performance or camera
fixtures run concurrently with the frozen forecast allocation.
