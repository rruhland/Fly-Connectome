# Selected-lineage continuity repair and fixed reward-learning screen

The user authorized continuous M1B iteration on 2026-10-21. Before the planned
larger policy search, saved-action replay identified a correctness defect:
one tracker beam slot alternated its selected lineage between distant objects,
while the public ID made encoder velocity and forecast differences treat those
endpoints as one continuous trajectory. At camera 11915 the selected tracker
branch itself had zero velocity, but the encoder reported horizontal velocity
-42.58 field units/second. This repair does not solve ambiguous association.

## Generic repair and prospective version

Preserve every beam hypothesis and its scoring. Compare the winning child's
parent last-observed endpoint and timestamp with the previously published
selected endpoint and timestamp, using exact value equality. A mismatch, including
a winning missed branch, retires the old public routing ID and issues a fresh ID.
Clear that slot's contextual key/confidence before endpoint teaching. Retire old
forecast history, cached forecasts and pending credits; identity uncertainty is
not credited as failed reidentification. Normal continuation, occlusion and
reacquisition with the same parent endpoint remain continuous.

The visual service is explicitly versioned
`+selected-lineage-continuity-v1`. Model weights/hash, beam rules, the 4389-feature
transport layout, generic 214-coordinate DCT matrix, score-free fit and cap256 stay fixed.
Fresh routing IDs are handles, never semantic policy features. Validate observed
and missed lineage switches, ordinary occlusion, name/order invariance, retired
credit/context isolation and durable replay; independently replay both known
outlier rallies plus short ordinary camera traces. Compare fixed-T norms/cap
activation without fitting to reward or oracle coefficients. No candidate clock
starts until these gates and parent review pass.

## One bounded corrected-vision acquisition attempt

From zero weights, train exactly **24,000 camera transitions**, seed **736187003**,
against opponent speed **.65**, with 20-second/2400-tick episode limits. Use the
historical independent arm's exact learner settings: hold1, explicit exploration
block1, true-online SARSA at each camera, alpha .001, lambda .995, epsilon .2 and
gamma .95/physical second. Reward is scalar **+1 own hit only**, zero score and
survival coefficients; raw game score remains separately logged. No backprop,
teacher labels, policy physics inputs or additional learner modifications.

This isolates the repair relative to a prior configuration, but the historical
learning comparison is not paired evidence: changed observations alter actions
and later states. Fresh matched controls establish any current competence.

Keep the exact already frozen DCT transform and fit. Record its compatibility
diagnostics rather than refitting it after the repair. The frozen long banks are
empty, using four-step context fallback. Save/evaluate only the 24k endpoint.
No intermediate evaluation, best checkpoint, seed replacement or wall extension.
One single-threaded training worker has **20 minutes**; the overall allocation has
**30 minutes**, both anchored to durable manifest creation. Reserve 120 seconds
for annotations/reporting, use four evaluation workers, retain every partial or
failed outcome and apply the existing finite/absolute-Q 1000 resource stop.

## Fresh development outcomes and controls

Before any expansion, fix 32 new serve seeds as NumPy default_rng with
`SeedSequence([1287749219,29103])`, 32 integers in `[1,2147483647)` in draw order.
Reject duplicates and reserved-final collisions, never inspect/select serves.
Each policy/serve starts a separate one-episode evaluator at constructor serve,
camera phase zero, 10-second/1200-tick limit. The six cohorts are corrected
endpoint, zero-Q initial/constant down, up, no-op, uniform random hold1 and uniform
random hold10. The random action seed is the first uint32 from
`SeedSequence([447201113,1287749219,i])`. Constants/random use the previously
validated perception-skip path, whose timing is not inference-speed evidence.
Endpoint evaluation is frozen greedy hold1 with raw score only and no updates.
Maximum evaluation allocation is 96,000 camera transitions (6×32×500).

Report actual own incoming collision-plane opportunities and successful returns,
first-incoming success, all returns/opportunities, ≥2 returns/rally, longest streak,
returns/physical second, scores/losses/timeouts, partial approaches and measured
opponent reliability. Verify exact pre-contact first-incoming geometry across
all six policies. No opponent-going serve counts as an own opportunity.

The unchanged promising-development screen requires at least 20 percentage-point
lift over the strongest of all five controls on **both** first-incoming success
and all-opportunity return fraction, with both paired 95% lower bounds above zero
and zero invalid draws. Use 10,000 paired serve-cluster bootstrap draws, seed
193870291; recompute the strongest control within each draw. A one-training-seed
screen is exploratory; it is not reproducible acquisition or complete M1B.
Reserved final streams 308713321/1203078761/1983461927 remain unexpanded.

## Recovery, evidence and continuation

Retain per-camera full journal commits, bounded snapshots, complete recovery
validation, report/semantic-journal/mapped-endpoint completion pins and zero-new-
work verification. Pin source, protocol, runtime, model, exact transform/fit,
serve/action seed recipes and all costs; archive source before launch. Parent
independently audits raw actions/physics, hit reward/TD, routing-replay evidence,
first-incoming pairing and statistics. Diagnostic reexecution is separate from
novel candidate exposure. After this fixed endpoint, adapt the next bounded
experiment under continuous authorization; no automatic use of final holdouts.
