# Durable engineering checkpoint and proposed toy gates

This is new engineering on ecb4194, while independent scientific review of the
learner remains pending. The Sarsa math is unchanged. No toy or broad training ran.
Only controlled correctness/recovery diagnostics updated policy parameters.

## Immutable bank and complete continuation

FrozenCodec extracts observer weights/bias, context keys/values/metric, and short/
long forecast key/value tensors into one deterministic, content-addressed bank.
Snapshots reference those tensors through pickle persistent IDs. Repeated object
references resolve to the same tensor, including the variance-cache key alias.
Mutable history, calibration, all RNGs, weights/traces/q_old, pending action and
camera/physics/cadence remain in the active state. A content check refuses a new
snapshot if a declared frozen tensor changed. Non-CPU tensors and tensor views
are rejected instead of silently changing storage-view aliases.

The canonical graph comparator moved from the diagnostic script into control/codec
without changing its comparison rules. The Session restore validator is shared by
full and deduplicated checkpoints. No learner update rule changed.

## Interval snapshots and verified replay

DurableRun uses a separate replay-v1 schema; the earlier Journal remains intact.
SQLite WAL and synchronous FULL are retained. Each action transition appends one
sample with contiguous sequence, hash chain and canonical post-state digest.
Snapshots occur every 100 transitions or 10 wall seconds since the last snapshot,
and on a clean stop. One initial snapshot permits replay before the first interval;
only one active snapshot and one immutable bank are retained.

When a checkpoint is due, sample insertion and checkpoint replacement commit in
one transaction. Otherwise the committed sample is sufficient to reproduce that
transition deterministically from the preceding checkpoint. Restore verifies bank,
payload, sequence and hash chain, then re-executes every committed tail transition
and compares BOTH its sample record and complete canonical state. A mismatch fails
closed. A failed writer is poisoned: it cannot continue its advanced in-memory
state or take a stop checkpoint; reopening recovers the committed prefix.

The source/model/configuration checks remain strict. Replay metadata additionally
pins Python, NumPy, Torch, platform and machine. A different runtime requires a
separate migration proof; it is not silently accepted.

## Crash and timing evidence

The final fresh-process proof compares 16 uninterrupted transitions against:
9 committed transitions, process exit inside an uncommitted SQL write, restore
checkpoint 7, replay transitions 8 and 9, then execute the remaining 7 transitions.
All committed sample rows and the final state hash match exactly. There are no
missing or duplicated committed transitions. Final policy has 259 nonzero weights,
134 active traces and a pending action. One frozen bank occupies 520497 bytes;
active snapshots at transition 16 occupy about 1.23 MB, dominated by retained
transient state. Serialization bytes need not be identical across processes;
canonical object state and audit records are identical.

The final proof took 7.84 seconds including child process startup/shutdown. It
executed 32 new transitions plus 2 replay transitions. A previous run before adding
the explicit runtime pin also passed the same 34-transition proof. All integration
proofs enforce a 180-second deadline; no long experiment was launched.

Timing events record starts, committed heartbeats and clean stops, with epoch wall
time plus cumulative monotonic process-prefix cost. Restart gaps are separate,
signed wall-clock intervals; negative clock movement is not silently clamped in
the gap record. Following a crash, the unobserved gap includes any execution after
the last heartbeat and must not be labeled purely downtime. Prefix accounting
begins at DurableRun initialization and includes restoration/replay and subsequent
snapshot/I/O work; it excludes fresh Session construction before that call and the
final timing-row commit/teardown. The supervisor's full subprocess wall time is
reported separately, so prefix costs are not misrepresented as total compute cost.
The proof JSON captures timing immediately before stop; the database then records
the clean-stop event. Tests separately assert clean-stop and seven-second gap data.

Focused result: 42 passed. Coverage includes tensor alias/RNG identity, immutable
mutation rejection, interval counts and elapsed-time snapshots, stop snapshots,
committed-tail replay, snapshot transaction rollback, poisoned writer, corrupt
chain, valid-chain-but-wrong replay content, runtime mismatch and timing gaps.

## Remaining cloud execution limitations

Recovery is explicit, by opening the SQLite database in this cloud runtime. It
is not an automatically scheduled or supervised background job. This work does
not establish that a replacement cloud task will retain this filesystem; if it
does not, these local checkpoints are unavailable there. No transfer workaround
was attempted. Source/runtime changes are intentionally incompatible until proved
or explicitly migrated. Same-platform trusted pickle remains the serialization
boundary; neither untrusted pickle nor Windows portability is supported here.
Power-loss filesystem guarantees beyond SQLite FULL were not tested. An uncommitted
final transition may have executed physically but is deliberately re-executed on
restore; only committed transitions belong to the recoverable experiment history.
Audit samples and active state are recoverable, not every discarded historical
full snapshot. Old full-state reconstruction requires retained earlier snapshots.

## Fixed toy proposal — not executed

These are proposed preregistered checks, contingent on the scientific review.
Both use camera/event inputs through the frozen promoted vision and existing
encoder, no oracle controller inputs, no backprop, and terminal score-only rewards.
No forecast-benefit claim follows merely from providing forecasts.

1. Immediate target choice: a 64x64 image shows one high-contrast target on the
   left or right of center. The two actuator actions select left/right. One action
   ends the episode with score +1 for the target side, -1 otherwise. Target side is
   balanced; pixel jitter is generated independently from policy RNG. Proposed
   budget: 200 learning transitions per seed on seeds 2101, 2102, 2103 (600 total),
   followed by 40 frozen held-out choices per seed (seeds 3101–3103). Include random
   and initial-policy evaluations. Proposed competence screen: >=80% correct on
   each seed, with raw counts and uncertainty, not just pooled performance.
2. Delayed cursor: a visible cursor starts at center of seven cells. Left/right
   actions move one cell per decision; a visible target is at one boundary, the
   other boundary is wrong. Reaching a boundary ends with score +1/-1; no movement
   reward. Time limit: 12 decisions, reported as truncation and counted as a failed
   evaluation episode. Proposed budget: 60 learning episodes per seed, at most 720
   transitions each, on seeds 2201–2203; 40 frozen held-out episodes on 3201–3203.
   Compare fixed lambda=0 and lambda=.8, matched seeds/budgets/alpha/features; no
   decay-based tuning. Proposed competence screen: >=70% correct per seed. Report
   the paired trace comparison, without assuming a trace advantage in advance.

Use the already tested alpha=.0001, gamma_per_second=.95, epsilon=.2 as a fixed
starting registration, not as an established good choice. Report failure at that
setting rather than quietly increasing budget/tuning against held-out data. Any
subsequent scale/alpha change must be a separate declared development comparison.
Each execution chunk remains <=200 transitions/180 seconds, with durable stop
checkpoints between chunks. The above budgets are proposals, not work already run
or permission to bypass review. Confidence in Pong learning still requires later
fresh-seed Pong evidence.

## Final regression result

Full suite: **777 passed, 4 skipped, 1 failed in 49.86 seconds**. The only failure is
the independently reproduced baseline native voltage exact-rounding mismatch,
`test_native_neural_step_matches_reference[False]`. Its assertion remains intact.
No changes were made to control/sarsa.py or its update equations during this stage.
