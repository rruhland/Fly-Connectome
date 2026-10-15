# M1B cloud reconstruction: stage 1 review checkpoint

This is **new work**, based on cloud commit
`b52f06e9a4cf1664d8429e7098a6371eafb77bc7`. It is not recovery of the unavailable
M1B commits, learning results, or weights. No Library downloads are needed or
attempted by this reconstruction. Historical M1B success and historical test
counts supplied in the handoff are not evidence for this implementation.

## Baseline and gaps

The cloud baseline supplies Pong physics/rendering, the promoted probabilistic
visual service, its generic sparse transport encoder, and legacy connectome
training. The legacy trainer is not the missing M1B true-online Sarsa learner.
The described M1B action learner, forward-view reference, action/reward boundary,
SQLite continuation journal, and frozen action-value evaluation were absent.

The tracked model is present and matches the baseline's own production test:
`src/fly_connectome/vision/data/default.pt`, 680428 bytes,
SHA256 `a899338c94278854dc9a790c21e3cba746658abb7ac91fc3f85c84754f253173`.
This establishes baseline artifact identity, not identity with an unreceived
handoff manifest. The source worktree is `/workspace/Fly-Connectome-m1b`, branch
`reconstruction/m1b-foundation`; the original `/workspace/Fly-Connectome` remains
on `work` at the baseline.

## Implemented now

- `FrozenLinearPolicy`: nonzero linear Q values from sparse encoded inputs,
  explicit legal-action mask, deterministic ties, defensive parameter ownership,
  no updates/exploration/RNG consumption. Actuator meaning belongs to the caller.
- `Journal`: SQLite WAL with synchronous FULL; compare-and-swap sequence checks;
  atomic sample plus active continuation transaction; SHA256 sample chain and
  active payload verification; content-addressed immutable artifact storage.
  Exactly one active continuation is retained. State bytes are opaque to the store.
- `prove_m1b_frozen_recovery.py`: CPU Pong + frozen promoted vision + existing
  encoder + fixed nonzero diagnostic policy; only score rewards, no hidden-state
  controller inputs. Physics, camera history, perception/encoder object graph,
  cadence, counters, policy, and Python/NumPy/Torch RNG state survive a trusted,
  same-platform pickle continuation. An explicit physics/body alias is checked.
- Linux `-fPIC` flags for the two experimental native shared-library builds.
  Their baseline linker failures were reproduced before fixing flags.

## Evidence and boundaries

The 48-transition uninterrupted trajectory matches a 24+24 fresh-process resumed
trajectory exactly: sample/Q/action/reward/RNG audit and canonical continuation
graph digest. The successful proof executes 96 total transitions in 15.29 seconds,
including process startup and journal serialization/I/O. See the adjacent JSON.
Across all diagnostic attempts in this reconstruction, 193 integration transitions
were executed; all were frozen diagnostics. Development learning observations
remain 0/0. No convincing Pong learning is claimed.

Two earlier diagnostic attempts each completed 48 transitions but incorrectly
compared inference cache state as though it were learned parameters. A one-step
probe isolated the issue. The corrected learned-state check covers the observer,
context keys/values/metric, and dynamics banks/calibration. Derived variance caches
remain in continuation state. Their Torch version counters are represented by
cache-validity equivalence in the digest because raw tensor version counters are
process-local. No numerical tolerance is used for trajectory/state comparisons.

Focused foundation + native portability checks: **31 passed** (16 new foundation
tests and 15 existing native experiment tests). Coverage includes wrong sequence,
stale second writer, hash corruption, injected SQL failure between sample and
checkpoint, process exit during an uncommitted transaction, artifact corruption,
and nonzero frozen Q values without policy mutation.

The bounded proof is a continuation test, not a durable learning proof. Specifically:

- No reconstructed training algorithm, Dutch traces, q_old, continuous-time credit,
  pending-action reuse, or truncation bootstrapping is implemented yet.
- The frozen runner checkpoints at completed action boundaries. Its pending action
  is empty at each checkpoint; resumption mid-action has not been demonstrated.
- No scoring/reset event is guaranteed within this short natural trajectory.
  Termination/reset and truncation need dedicated tests before learner integration.
- Model bytes are stored once by hash, but the active pickle still embeds immutable
  visual banks. Deduplication of those banks is not yet implemented.
- Audit samples are reconstructible from the journal; old full states are not.
  Deterministic delta replay, periodic snapshots at 100 transitions/10 seconds and
  stops, persisted wall-time gaps, and full experiment cost accounting remain gaps.
- Pickles are only for trusted checkpoints from this Linux source/runtime, with no
  Windows compatibility claim. Hashes detect corruption, not malicious tampering.

## Next stages before development experiments

1. Review this storage/evaluation foundation and its limitations. Preserve the
   baseline native exact-equality regression as a visible Linux compatibility gate.
2. Implement sparse true-online Sarsa with Dutch correction, elapsed-time discounts,
   explicit episode/termination/truncation semantics, and pending-action reuse.
   Test each update prefix against an independent forward-view implementation.
3. Integrate complete learner continuation and immutable-bank references, and test
   abrupt interruption/recovery including RNG aliases and nonempty pending action.
   Add interval snapshots, replay validation and timing-gap/cost records.
4. Re-establish small-task learning evidence on fresh seeds under frozen vision,
   score-only reward and no backprop. Only then register a bounded Pong experiment.
   Do not select lambda from nominal decay or claim historical results as new proof.

No public push, merge, paid compute, or access-policy change was performed.

## Final Linux suite result

After installing the project into the isolated environment and correcting the
Linux linker flags: **751 passed, 4 skipped, 1 failed in 46.40 seconds**.
The remaining failure is
`tests/test_native_cpu.py::test_native_neural_step_matches_reference[False]`:
voltage tick 1 differs by at most `1.1920928955078125e-07` (exact-equality gate).
The untouched baseline reproduces that failure under the same runtime. Its
assertion was not relaxed and the native arithmetic was not changed. This is a
remaining compatibility blocker for declaring the entire Linux suite green,
not a failure of the newly added frozen control path.

Runtime: Python 3.12.14, NumPy 2.0.1, Torch 2.5.0+cpu, SciPy 1.18.1,
PyArrow 26.0.0, pytest 9.1.1, matplotlib 3.11.2. Interpreter:
`/workspace/venvs/fly-m1b-cpu/bin/python`. The runtime dependency freeze and raw
validation logs are in `/workspace/m1b-verification/`. No actual assistant model
identifier or reasoning setting is exposed by the execution runtime.
