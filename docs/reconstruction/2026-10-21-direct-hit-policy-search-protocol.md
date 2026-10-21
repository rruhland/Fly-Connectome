# Prospective direct hit-return policy search

This fixed experiment follows the independently verified negative repaired-vision
SARSA endpoint. User authorization is continuous; no backprop, teacher policy
inputs or major architecture changes are introduced. Source/protocol are frozen
before the candidate manifest starts its clock.

The next hypothesis is that direct greedy-policy optimization can learn returns
without fitting Bellman Q values. Prior mirrored-search failures optimized
terminal game score, not direct hit return, and predated truthful selected-lineage
continuity. This is an acquisition experiment, not an oracle capacity witness.

Keep the exact existing score-free DCT matrix/ridge/rank 183 and cap256; frozen
vision includes selected-lineage-continuity-v1, empty long banks and four-step
fallback. No refit, oracle coefficients or policy-class change. Initialize from
scratch with the existing score-free random function-RMS .10 recipe, not zeros.
Use eight antithetic directions, top four, noise .25, normalized update .10 with
function-RMS step cap .10. Preserve the existing zero-variance/tie rules.

Fitness is mean **total own-hit count** over six fresh ten-second training serves.
All sixteen candidates share the same six serves within each generation; rotate
the serve set each generation. Truncation retains acquired hit rewards. Terminal
scores are logged but have zero fitness coefficient. No opportunity, interception,
teacher, object label or privileged state enters the policy or optimizer.
Episode-fixed perturbations provide coherent exploration; inference is hold1.
The .65 opponent and all remaining physics are unchanged.

Before seed expansion, fix optimizer seed 1337923911 and training seed 1563241907.
Generation g uses `SeedSequence([1563241907,872631,g])` for six unique seed IDs in
`[1,2147483647)`, rejecting the new DEV cohort, prior corrected-SARSA DEV cohort
and existing final/calibration exclusions. Optimizer
directions use only the durable independent optimizer RNG; seed generation does
not consume it. Each rollout starts its own constructor serve and camera phase.

Stop at sixteen complete generations, 160,000 new training cameras, or 35 minutes
shared training wall, whichever comes first; four single-threaded workers.
Overall wall cap 45 minutes, with no extension. Freeze the **last complete**
generation, never the best; at least two are required for valid endpoint testing.
Keep all partial observations/costs but never update from a partial generation.
A recorded rollout failure invalidates the endpoint rather than silently
substituting an earlier candidate.
An unused/partial draw is recorded with its post-draw RNG; future continuation,
if separately registered, must explicitly handle that index and state.

Fresh 32 DEV seed IDs are the fixed-order draws from NumPy default_rng seeded by
`SeedSequence([933715231,29104])`. These are distinct from the corrected-SARSA
cohort, not final holdouts. Random-control action seed i is the first uint32 from
`SeedSequence([447201113,933715231,i])`. Seven cohorts/224 one-serve jobs:
endpoint, its own restricted initial readout, constant down, up, no-op, random
hold1 and random hold10. Trained/initial readouts are frozen greedy hold1, with
raw-score-only independent evaluation and no updates. Observation-independent
controls may use the validated perception-skip path. All episodes last at most
1200 physics ticks/ten seconds, so the evaluation camera ceiling is 112,000.

Co-primary diagnostics: first-incoming return success and all own returns/all
incoming collision-plane opportunities. Verify pre-contact first incoming
geometry on all seven cohorts. The promising development screen requires at
least 20 percentage-point lift on BOTH metrics over the strongest of ALL six
comparators including own initialization, with paired serve-cluster 95% lower
bounds above zero and no invalid draws. Use 10,000 bootstrap draws, seed 193870291,
recomputing comparator maxima inside each draw. Report streaks, returns/physical
second, actual time, losses/timeouts, partial approaches and opponent reliability.
This one-seed development screen cannot complete M1B or establish reproducibility.

Training-only diagnostics record per-serve hit contrasts, nonzero mirrored mean
contrasts, reward concentration, contact-bearing rollouts, durations, timeouts,
function-RMS/noise ratio and update sizes. These never select intermediate
checkpoints or change the fixed allocation. Degenerate constant policies and
large unsupported search dimension remain live failure hypotheses.

Reuse the existing phase/reservation engine through explicit default-preserving
population/update/completion callbacks. Retain per-camera commits, complete
actual recovery validation, bounded snapshots and report/semantic-journal/
mapped-policy completion pins. Persist optimizer weights/RNG/config, definitions,
direction arrays, coefficient hashes, raw actions/hits/scores and all costs.
Tests cover variable serves/default equivalence, hit-only update independence
from score, zero variance/ties, partial-generation rejection and restart boundaries.
Pin/archive exact source and protocol before manifest creation. Independent parent
audit reconstructs directions, RNG, updates, raw physics/hits and paired metrics.
The candidate coordinator annotates evaluation opportunities only; full training
physics replay belongs to the separate independent audit and its separate cost ledger.
Final streams 308713321/1203078761/1983461927 remain unexpanded and unused.

## Fixed execution optimization

Each fresh spawned pool worker enables the registered single-thread Torch mode,
collects cyclic garbage, then calls `gc.freeze()` **once before creating any
candidate session**. Normal cyclic collection remains enabled for all later
candidate graphs; no per-frame, per-chunk or repeated-session freezing occurs.
Importing the coordinator does not change GC state. Pin mode
`freeze-imports-once-before-first-session:gc-enabled-v1` in the manifest.
The allocation's bounded workers exit normally and release their process memory.

Independent old-trajectory ABBA replay measured 20.1% loop/18.6% full-fixture wall
reduction with exact records/state/chains; a four-session lifetime check retained
stable RSS after warmup and normal collectibility. Those diagnostics supplied zero
novel candidate cameras. Timing overlapped a single annotation process under a
four-CPU quota, so deployment speedup is an estimate, not a guarantee. The slower
closure-cleanup prototype was rejected; codec, complete state checks and durability
are unchanged. Record actual search throughput and all costs.
