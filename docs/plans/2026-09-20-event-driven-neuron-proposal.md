# Bounded event-driven neuron and local-state prototype

Status: approved by the user; isolated experimental implementation in progress.
Approval covers measuring regrouped neuron propagation, not adopting changed
spike schedules. Biological parameters and local learning remain fixed.

## Evidence and purpose

The approved geometric learning experiment reaches about 52 Pong frames/s;
the 120fps target remains open. Exact fixed-point sleeping was slower than dense
execution. Only 3.55% of neuron-ticks receive new input, but 94.97% of spikes occur
without same-tick input. A useful event-driven engine must schedule autonomous
activity instead of simply ignoring neurons without incoming events.

Changing only the neuron kernel cannot close the whole throughput gap: dense
local-error, trace/rate and activity-output work also runs every neural tick.
This prototype would integrate lazy neuron state with local-state queries rather
than continually expanding event activity back into dense tick-sized tensors.

## Bounded scope and invariants

- Separate experimental B=1 CPU backend on the existing measured graph.
- Keep every measured edge, sign, delay, resting current, cell-class parameter,
  threshold, adaptation/reset rule and neural clock tick unchanged.
- Keep no backpropagation and the existing local visual and behavioral equations.
- Keep the existing weight synchronization interval and event/retina/Pong stream.
- No epsilon-to-zero state pruning, reduced time resolution, graph pruning,
  weight normalization or lowered activity used as a performance shortcut.
- Retain current native and tensor references and their existing checkpoints.

## Proposed execution

1. Store each cell's last materialized tick and full state. Deliver anatomical
   delayed arrivals with their original within-tick summation order. Wake cells
   for arrivals, injection, refractory expiry, autonomous threshold candidates,
   and parameter/silencing changes.
2. Advance no-input stretches with the closed form of the *existing discrete*
   recurrences, bounded by a measured scheduling horizon. Group ready
   cells by fixed class parameters and use contiguous vector operations where
   that reduces work. Preserve canonical body/edge identity via index mappings.
3. Find the first qualifying **integer neural tick** for autonomous spikes.
   Mixed excitation/inhibition and decaying adaptation can be nonmonotone, so
   do not assume a single monotone voltage curve. Use conservative interval
   bounds and exact tick replay when the earliest crossing is uncertain.
4. Query local prediction/eligibility state on relevant events and synchronization
   boundaries. Use the approved local sums where applicable; retain exact
   behavioral updates initially. Any later lazy behavioral trace/rate extension
   must be separately verified against its original local equation.
5. Materialize complete canonical state before checkpoints and public snapshots.
   Never expose stale state as current. Label experiment provenance explicitly.

For an eligible cell without arrivals, injection or a spike, each current term
decays as x_q(k)=b_q^k*x_q(0), adaptation as A(k)=c^k*A(0), and voltage obeys

    V(k) = a^k*V(0) + R*(1-a^k)
           + (1-a)*sum(x_q(0)*b_q*(a^k-b_q^k)/(a-b_q))

where a is membrane decay, R is rest current, and b_q covers pathway/sensory
decays. Equal-decay cases use the corresponding finite-sum limit. Refractory
intervals and resets split this expression into separate segments. This is a
derivation from the repository's discrete dynamics, not a replacement neuron law.

## Numerical and acceptance limits

Regrouped neuron propagation generally differs from sequential float32
arithmetic. Unlike tiny weight differences alone, those differences can move a
threshold crossing. Approval would authorize measuring this as an experiment,
not accepting a changed spike schedule automatically.

Tests must cover autonomous rest-driven spikes, mixed-sign transients,
nonmonotone threshold approaches, refractory expiry, adaptation, sensory filters,
silencing changes, equal decay constants, clipping in local learning, delayed
arrivals, pruning, and interrupted save/resume. Compare every spike time and
state/weight differences against the reference, including first divergence.

Run balanced whole-training comparisons over at least1000 frames and a longer
development trajectory before proposing adoption. Include queue management,
state queries/materialization, learning and synchronization in timings. Reject
the prototype if overhead erases gains. Do not claim120fps from isolated kernel
timings or claim effective learning from faster execution.

If missed/spurious/shifted spikes occur, report them and retain the reference;
any acceptance of divergent neuron behavior must return to the user. Final
scientific evaluation seeds and milestone learning gates remain untouched.

## Execution ledger

The user's approval explicitly prioritizes exploiting 10-100 tick sleep gaps.
The 1000-frame wake diagnostic finds a median per-neuron mean gap of 30.23 ticks.
Forcing materialization every eight ticks multiplies wakes by 4.20. Therefore
the prototype permits neuron caches across weight synchronization boundaries:
synchronization changes weights used by future arrivals, not currents already
delivered. Arrival delivery still wakes its target before using current weights.
This changes no weight visibility or biological update timing. Checkpoints and
public snapshots still materialize all state. The initial prototype retains
dense activity queries and existing learning to isolate neuron scheduling costs.
