# Proposed next throughput experiment â€” approval required

Status: proposal only. The implementation continues to perform every local update
on every neural tick. Do not implement this experiment without user approval.

## Evidence and target

The exact CPU backend has improved the measured M1A graph from about 3 to
49–50 Pong frames/s in a balanced 100-frame comparison. The final1000-frame
profile took20.545 seconds (48.67fps). The graph has47,413 neurons and1,377,103
measured edges. Each frame still contains eight neural ticks. The goal of120
frames/s remains unmet;10,000 frames project to3.4 minutes, excluding load/save,
rather than an hour. This is computation cost, not waiting for physical events.

The final1000-frame profile attributes10.49 seconds to learning,4.64 to neural
execution,0.65 to synchronization, and4.76 to other frame work. Improving sparse
eligibility alone cannot achieve120fps. Further frame orchestration work and
neuron execution improvements would also be necessary. These timings are not
proof of a hardware ceiling.

## Bounded execution revision

Experiment with deferred evaluation of visual eligibility and prediction-error
proposals *within the existing weight synchronization interval*. Weights must
become visible on precisely the same ticks as today. Neurons, input events,
refractory state, and behavioral R-STDP keep their existing tick updates.

Limit the first experiment to the approved `input-arrivals-v1` target with
`forecast-causal-v1` visual eligibility. Between relevant local arrivals, an
unclipped prediction and its visual eligibility decay geometrically. Their
product therefore has a geometric sum. Accumulate that sum when a relevant
local event occurs or immediately before the existing synchronization boundary,
instead of visiting the same quiet synapse on every intervening tick.

This is an execution change to the same local equation, not a new objective.
For a quiet interval with initial prediction p, eligibility e, common decay d,
and m ticks, its real-arithmetic contribution is

    -eta * p * e * sum((d*d)**k for k in range(m))

The implementation must derive the indexing from the reference's score-before-
decay convention. Interrupt an interval on any relevant presynaptic arrival,
postsynaptic sensory/feedforward target event, or recurrent prediction-current
arrival. Handle clipping transitions and eligibility pruning on their original
logical ticks; use the existing tick implementation wherever that cannot be
established. No events may be ignored because they are small.

## Invariants and numerical limit

- No added or removed edges, changed signs, weight normalization, backpropagation,
  nonlocal errors, learned readout, teacher, or changed learning equation.
- Same eight neural ticks per Pong frame, local information boundaries, and
  existing weight synchronization schedule.
- Behavioral R-STDP and neural dynamics retain their current implementation.
- Materialize private traces/proposals before checkpoints and public inspection.
- Keep the exact CPU and tensor implementations available as references.

Closed-form accumulation changes float32 rounding order. It therefore does not
promise bit-identical weights, unlike the current exact backend. This numerical
and execution change is the reason for requesting approval. It must remain a
separately identified experiment, not silently replace existing checkpoints or
be reported as successful scientific learning.

## Verification before any adoption

1. Hand-derived and randomized local sequences: positive/negative arrivals,
   cancellation, clipping transitions, pruning crossings, empty intervals,
   synchronization, and save/resume. Compare against per-tick reference sums.
2. Record maximum and distribution of state/weight differences, every spike
   time, and event scores on identical fixed trajectories. No relaxed claim of
   spike or behavioral equivalence if they diverge. Report divergence before
   proposing adoption; retain the exact backend.
3. Profile whole-frame online learning, including interval bookkeeping and
   synchronization. Reject the experiment if bookkeeping erases the gain.
4. Use at least 1000 timed frames and repeat a balanced comparison. Do not
   extrapolate a sparse-kernel gain to whole-model realtime performance.

Approval would authorize this bounded experiment, not a changed biological
model or automatic adoption of numerically divergent training. Useful learning
and the milestone gates remain separate acceptance requirements.

Relative weight normalization is not part of this proposal: it changes current
amplitudes but does not reduce the number of edge operations. It could change
activity and runtime indirectly; neither a speed nor learning improvement has
been established for it here.
