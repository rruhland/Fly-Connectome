# Proposed bounded bridge from context efficacy to Pong M1A

## Decision requested

Approve an **opt-in experimental full-graph M1A bridge**, starting with the
directly event-camera-observed lamina cells. This is an architectural scale-up
of the now-successful controlled rule, not an automatic promotion to the
canonical trainer. No new edge, neuron, transmitter-sign change, learned
delay, backpropagation, reward-driven visual update, or Pong-state input is
proposed. Production defaults and existing checkpoints remain readable and
unchanged. Stop and report before M1B.

The approved live motif experiment cleared both event signs and the quiet
criterion at every tested tempo. Its main remaining limitations are weak
steady slow-tempo ON margin and missed first events after an omission. More
critically, repeated two-position motion is not event-camera Pong. A bounded
open-loop Pong test is now the most direct way to find whether this rule can
support M1A.

## Anatomy and scope

The pinned T5 visual checkpoint has 47,413 neurons and 1,377,103 measured
edges. Of 953,031 predictive edges, 17,557 enter 2,660 directly injected
L1/L2/L3 targets. These cells have the same unambiguous local signed sensory
state used by the controlled experiment. Most predictive targets do **not**
have direct event-camera injection; applying the sign selector to all
953,031 edges would assign an undefined context to many cells and multiply
state unnecessarily.

In this bridge, only the existing predictive edges entering those directly
observed cells acquire two magnitude components. Their context is the local
post-injection sensory-state sign; their physical E/I currents and local
eight-tick credit follow the verified rule. All other predictive magnitudes
are held fixed during this first diagnostic. Their physical recurrent current
still participates in the network. Feedforward observation edges and
behavioral weights remain fixed. Use the already tested 20 ms area-matched
excitatory and 5 ms inhibitory predictive kernels across this experimental
network; no new kernel is tuned on Pong. This isolates whether the verified
local learner transfers to real event-camera input before choosing a context
for indirect visual cells.

## Dependency-ordered checks

1. Generalize the one-target experimental code to vectorized, private
   per-neuron context/current/timing state for the 2,660 eligible targets at
   batch one. Keep the two magnitude components only for their 17,557
   existing incoming predictive edges. Require equal-component parity against
   an area-matched full-graph reference for spikes, physical currents,
   eligibility, forecasts, local targets and weights. Include sign/order,
   closed-gate surprise, issue-context credit, no non-target update, bounds,
   finite state, and exact checkpoint resume tests. No training until parity.
2. Run a short, capped headless benchmark on the full graph with real Pong
   event-camera frames at eight neural ticks per frame. Record camera frames/s,
   active edges/eligibility, memory, and firing. Only optimize software
   overhead/memory traffic if necessary; preserve the exact model. Do not
   silently reduce graph, neural ticks, or synaptic state to gain speed.
3. Train a bounded open-loop M1A development run with camera input only and
   the existing scripted paddle used in M1A. Use the controlled rate 1.0 as
   the first predeclared candidate; at most compare it with .1 on development
   validation. Keep source checkpoints and held-out trajectories fixed and
   final seeds 1201-1204 untouched. Preserve membrane, camera, eligibility,
   timing and weight state across rallies, including reset-generated events.
4. Evaluate frozen initial versus learned weights on untouched Pong initial
   conditions, separately reporting ON, OFF and quiet targets for directly
   observed cells, next-frame event MSE versus zero and persistence, temporal
   alignment, firing stability, and T4/T5 direction/velocity-related
   anticipatory activity. Compare equal-component and learned neural spikes.
   Report training throughput and exact local update reconstruction. An
   aggregate low error dominated by quiet pixels is not an M1A pass.

If parity fails, fix implementation ordering before training. If the short
benchmark is too slow for a bounded run, stop and profile exact-model
software costs. If the directly observed cells learn but downstream
anticipation does not, investigate local context for feedforward-observed
neurons separately; do not guess a sign selector or change all remaining
synapses. If Pong event prediction fails, preserve the controlled result and
diagnose the mismatch before more training. This bridge can supply M1A
evidence, but full M1A acceptance and M1B advancement still require the
original design's criteria and separate verification.
