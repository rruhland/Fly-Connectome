# Proposed correction to STP target-comparison protocol

Status: user approved; implementation underway. No new model mechanism or parameter change.

## Why the original guard stopped the experiment

The [approved STP proposal](2026-09-21-short-term-synaptic-context-proposal.md)
requires stopping before training if observation targets change. Preflight checked
all neurons and found downstream differences under both D and F, so training did
not start. All directly injected sensory targets and the scored L3 body82450
target remained exactly identical at all four tempos.

An independent reconstruction from recorded presynaptic spikes, fixed feedforward
weights/signs and anatomical delays reproduced every downstream target after the
initial history boundary with zero error. The changed targets are the consequence
of changed physical transmission and spike timing, not a new target encoding.
The unit-release control reproduces baseline learning bit-for-bit.

Requiring every internal observation to remain unchanged was too restrictive for
an experiment explicitly changing physical synaptic dynamics. That restriction
is appropriate for an expression-only identity check, but it prevents this STP
experiment from reaching its actual learning test. This is a protocol defect,
not evidence for or against STP's ability to learn prediction.

## Exact amendment requested

Replace the all-neuron observation-equality gate with both of these requirements:

1. Rendered/event-camera inputs, every directly injected sensory target, and
   the scored L3 body82450 target must match the uncoupled baseline exactly.
   Verify this on the complete frozen evaluation sequences, not only preflight.
2. Internal local targets must equal each model's own actual signed feedforward
   arrivals under the unchanged observation rule. Record their differences from
   baseline descriptively; do not use cross-model internal MSE to claim learning
   improvement when the underlying observations differ.

The published task metrics and selection criteria remain on the same fixed L3
sensory-event target. No model is scored against its own easier target, and no
baseline target is injected as an artificial internal teaching signal. Local
learning still uses actual local observations everywhere, as originally designed.

Retain all other approved details: D/F utilization .5 and recovery100ms, physical
area-matched20/5ms synapses, original magnitudes, fixed topology/signs/delays,
8ticks/frame, +8tick local credit, no backprop, existing homeostasis and behavioral
rule, 200mixed trials seed9060, frozen15trials/tempo seed9081, unseen tempo3,
unchanged anticipation/error/quiet thresholds. Confirm only qualifiers with
30trials/tempo seed9082 and the specified unit-release ablation. No new tuning.

The current preflight supplies the required sensory-target equivalence and
downstream reconstruction evidence. After approval, proceed with the two already
planned learning runs rather than repeat the same preflight. Add the narrower
comparison assertions to the evaluation runner and test them before use.

If any directly injected or scored target changes, stop and report as before.
If neither trained candidate qualifies, stop this family without extending it.

## What approval permits

Only this comparison-scope correction and completion of the previously approved
bounded learning/evaluation protocol. It does not approve a new neuron model,
new temporal trace, learned gate, parameter sweep, or promotion to M1A/M1B.
