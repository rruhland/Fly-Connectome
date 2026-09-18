# Learning-effectiveness audit

The user requested effective M1 learning. Audit development seeds are 1101/1102,
separate from training seed 1 and prior report seeds 1001/1002. Each frozen run
uses 500 scripted Pong steps with no plasticity. The diagnostic wrapper returns
unchanged network-step results, and a fixture test verifies agreement with ordinary
evaluation and unchanged source checkpoint bytes. It never runs during training.

## Event prediction versus quiet-period suppression

`scripts/audit_prediction.py` separates actual events from quiet samples. For the
500-step trained motion-v2 checkpoint:

| Population | Active event samples | Event-conditioned MSE | Quiet-sample MSE | All-sample model MSE | Zero MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | 834 | 1.00930 | 0.000209039 | 0.00115252 | 0.000934978 |
| L2 | 844 | 0.999608 | 0.000113150 | 0.00105780 | 0.000945129 |
| L3 | 834 | 1.00000 | 0.000001275 | 0.000936251 | 0.000934978 |

The zero predictor has event-conditioned MSE exactly 1. Training reduces L2's
quiet-sample MSE from 0.000490818 to 0.000113150, while its event-conditioned MSE
only changes from 1.000163 to 0.999608. L1 event-conditioned error worsens. This
does not demonstrate useful event anticipation; the aggregate improvement primarily
reflects suppression of false predictions during quiet periods. Both initial and
trained full audits are saved as `2026-09-18-learning-audit-{initial,500}.json`.

## Target mismatch

The approved objective predicts clipped filtered currents. The event score tests
unit impulses. A durable two-neuron counterexample verifies that an isolated -30
sensory impulse leaves current -19.7772 at the next frame after 8 neural ticks;
the local target is -1 despite the event target being zero. The local target stays
saturated for approximately 65 ticks. No extra training can make those two targets
identical. The event metric has no future-observation leakage; the mismatch is
the quantity being forecast, not when the new frame is exposed.

The current eligibility adds present arrivals before scoring the preceding forecast.
Its 1-second decay also extends far beyond the 5-ms synaptic current. These semantics
match the approved rule; changing them is a scientific revision, not a bugfix.

## Inactive measured feedback

Of 892 L1 neurons, 884 have anatomical inhibitory prediction input but only 2
receive spikes from such a source in the frozen Pong audit. Most retained inhibitory
contact mass onto L1 comes from C2 (24,280 contacts); C3 provides 61,703 onto L2.
These counts use the same T5 measured subgraph and stage partition as the checkpoint.

`scripts/trace_motion_inputs.py --populations C2,C3,L4,Dm12,Lawf1` audits actual
arrival currents using frozen edges and matched no-event controls. The saved
`2026-09-18-feedback-input-audit.json` shows:

| Population | Peak recorded voltage | Spikes across ON/OFF/control conditions |
| --- | ---: | --- |
| C2 | 0.0496 | 0 / 0 / 0 |
| C3 | 0.0387 | 0 / 0 / 0 |
| L4 | 0.1866 | 0 / 0 / 0 |
| Lawf1 | 0.1092 | 0 / 0 / 0 |
| Dm12 | 0.9992 after resets | 18 / 16 / 0 |

Threshold is 1, with zero intrinsic current in these classes. The measured paths
exist but many are functionally unavailable. Dm12 is not globally silent and must
not be reported as such. No topology or signs were changed in these audits.

## Decision and controls

The proposed bounded correction is
`../plans/2026-09-18-event-aligned-learning-proposal.md`: a local event-increment
target, causal visual eligibility, and frozen calibration of existing feedback
classes. The user approved its implementation and experiments on 2026-09-18. Separately, the
unchanged rule is being continued from 500 to 2,000 steps, saving every 100 steps,
to test whether more training resolves the observed failure.
