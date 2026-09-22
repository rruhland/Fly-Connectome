# Local context efficacy learns both event signs in the live measured motif

The approved two-component efficacy experiment resolves the main conflict seen
with one shared magnitude. A component chosen by the L3 neuron's *current local
sensory-state sign* controls each of its 12 existing predictive synapses, both
in physical recurrent current and in eight-tick local credit. The graph,
pre/post identities, transmitter signs, delays, excitatory/inhibitory kernels,
timing gate, and no-backprop rule are unchanged. This is an experimental
single-target implementation; the production network and checkpoint format
were not changed.

## Validation and held-out behavior

Equal components exactly reproduced the prior 411-frame neural run: every
spike, signed forecast, observed target, sensory state, timing gate, reference,
and final magnitude. A signed fixture also matched physical E/I currents and
edge eligibility tick by tick. Training used the same 42 continuous trials and
nine validation trials as the shared-magnitude rule. The predeclared rates
were .1 and 1.0; validation chose 1.0 with ON/OFF anticipation .542/.724,
MSE .0483, and 1.31% quiet false alarms. No held-out trial influenced this
selection.

| Untouched condition | Initial -> context MSE | Shared-magnitude MSE | Context ON | Context OFF | Quiet alarms |
|---|---:|---:|---:|---:|---:|
| Dwell 2 | .1738 -> .0934 | .1470 | .324 | .667 | .44% |
| Unseen dwell 3 | .1459 -> .0498 | .0871 | .667 | 1.000 | 2.93% |
| Dwell 4 | .1282 -> .0425 | .0840 | .645 | 1.000 | 2.49% |
| Dwell 6 | .1023 -> .0299 | .0759 | .805 | .455 | 1.68% |
| Continuous switches | .2288 -> .0679 | .1627 | .599 | .798 | .64% |
| Continuous with omission | .2239 -> .0649 | .1551 | .606 | .817 | .95% |

Every held-out MSE is also below zero prediction and event persistence. Every
tested tempo clears the prespecified .1 anticipation threshold for both signs
and the 5% quiet-alarm ceiling. The training stream itself has pre-update
ON/OFF anticipation .513/.656. The model is learning during the live neural
stream, not only fitting a frozen readout: trained weights change 203, 261,
305, and 360 spike entries in the four dwell evaluations and 311/289 in the
two continuous streams.

The demanding steady dwell-6 segment of the standard continuous stream now
has ON/OFF anticipation .105/.333, compared with the shared rule's
**-.291/.057**. The omitted stream has .099/.417 in the analogous segment.
This fixes the wrong-sign ON response and substantially strengthens OFF, but
the ON margin at slow tempo is narrow. The first two returning events after
the omission are still missed: the local timing gate is closed as its sensory
reference reacquires the event cycle. Whole-stream averages should not hide
this remaining recovery limitation.

## Local update and stability audit

At each issue, the rule stores only its local state context, gated recurrent
forecast, and existing anatomical-edge eligibility. At confirmation, the
event/quiet error updates the component chosen at issue. Closed-gate
surprises update the local timing reference and counts, with zero amplitude
credit. The target's nonnegative components stay within [0,10]; all
non-target magnitudes remain bit-for-bit fixed. Replaying every recorded
per-frame delta reconstructs both component histories to within 4.7e-7.
Neural states are finite, with no trial-boundary reset.

The first two incoming edges illustrate the learned separation. For
negative sensory context their magnitudes become **10.000/0.000**; for
positive context they become **.554/2.567**. They remain the same measured
edges with their original signs. One component reaches the upper bound, so
this run does not establish that its asymptotic optimum lies inside [0,10].
The selected 2,453-frame training pass took 51.78 seconds, or 47.37 camera
frames/s at eight neural ticks/frame. Evaluation ran 62-70 frames/s; training
and evaluation are near real time on this small motif, not a full-connectome
throughput result.

## Consequence for Milestone 1

This is the first live coupled controlled test in which the measured circuit's
local plasticity meets the prespecified ON, OFF, quiet, and MSE criteria at all
tested tempos, including an unseen tempo. It supports context-dependent
efficacy as an effective *local* mechanism for this simple repeated motion.
It does not prove general visual prediction, robust omission recovery, or M1A:
the model has not yet trained on open-loop event-camera Pong sequences or
passed the original visual-subnetwork acceptance checks. Promoting this
experimental single-target rule to that larger stage needs an explicit
integration design, especially for visual neurons whose local observation is
feedforward current rather than directly injected sensory current.

Evidence: [parity](2026-09-22-context-efficacy-parity-results.json),
[training/selection](2026-09-22-context-efficacy-training-results.json), and
[all held-out results and update audit](2026-09-22-context-efficacy-evaluation-results.json).
Full per-frame traces remain in `runs/live-context-efficacy-v1/`.

```powershell
.venv/Scripts/python scripts/live_context_efficacy.py parity
.venv/Scripts/python scripts/live_context_efficacy.py train
.venv/Scripts/python scripts/live_context_evaluate.py
```

Saved run files are protected against overwrite. A fresh reproduction should
use an empty run directory while retaining the pinned source artifacts.
