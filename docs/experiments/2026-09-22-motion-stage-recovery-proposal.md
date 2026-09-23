# Bounded recovery of the measured motion pathway before more M1A learning

Proposed; **not yet an adopted model change**. Motivated by the [frozen motion-stage audit](2026-09-22-motion-stage-audit-findings.md). This is the next architecture experiment for user review, not a new milestone acceptance gate.

## Question and scope

Can the existing measured optic-lobe graph produce position-stable T4/T5 motion responses when its anatomically present but silent input classes have a plausible operating point? The immediate alternative is that the point-neuron/sensor representation omits the temporal or spatial integration needed for motion. Establish this before doing another full-Pong plasticity run.

Retain the exact graph and edge signs, delays, area-matched synapses, 8 ticks/frame, local/no-backprop learning rule, and original checkpoint. Do not add edges, train a readout, share learned weights across positions, or tune parameters on Pong reward. Record each trial as an opt-in dynamics profile; the production baseline remains available byte-for-byte. Evaluate with plasticity disabled.

## Experiment A: minimum cell-class operating point

Use the current `NeuronConfig` rest-current mechanism, changing only Mi4, Tm4, and Tm9 from their current class-default zero. Calibrate one class at a time at rest currents `0, 0.85, 0.95, 1.05, 1.15`; reuse the pinned initial checkpoint and warm/reset protocol. On a local ON and OFF bar plus matched blanks, record class spike count, number of changed neurons, blank rate, signed afferent arrivals at T4/T5, and finite/peak network activity. Choose the **smallest** value for which at least five local neurons differ from their matched blank, at least one measured T4/T5 target receives an arrival from that class, and blank firing remains below 0.01 spikes/neuron/tick; if none exists, mark the class as not recoverable by rest current. Combine only the selected class values, then run the preregistered dot/bar, both directions/polarities, at x=18 and x=46. Include stationary flashes and a static bar to check that any response is motion dependent rather than merely luminance or autonomous spiking. Hold out at least one additional location and one speed for the final frozen check; do not choose parameters on these holdouts.

Advance beyond this stage only if the missing pathways now deliver spikes to their measured T4/T5 targets, T4 ON and T5 OFF responses exceed matched blanks without excessive autonomous firing, and a per-cell or per-subtype direction preference has the **same sign at both calibration locations and at the held-out location**. As a conservative screen, require at least ten evoked spikes across the opposed-direction pair and a direction index `|right-left|/(right+left) >= 0.2` for a responding subtype at each location, plus a lower response to a static bar. This screen is not proof of biological tuning; repeat across cells, speeds, and polarity before a learning change. Report raw counts, denominators, blank rates, and per-cell distributions; avoid a bare accuracy score that can hide silence. A rest-current setting that merely makes all cells fire is a failure.

If one class cannot respond appropriately without high blank firing, stop the rest-current approach. The next proposal would address graded transmission or class-specific temporal filters, grounded in the measured input physiology, rather than another unbounded gain sweep. The two-location audit's tiny, sign-flipping T4 voltage contrasts mean threshold tuning alone is **not** assumed to succeed.

## Experiment B: pathway audit, independent of a successful setting

Use single-location ON/OFF flashes and ordered neighboring-column events to trace *signed* sensory input, L1/L2/L3 spikes, Mi1/Tm3/Mi9 and Tm1/Tm2/Tm4/Tm9 activity, and T4/T5 currents tick by tick. Compare the same stimuli at the two existing positions plus the holdout. This is read-only analysis with the existing sensor map; no label or simulator direction enters the network. It tests whether the shared `OFF − ON` sensory injection plus fixed anatomical signs produces the expected temporal ordering and cross-column asymmetry. If it does not, write a separate, physiology-grounded sensor/transduction proposal; do not silently reroute L1/L2/L3 or change transmitter signs.

## Learning resumes only after a motion-bearing stage exists

If frozen motion responses pass the checks, separately propose and test local credit on existing predictive edges into motion-related stages, using their *locally observed feedforward arrivals* or activity as the future target. Keep causal alignment and per-synapse local eligibility; no direction labels, backprop, or globally shared trainable weights. First require online improvement on a small repeated-motion task, including unseen positions, against its own frozen initial checkpoint and persistence. Then return to M1A full Pong and require anticipation improvement without wrong-sign OFF responses or high quiet-frame false alarms. A negative recovery experiment is still decisive: it prevents more full-Pong learning sweeps on a motion stage that is not functioning.

This proposal changes only opt-in neuron class rest currents in Experiment A. Any later change to graded release, transduction, dendrites, or predictive targets requires a separately reviewable rationale and experiment.
