# Live local timing learner: fixed test protocol

Approved mechanism: [live-rule proposal](2026-09-22-live-local-timing-proposal.md).
The exact eta0 parity control passed before learning. This protocol fixes
the subsequent schedules and selection before held-out tests are run.

Use the 250-neuron measured motif, eight ticks per camera frame, area-matched
20 ms excitatory / 5 ms inhibitory predictive kinetics, and the same fixed
connectome. Start all non-target weights from the saved mixed-training
checkpoint and reset the 12 incoming predictive edges of L3 body82450 to
their original anatomical magnitudes. Keep all non-target weights frozen,
homeostasis off and reward learning off. For this one target, use the approved
past-event timing comparator, shared edge magnitudes, causal local event-count
balance and existing +8-tick eligibility update. No trial-boundary state reset.

Train one pass on 42 motion trials (14 each at dwell2/4/6), seed9170 for
shuffled dwells and blank lengths12..36. Compare eta_prediction=.1 and1.0,
starting each run from identical weights and normal warmup. Validate each
final weight snapshot with eta0 on nine separate trials (three each known
tempo), seed9171. Select eta by: both ON/OFF anticipation >=.1 and quiet
alarms <=5%, then lower MSE; otherwise, among quiet-compliant candidates,
maximize the weaker anticipation and then lower MSE; if none are
quiet-compliant, minimize quiet alarms before anticipation. No test or unseen3
results influence this choice. The training stream's forecasts are scored
prequentially: each issue before its own confirmation update.

After selection, evaluate learned and initial target weights on matched
fresh sequences: eight trials at each dwell2/3/4/6, varied blanks from a
single seed9172, and both standard and omitted 411-frame continuous tempo
challenges. Each evaluation is an independent fresh warmup, then one
uninterrupted stream across its trials/tempo switches. Do not learn during
evaluation. Score only the existing recurrent boundaries in repeated-trial
tests, aligned issue to next-frame target. On the continuous challenges score
all frame issues with next-frame targets and report adaptation, steady state,
and omission recovery separately.

Report ON/OFF anticipation, quiet alarm fraction, total/quiet MSE, zero and
persistence references, initial-versus-learned change, firing rate and state
maxima, bounds/sign invariants, source hashes, update/eligibility audit,
frames per second and elapsed time. A motif pass requires held-out error below
both initial and persistence, ON/OFF anticipation >=.1, quiet alarms <=5%,
and finite/stable sparse firing. Do not equate a motif pass with M1A Pong.

The rate grid is deliberately small because frozen-history replay already
showed these two useful scales. If neither trains stably, inspect the
issue/confirmation credit chain before extending the grid. Do not tune on
unseen3 or omission data.
