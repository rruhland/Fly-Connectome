# Adaptation-dependent local prediction: bounded architecture experiment

Status: approved experiment completed; both fixed candidates rejected. See
[findings](../experiments/2026-09-21-adaptation-context-findings.md). No tuning or
confirmation runs were added after neither candidate qualified.

## Purpose and evidence

Test whether existing postsynaptic adaptation can supply useful context to the
local predictor and its plasticity. The cross-tempo audit found different
adaptation values in275 pairs with similar E/I and different futures, but neither
that pairwise difference nor the unsuccessful nearest-neighbor probes establishes
a general predictive mapping. Mixed-tempo learning weakened the OFF pathway;
single-tempo adaptation partly restored it while increasing quiet errors.

User priority: reuse adaptation first, short-term synaptic plasticity second,
explicit interval state third. Architecture adjustments must be reviewed first.
This proposal covers only the first step. It does not authorize or specify a
fallback architecture if the first step fails.

## One small family, two predeclared orientations

At forecast issue time let:

- E be the existing nonnegative excitatory prediction current.
- I be the existing signed, nonpositive inhibitory prediction current.
- a be the target neuron's existing nonnegative adaptation state.
- theta be its existing positive membrane threshold, in the same voltage units.

Define u=a/(theta+a). No fitted threshold, tempo statistic or new trace is used.

Candidate A, stronger inhibitory expression at higher adaptation:

    g_E = 2*(1-u)
    g_I = 2*u
    P = g_E*E + g_I*I

Candidate B reverses the two gains:

    g_E = 2*u
    g_I = 2*(1-u)
    P = g_E*E + g_I*I

Both gains lie in[0,2] and sum to2. At a=theta they both equal1 and recover
the original predictor exactly. Each pathway retains its anatomical sign,
although the sign of their net sum can change, as expected when relative
expression changes. The sum-to-two constraint does not conserve total current
or synaptic area: actual current depends on E,I and state-dependent gains.
The factor2 makes the symmetric point equal to baseline; it is not fitted.

These are explicit modeling hypotheses, not a claim that a fly L3 cell uses
this exact divisive formula. Testing both directions avoids choosing one from
the same data used to evaluate it. No further slope, offset or gain search is
included. This family can fail even if another use of adaptation could work.

## Physical state and locality

Keep the existing area-matched20ms excitatory/5ms inhibitory physical currents,
membrane integration and adaptation update unchanged. P replaces only the
visual forecast expression. It is computed from local state AFTER the issue
tick's neural update and BEFORE the future target is available. Do not feed P
back into membrane dynamics as a substitute for the physical current sum.

Frozen weights must therefore produce identical physical currents, membrane,
adaptation, arrivals and spikes under baseline and candidates. Once learning
is enabled, changed weights may naturally change subsequent neural activity.

This introduces an explicit distinction between physical predictive input and
the local forecast expression. Such a distinction already exists in prior
experimental forecast variants, but it remains an architecture change requiring
approval. No extra neuron, anatomical edge, predictive weight, memory variable,
external phase clock or global error network is added. No tempo/trial labels
or hand-fitted event-region thresholds are available to the model.

## Matching local learning

At each existing frame boundary, capture the forecast P and the existing signed
visual eligibility e_ij multiplied by the target-local gain appropriate to that
edge's transmitter sign:

    e_context_ij(t) = g_sign(a_j(t)) * e_ij(t)
    delta_w_ij = eta * (y_j(t+8) - encoded_P_j(t)) * e_context_ij(t)

Retain the existing encoding/threshold normalization, weight bounds and update
application. The formula above uses the same conventions as FramePrediction;
it must not add a second normalization or a new derivative of the encoder.
The gain belongs to ISSUE time, not confirmation time. Do not differentiate
through adaptation, membrane, past spikes or future activity. This is the
existing local error-times-eligibility rule applied to the context-modulated
expression, not backpropagation through the network.

The eligibility trace remains the same locally available causal trace; only its
captured forecast contribution is scaled. All eight simulation ticks/frame,
the +8tick forecast horizon, quiet confirmations, behavioral R-STDP, homeostasis
and sensory/behavioral pathway handling remain unchanged.

## Verification before training

1. Gain bounds, sign preservation, both orientations and exact identity at
   a=theta, including zero adaptation and large adaptation.
2. Frozen baseline/candidate physical states and spikes match exactly; only
   forecast expression differs.
3. Captured local eligibility uses the same gain and issue-time adaptation as
   its forecast, even if adaptation changes before confirmation.
4. Reconstructed signed edge contributions sum to the expressed forecast.
5. Existing tests pass; opt-in experimental classes do not change defaults or
   claim unsupported production checkpoint/native-kernel support.

Initial sign changes are reported as diagnostics, not a post-hoc gate for
selecting the candidate most likely to look good. Neither orientation is tuned
or silently dropped because its initial prediction is poor.

## Fixed learning experiment

Train each candidate from the same initial crop for200 mixed trials at dwells
2/4/6, using the exact previous seed9060 schedule. Reuse the saved uncoupled
mixed-training weights as the baseline, with identity-control equivalence
verified. This is two candidate runs, not an open-ended sweep.

For each predictor compare its own untrained weights, trained weights, zero and
persistence. Evaluate baseline and candidates on identical15-trial sequences at
dwells2/3/4/6, seed9073. Dwell3 is absent from neural training. Preserve first-cycle
exclusion, all remaining quiet frames, and separate per-tempo metrics. Include
the uncoupled trained model to detect improvements caused only by changing the
untrained reference predictor.

Primary outcomes are actual neural forecast MSE, ON/OFF errors and anticipation,
correct-sign counts and quiet false alarms. Use unchanged controlled-task
criteria, including both anticipations>=.1 and quiet false alarms<=5%; no offline
classifier counts as a pass. Report runtime and memory overhead on this motif.

A candidate is promising only if it passes at every trained tempo AND held-out3
and improves held-out MSE over the uncoupled trained baseline. If either meets
that bar, confirm qualifying candidates without retuning on30 fresh trials per
tempo, seed9074. Report both candidates, including failures; do not claim a
general vision result from the same two-position trajectory.

If neither qualifies, stop this family after its fixed runs. Do not lengthen
training, fit thresholds or add another state within this experiment. Summarize
whether failure concerns polarity, temporal localization, transfer, or numerical
stability, then prepare a separate short-term synaptic-dynamics proposal.

## Efficiency and scope

No new persistent temporal state is required. Forecast-time arithmetic is a
bounded rational gain and two current multiplications per target; captured
eligibility adds a gain multiply per participating predictive edge. Existing
tick dynamics remain as implemented. Avoid dense per-edge per-tick gain arrays;
apply synaptic scaling only to the sparse captured visual entries. Benchmark
the actual implementation rather than promise a particular overhead.

This is a controlled-motif experiment, not completion of M1A, promotion to the
full connectome/Pong task, or permission to begin M1B. Any successful mechanism
must subsequently pass broader stimulus and M1A checks under the same locality,
topology/sign and no-backprop constraints.
