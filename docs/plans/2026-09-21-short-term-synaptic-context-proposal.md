# Short-term synaptic context: bounded next experiment

Status: user approved; implementation underway. The fixed adaptation A/B family
is now complete and rejected; see
[findings](../experiments/2026-09-21-adaptation-context-findings.md).
No production architecture changes.

## Question and scope

Can recent activity at existing predictive synapses make their transmission
sufficiently history-dependent for the same local rule to learn both event
polarities at multiple tempos, including quiet-frame suppression?

The user ranks short-term facilitation/depression after reusing adaptation and
before an explicit interval state. This proposal tests that second direction.
It does not assume that new state is necessary, that STP will work, or that the
particular parameters below describe the fly cells in this crop.

Alternatives are another adaptation coupling (cheaper but extends a rejected
family), and an explicit interval trace (more directly encodes elapsed time but
requires deciding which local events define an interval). Recommend a small STP
screen first, without combining it with adaptation-dependent forecast expression.

## Two fixed transmission variants

Start from the uncoupled area-matched 20 ms excitatory / 5 ms inhibitory model.
Keep its impulse normalization, initial weights, fixed signs, measured edges,
delays, neuron dynamics, eight ticks/frame and +8-tick forecast horizon.

Only existing predictive edges receive private per-environment release state.
Apply the same law to both transmitter signs; do not select cells or pathways
from their task performance. Two arms isolate depression and facilitation:

**D: depression only.** Each edge has available fraction r, initially 1.
For elapsed time dt since its last arrival, immediately before the next arrival:

    r_minus = 1 - (1-r_previous) * exp(-dt / 0.100 s)
    release_gain = r_minus
    r_next = (1-0.5) * r_minus

**F: facilitation only.** Each edge has facilitation u, initially 0.

    u_minus = u_previous * exp(-dt / 0.100 s)
    u_next = u_minus + 0.5 * (1-u_minus)
    release_gain = u_next / 0.5

D gains lie in [0,1]; F gains lie in [1,2]. A first impulse from fully recovered
state has gain 1 in both arms and exactly matches the baseline waveform and
integrated area. Repeated impulses deliberately change total drive; this is not
an area-conserved spike-train comparison. The 0.5 utilization and 100 ms recovery
are fixed phenomenological test values, not fitted to task labels or claimed as
fly measurements. There is no parameter sweep or joint D+F arm in this proposal.

These are simplified deterministic resource/facilitation limits, not a full
vesicle model. One aggregate state belongs to each existing computational edge;
no contact-level release sites or additional edges are introduced.

At arrival, multiply that edge's existing signed physical impulse by the saved
release_gain. Use the same realized impulse in the physical predictive current,
its signed E/I decomposition, and the arriving local causal eligibility. Gains
must be computed once per arrival, not advanced again by a diagnostic or learning
query. The forecast remains the physical E+I prediction; no additional gate.

Unlike the adaptation-expression experiment, this changes physical neural
activity even at frozen weights. This is a biological-model experiment, not an
exact-behavior software optimization.

## Local learning and efficient state updates

The existing signed eligibility receives release-scaled arrivals and decays with
the existing sign-specific synaptic waveform. FramePrediction captures it at
issue time, then applies the same local error-times-eligibility update when the
target arrives eight ticks later. Do not multiply the entire old trace by the
latest release gain: each contribution retains the gain of its own arrival.

No derivative through release state, adaptation, membrane or previous spikes is
taken. Long-term magnitudes alone are learned; release parameters are fixed.
Behavioral R-STDP and nonpredictive transmission remain unchanged. Existing
homeostasis and weight bounds remain active.

Store state plus a last-arrival timestamp only for predictive edges and each
environment. Recover analytically when an arrival occurs; do not scan inactive
edges to decay state every tick. Timestamps measure local elapsed time, not camera
phase or a supplied motion period. Use vectorized/compiled arrival operations,
not Python loops over synapses. State is private per environment and continues
through warmup, quiet periods and trial boundaries; no stimulus-triggered reset.

Expected additional storage with float32 state and int64 timestamps is 12 bytes
per predictive edge per environment, plus indexing and transient arrival buffers.
Measure actual storage/runtime rather than extrapolate motif throughput to the
full connectome. Native backends and production checkpoints are outside this
experimental implementation until explicitly supported and tested.

## Preflight before spending training compute

1. Analytic single-impulse and paired-pulse checks at 10/30/100/300 ms verify
   gain bounds, recovery, signs and preserved first-impulse area. A burst verifies
   bounded saturation. These intervals are diagnostic inputs, not fitted targets.
2. A unit-release control reproduces baseline physical spikes, currents,
   predictions, eligibility and updates. Reconstruct signed currents and local
   eligibility independently from recorded release impulses and their ages.
3. Verify repeated queries do not consume resources, environment state cannot
   leak, and silent elapsed-time recovery agrees with an explicit reference.
4. On five matched trials at each dwell 2/3/4/6 (blank seed 9080), record realized
   gains and state during events/quiet, firing, finite/bounded state, and runtime.
   This checks whether autonomous firing leaves the release state effectively
   saturated; report it before interpreting any later outcome as useful memory.
5. Assert the existing observation targets equal the uncoupled baseline on these
   sequences. Changed physical dynamics may change internal targets: if this
   assertion fails, stop and report before training. Do not silently redefine
   the target or score against an easier, model-dependent observation stream.

## Fixed learning and selection

If preflight passes, train each arm for 200 mixed trials using the exact saved
seed 9060 schedule at dwells 2/4/6, from identical original magnitudes. Use the
same uncoupled mixed baseline; do not carry adaptation A/B weights into STP.

Freeze weights and evaluate initial/trained D, F and uncoupled baseline with
15 trials per dwell 2/3/4/6, blank seed 9081. Tempo 3 remains unseen in training.
Use the existing first-cycle exclusion, +8-tick targets and all remaining quiet
frames. Retain zero and persistence controls, and assert matching observation
targets and unchanged evaluation weights across every condition.

Require the unchanged controlled-task criteria at every tempo (including both
anticipations >=0.1 and quiet false alarms <=5%), plus held-out MSE better than
the trained uncoupled model. Save both failures and successes. Stop after these
two arms if neither qualifies; no longer training, fitted gain thresholds or
additional time constants under this approval.

Confirm only qualifiers on 30 fresh trials per tempo, seed 9082, without tuning.
For a qualifier also evaluate frozen trained magnitudes with release forced to 1
through warmup/evaluation. This ablation tests dependence on dynamic release,
but cannot by itself distinguish timing information from changed average drive.
Do not claim that distinction without a separately designed matched-drive control.

A controlled-task pass remains preliminary: it must transfer to broader visual
stimuli and then actual M1A Pong prediction before M1B. M1A is not replaced by
this screen. No learned artificial readout, global classifier, backpropagation,
new connections, sign changes or simulator labels are permitted.

## Biological basis and limitations

[Tsodyks and Markram (1997)](https://pubmed.ncbi.nlm.nih.gov/9012851/) combine
neocortical recordings and modeling to connect synaptic depression with temporal
signaling. [Buonomano (2000)](https://pubmed.ncbi.nlm.nih.gov/10648718/) models
temporal selectivity arising from short-term plasticity and changing E/I balance.
These motivate the mechanism family; neither validates these parameters, this
uniform edge assignment, this fly circuit, or our local learning rule.

The main risk is that STP merely attenuates/amplifies a tonically driven circuit,
without exposing a usable future-event distinction. The preflight gain audit,
mixed-tempo evaluation and quiet-error criteria make that failure visible. A
negative result rejects these two fixed settings, not all short-term plasticity.

## Approval boundary

Approval covers the two formulas, physical transmission change,
matching arrival eligibility, and bounded preflight/training/evaluation above.
STP remains an opt-in controlled experiment, separate from production and the
completed adaptation-context work.
