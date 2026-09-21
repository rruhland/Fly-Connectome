# Bounded proposal: local prediction confirmation windows

Status: approved, then stopped at failed Gate A. Zero fixed pairs passed both
fresh development seeds (two required). No new learning mode or training was
performed. See `../experiments/2026-09-20-confirmation-window-gate-a.md`.

## Purpose and limits

Test whether existing connectome activity can learn useful sensory anticipation
when local credit is resolved over a short future window rather than the next
neural tick. Preserve measured topology, transmitter signs, point-neuron dynamics,
no backpropagation, local updates and behavioral R-STDP exactly as constraints.

The evidence is limited but concrete: three L2-to-L1 physical signed traces pass
the preregistered next-event screen on both development trajectories, while their
one-tick screen fails. Two retain this alignment despite strong weight depression.
See `../experiments/2026-09-20-local-trace-horizon-findings.md`. This is not proof
that a new objective will work, nor that the traces add information beyond local
event history. The proposal therefore contains an initial control gate.

The current one-tick model remains a separate, reproducible baseline. This is an
explicit experimental change to sensory prediction semantics and temporal credit,
not an exact-behavior software optimization or a silent replacement of M1A goals.

## Alternatives considered

1. **Recommended: bounded first-event confirmation.** A prediction issued now is
   confirmed by the first local event within 100 ms, or expires against zero.
   Retain the issuing prediction and its local causal tag until resolution. This
   directly tests the strongest observed trace result and penalizes false alarms.
   Cost: new bounded temporal state and a changed experimental prediction target.
2. **Fixed 50 ms delayed confirmation.** Simpler timing and some supporting trace
   evidence, but still demands the event at one exact future tick and omits the
   strongest next-event finding. Keep as a later alternative, not another arm here.
3. **Lengthen the existing eligibility decay alone.** Small code change, but it
   would still compare an immediate target against a one-tick prediction while
   crediting older causes. It does not cleanly test the observed hypothesis and
   risks repeating the earlier trace/forecast mismatch. Do not pursue it here.

Do not restore C2 input weights or raise tonic currents as part of this test.
The previous C2 restoration recovered activity but worsened prediction error.

## Gate A: information beyond local event history

Before implementing the rule, test the three already selected L2-to-L1 body pairs
50062->51452, 75482->71110 and 42416->44723. Selection is fixed now; do not replace
failures with newly discovered edges.

- Freeze the existing 10,000-frame checkpoint and run 2,000 scripted frames on
  new development seeds 1103/1104. Final seeds 1201-1204 remain reserved.
- Reproduce the frame-end next-event diagnostic and its exact censoring rules.
- Form a strictly causal local-history stratum at each decision: last nonzero
  local polarity and its age in camera frames, capped at 12, with a separate
  never-observed stratum. The decision-frame event is already observed.
- Within each seed, postsynaptic neuron and history stratum, shuffle target
  labels using ten fixed seeds 421-430. This preserves the selected history
  association while disrupting additional trace/target timing. Do not fit an
  artificial predictor or feed these diagnostic labels to the network.
- Require at least two of the fixed three pairs to pass on both new development
  seeds: at least five unique referenced future events, trace power >1e-10,
  correlation >=0.05, positive target-times-trace moment, and observed covariance
  above every history-stratified shuffle. Report stratum sizes and the fraction
  of windows whose labels can actually be permuted; non-permutable strata do not
  establish additional information. Retain the zero/persistence/polarity controls.

This remains an exploratory gate, not a significance test: history is deliberately
coarse and does not exhaust possible confounding. If insufficient event coverage
or non-permutable labels prevent the comparison, call it inconclusive. If the
gate fails or is inconclusive, stop this proposal before changing the model and
report the result; do not lower thresholds, extend the run or search more edges.

## Experimental rule, conditional on Gate A

### Scope

Apply the new mode to **all existing predictive edges terminating on directly
injected L1-L3 sensory neurons**, determined by anatomy/pathway and the fixed
retinal injection mask. The three diagnostic body pairs do not define a learning
mask. Feedforward edges remain fixed. Predictive learning at other neurons keeps
its existing rule. Behavioral R-STDP and homeostasis keep their existing rules.

No changes to graph threshold, signs, delays, rest currents, membrane/current
time constants, sensory gain, event camera, Pong physics, motor control or reward.
The C2 operating-point limitation remains something to measure, not a problem this
proposal claims to solve.

### Prediction issuance and local resolution

Use every neural tick; the rule must not receive a privileged camera-frame or
simulator-event flag. With the current 1/960 s neural step, H=96 ticks is 100 ms.
The experiment fixes H before training and does not tune it on performance.

At the end of tick s, the existing network issues:

```text
p_j(s) = clip(predictive_current_j(s) / baseline_threshold_j, -1, 1)
```

Snapshot the existing signed physical eligibility e_ij(s), after that tick's
arrivals, for each relevant measured synapse. Its ongoing physical trace continues
to decay and receive arrivals normally; the snapshot is a temporary causal tag
for this issued prediction. It neither injects current nor schedules spikes.

For this prediction, the confirmation interval is ticks s+1 through s+H inclusive:

- At the first nonzero actual local camera-derived input y_j(t), resolve against
  that signed value. No object labels, positions, future observations or reward.
- If no event occurs by s+H, resolve against zero.
- An event exactly at the deadline confirms before expiry. An event after expiry
  cannot revive the prediction. An event at s cannot confirm a prediction issued
  later in the same tick.
- Confirm all older unresolved predictions at that neuron whose intervals include
  this first event. Each prediction resolves exactly once; later events cannot
  overwrite its target.

At resolution, propose the local magnitude update:

```text
delta_magnitude_ij = (eta_prediction / H) * (confirmation_j - p_j(s)) * e_ij(s)
```

Use the stored issuing prediction, not the current prediction. Use its issuing
eligibility, not the trace enlarged by the confirming event. Signs stay fixed;
existing magnitude bounds and synchronization still apply. Eligibility can be
nonzero even at a zero-magnitude measured edge, allowing it to strengthen again.

The fixed 1/H factor explicitly bounds the aggregate scale when one event resolves
up to H overlapping cohorts. It changes the effective learning rate and must be
recorded, not described as an algebraically equivalent implementation. Use the
existing eta_prediction=0.01, giving 0.0001041667 per issued tag. No rate search is
included; a negative result does not establish that every rate would fail.

This is local delayed confirmation, with no differentiation through dynamics,
backpropagation, optimizer, artificial learned readout or additional connections.
The temporary stored tags are an explicit modeling approximation, not claimed
physiological measurements.

### Sparse state, resets and checkpoints

Use bounded deadline cohorts and active postsynaptic/edge indices. Issue no stored
edge record when its eligibility is already pruned under the existing threshold;
implicit zero forecasts still count in evaluation. Do retain nonzero eligible
traces when the summed prediction is zero, including cancellation and zero-weight
edges. Do not scan the entire graph or loop over individual edges/spikes in Python.

With 17,557 relevant edges, 96 dense float32 eligibility slices alone would be
about 6.74 MB for B=1, before prediction/index metadata. Use this only as a sanity
bound, not a measured memory claim. Measure actual additional memory and runtime;
do not shorten the window or discard biological activity to make it cheaper.
The bounded experiment is B=1 on CPU. Initially reject the experimental mode on
unsupported backends/batch sizes explicitly; legacy CPU/CUDA/batching remain
unchanged. This is not final backend acceptance or authorization for a new round
of performance optimization.

Pending tags are private local state. Ordinary Pong point resets do not clear
them; reset-induced camera events remain ordinary confirmations. Save their issue
times, deadlines, predictions, eligibilities and unresolved status for exact resume.
Define an explicit new sensory-prediction mode/version and window length; include
them in compatibility checks and experiment identity. Old checkpoints retain their
old behavior. Never reinterpret old one-tick predictions as window-trained outputs.

## Gate B: software and controlled local learning

Before the measured graph, verify:

- Confirmed, wrong-polarity and expired predictions of both signs have the exact
  hand-calculated update and retain transmitter signs.
- Same-tick input gives no retrospective credit; first event wins; deadline ties
  confirm; expired records cannot update twice.
- Later arrivals and weight changes cannot mutate the stored causal tag.
- Cancellation and existing zero-weight edges retain valid local eligibility.
- Blank windows penalize false predictions; always-zero outputs remain in metrics.
- Rewards do not affect these updates; pending sensory state does not alter R-STDP.
- Forward trajectories are identical to the old network when weight updates are
  disabled. No artificial tag is used as neural input.
- Exact resume includes partially resolved windows, including across point resets.
- A deterministic delayed-association fixture learns with the local rule, while
  temporally shuffled input does not satisfy the same test. This checks software,
  not biological or milestone success.

Run relevant tests after each chunk and the complete suite before measured work.
Failure here means fix the implementation, not weaken the checks or start training.

## Gate C: one bounded measured-graph experiment

Start from the unchanged initial measured graph/checkpoint with empty pending
credit, retaining the existing neural warmup. Keep B=1, training seed 1 and the
approved scripted M1A stream. Train one window-mode arm up to 10,000 frames, saving
at least every 100. Inspect at 2,000 for nonfinite state, invalid bounds or persistent
runaway/global silence; checkpoint and stop on those failures. Otherwise continue
to the fixed 10,000-frame endpoint without intermediate parameter changes.

Compare against the same initial network and existing matched one-tick checkpoints
at 2,000 and 10,000 frames, evaluating frozen 2,000-frame sequences on development
seeds 1101/1102. The baseline weights are never window-trained. Score all arms on
the same window target, sampling the fixed frame-end decisions used in diagnostics,
including zero predictions and complete no-event windows. Explicitly distinguish
this evaluation sampling from every-tick online learning.

Report:

- Window-target MSE versus initialization, matched one-tick learner, zero, causal
  persistence, and last/opposite-polarity controls with a 100 ms expiry.
- Event-conditioned supported/unsupported ON/OFF errors and quiet-window false
  alarms. No omission of anatomically unsupported events from the overall score.
- Fixed timing-shuffle controls and the history-controlled diagnostic.
- Original one-tick errors, unchanged in name, to expose any tradeoff rather than
  declaring a newly relabeled metric successful.
- L1-L3, C2/C3 and T4/T5 activity, sampled no-event behavior, weight changes,
  checkpoint identity, simulated time, wall time and actual extra memory.

Evidence to justify further development must include improvement over initialization
and the matched one-tick arm, and performance better than zero and the specified
causal baselines on **each** development seed for the new objective. Event-conditioned
error must also improve; suppressing quiet activity alone is insufficient. Report
all population/polarity regressions and do not claim a timing-shuffle comparison
is statistical significance.

If the one arm does not meet this evidence gate, stop: no automatic learning-rate,
horizon, excitability or topology sweep and no additional training exposure. A
negative or inconclusive result triggers discussion of the modeling objective or
architecture. A positive result justifies a later confirmation plan; it does not
by itself pass M1A, release final seeds, change milestone acceptance criteria or
advance to M1B. A window objective does not establish exact next-tick timing.

## Approval scope

Approval requested for this written rule and its conditional gates: frozen history
controls first; if they pass, implementation, tests and the single bounded learning
experiment above. Routine implementation chunks can proceed without repeated
approval. Any further model revision, new sweep or change to milestone goals
returns to the user. Until approval, preserve the current implementation and all
scientific checkpoints.
