# Directly driven temporal visual test

User authorized the next recommended investigations. Keep all biological-model
and local-learning settings unchanged. This is one bounded end-to-end temporal
test, conditional on a frozen precursor-trace preflight.

## Anatomy correction and fixed choice

All137 measured L2->L1 edges are excitatory, so that route alone cannot supply
negative ON forecasts. The previous recommendation did not check this sign.
Inventory all predictive edges between injected L1/L2/L3 neurons before any
simulation. Among targets with rendered pixels, exactly one has both inhibitory
and excitatory directly injected same-column sources: L3 82450, with L1 26550
(-,9contacts) and L2 20655 (+,5contacts). Fix this unique target without consulting
learning results. Preserve all its predictive sources, all direct parents of
these sources, and the sensory cells at the two stimulus positions. Keep every
measured edge induced by that roster and original parameters; no normalization.

## Stimulus and scoring

A single bright pixel alternates between row30,x39(target) and row30,x38 for
three camera frames each:25ms dwell,50ms full cycle. Four cycles per trial,
then an explicit blank image. Random12-36-frame blanks separate trials. Camera
and neuron state persist across training trials. This is a deliberately simple
temporal task, not a claim of learned spatial motion or M1A success.

Prediction lead remains8ticks/8.33ms. The first cycle supplies the cue; its six
frames are excluded from primary evaluation because onset is unpredictable.
Keep all quiet frames outside that cue, including blank intervals, in scoring.
Report all-frame scores separately so the exclusion is visible. Require the same
20% pooled-error improvement over frozen/zero/persistence,10% event improvement
for each polarity over frozen/zero,0.1 signed anticipation for each polarity,
and <=5% quiet false alarms(|p|>=0.1). No thresholds are tuned from outcomes.

## Frozen preflight and bounded training

Ten full/blank paired trials(seed9021), freshly copied normal warmup state. Blank
controls show no dot at all. For each recurring ON/OFF event, require a compatible
signed input trace magnitude>0.001 and an absolute trace difference from blank
>0.001, on the SAME edge at issue time. Require80% of30events per polarity.
Trace suppression as well as enhancement counts: learning can reweight competing
signals. Do not require the frozen total prediction already to have the right
sign. This gate demonstrates available stimulus-dependent traces, not enough
information to solve the task or a mathematical necessity for all learning.
Reconstruct signed traces from actual spikes and delays; actual frozen forecast
must agree with their weighted sum within1e-6 at scored issuing ticks.

If this gate passes, run one unchanged200-trial training arm(seed9022) and50-trial
fresh-state frozen/trained evaluation(seed9023). Use the existing CPU reference,
camera, dynamics, local update and homeostasis. No tuning, extra candidates or
full-Pong experiments. Bound all neural execution to600seconds. Record phase-
resolved predictions, eligibility, updates and spikes, plus checkpoint identity.
On failure, inspect these saved traces and stop before architecture changes.

## User-requested exposure extension

After inspecting the200-trial result, the user requested a1,000-step test.
Interpreted explicitly in conversation as1,000total trials, using the same unit
as the200-trial result. Keep all parameters, seeds, stimulus and evaluation
unchanged. Reproduce the original first200trials exactly before continuing to
1,000; this avoids an approximate restart from weights without neural state.
Budget900seconds for the extended run and evaluation. No new architecture,
learning-rate search or acceptance changes. `--training-trials 1000` supports
reproduction with the same fixed preflight; the original200-trial default remains.
