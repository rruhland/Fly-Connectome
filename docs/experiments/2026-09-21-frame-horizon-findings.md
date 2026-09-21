# Matching supervision to the forecast horizon improves ON prediction

The user-requested eight-tick supervision experiment produces substantially
stronger learned ON anticipation than the original one-tick objective. It does
not solve OFF prediction or quiet false alarms. This identifies a consequential
objective mismatch, but does not establish a complete predictive learner or pass
M1A.

## Matched comparison

Both runs train 200 trials from identical initial weights on identical camera
inputs, with the same learning rate. The new schedule issues one visual forecast
per camera frame and confirms it eight neural ticks later using retained local
issue-time eligibility. All eight neural simulation ticks remain. The old
schedule supervises every tick against the next tick. Consequently the number
of supervised updates changes deliberately; this comparison does not isolate
horizon from update frequency. No learning-rate retuning was performed.

The induced measured graph, signs, neuron dynamics, biological state updates,
input timing, synchronization, homeostasis and behavioral R-STDP are unchanged.
There is no backpropagation, added connection, external error network or trained
readout. The experiment changes all visual edges' supervision schedule, not only
the reported target. See the [protocol](2026-09-21-frame-horizon-protocol.md).

Same 50 frozen evaluation trajectories, 2,153 scored frames, 150 events of each
polarity, and original thresholds:

| Metric | One-tick supervision | Eight-tick frame supervision |
|---|---:|---:|
| Overall MSE | 0.134747 | 0.121254 |
| ON-event MSE | 0.928360 | 0.543302 |
| OFF-event MSE | 1.002372 | 1.017638 |
| Signed ON anticipation | 0.036593 | 0.270484 |
| Signed OFF anticipation | -0.001185 | -0.008774 |
| Quiet MSE | 0.000271 | 0.014527 |
| Quiet false-alarm fraction | 0% | 16.30% |

Overall error is 10.01% lower than the one-tick learner; ON error is 41.48% lower.
ON anticipation exceeds the original 0.1 threshold. However OFF remains wrong-sign,
quiet false alarms exceed 5%, and overall improvement relative to frozen initial
and zero predictions remains below the required 20%. The full fixed gate fails.
All-frame MSE, including the uncued first cycle, also improves (0.159028 to
0.147173); cue exclusion is not responsible for the direction of the result.

## Exact frozen precursor-response checks

Every array in all 20 initial paired precursor/blank files, and the full frozen
initial evaluation, matches the original run exactly. Training targets, initial
weights and blank intervals also match exactly. Thus the stimulus and initial
response were not changed to obtain the gain.

After training, rerun the identical 10 paired precursor/blank probes separately
with each learned weight set, fresh warmed neural state, and frozen weights.
Score the same 30 recurring events of each polarity, at eight-tick lead:

| Mean signed quantity | One-tick learner | Frame learner |
|---|---:|---:|
| ON forecast | 0.038497 | 0.284456 |
| ON stimulus-minus-blank effect | 0.037152 | 0.277513 |
| OFF forecast | -0.000930 | -0.006858 |
| OFF stimulus-minus-blank effect | 0.002288 | 0.028434 |

The stronger ON forecast is stimulus-dependent, not merely increased background
current. OFF stimuli move prediction in the appropriate direction relative to
blank, but insufficiently to overcome the negative baseline: absolute OFF
forecasts are still wrong-sign. This distinction prevents counting relative
suppression of a wrong-sign prediction as successful anticipation.

The inhibitory L1 target-input magnitude rises from 0.045 to 0.342841; the
excitatory L2 magnitude falls from 0.025 to 0.007965. These observations are
consistent with stronger predominantly negative prediction, including unwanted
negative predictions on quiet frames. This experiment does not establish that
these two changes alone cause the entire improvement.

## Decision

Keep the matched-horizon schedule as an experimental reference. Do not claim
that more trials or timing alignment alone will solve the task. This actual
coupled-network result is consistent with the prior conditional fixed-spike
analysis: stronger ON anticipation, persistent OFF failure, and quiet errors.

The next architecture experiment should target temporal discrimination between
the existing excitatory and inhibitory prediction pathways, with both event
polarities and quiet periods required to pass together. One candidate for a
bounded proposal is distinct temporal filtering on those existing pathways,
tested first on frozen recorded source activity and then in the coupled model.
That would change synaptic dynamics and requires user approval; it has not been
implemented here. This single-seed result is not proof that such a change is
sufficient, or that the current architecture can never learn the task.

## Verification and artifacts

Run completed in 85.53 s; the additional learned paired probes took 10.74 s.
All checked states are finite, weights remain bounded, and frozen evaluation
weights and the source checkpoint are unchanged. Five added tests cover timing,
issue-time credit, quiet supervision, unchanged live traces/behavioral learning,
and diagnostic update reconstruction. Full suite: 215 passed, 4 CUDA skips.
Independent review found no learning-rule blocker and clarified the saved trace
index convention documented in the protocol.

[Full results](2026-09-21-frame-horizon-results.json),
[learned precursor comparison](2026-09-21-frame-horizon-precursor-comparison.json),
and [local artifact hashes](2026-09-21-frame-horizon-artifacts.json).
Raw arrays are in `runs/temporal-frame-horizon-v1`.
