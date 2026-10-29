# Learned delta model and primitive feedback pilot

Implementation: `pong_learned_delta_planner.py` and its fixed fit runner. The
machine-readable spec pins the approved ridge-plus-eight-tree delta residual,
16-tree competing-event regression, 117/147 input dimensions, physical discount
and H8/W8 primitive beam. Execute one command, observe, then replan. Frozen C1
option3 is an approximate own-reward Q tail, not a guaranteed baseline value.

Use the staged 452 unique camera traces, 114,229 actual transitions and 206 TRAIN
starts. Preserve mask changes; no identity correspondence is assumed between
sorted scene slots. Only public planned intervals enter predictors. Actual
clipped durations enter discounted event targets. Unknown short-censored intervals
are excluded, and actual hit/end mass competes with survival. No class balancing,
geometry cost, object roles, expert labels or simulator lookahead is used.

One 180-second fit allocation includes exports, compiled-tree parity,
serialization, and 32 saved post-hit inference probes across starts. Empty or
overflow inputs, invalid predicted scenes and unsupported C1 tail references
retain the declared baseline fallback. First-return opening actions stay baseline.
Native validation belongs to the separate worker after parent source/spec review.

Focused tests cover public-phase alignment, past-action history, competing
discounted event mass, tail-based pruning at every depth, serialization and
fingerprint coverage of the actual compiled model/head/tail parameters.

Fit completed in 39.131 seconds, including exports and 32 saved post-hit probes.
Delta trees took 8.719 seconds and event trees 27.859. The world used 112,655
transitions; the event model used 113,479, retaining all 481 hits and 328 natural
ends. Public timing matched exactly. Compiled predictions matched sklearn on
2,048 inputs per head; all 32 serialized beam decisions matched exactly. Inference
averaged 9.145 ms, maximum 18.501 ms, with 156 expanded nodes in every probe.
Artifacts occupy 1.47 MB. Source commit: `18f09cb`; spec SHA:
`f517411e53aad34a549501f0a0dbe0c6ad1e3ddad0395ee05e2f3f0db18c1017`.
Model SHA: `171b9acabd3c4abc4b446b9f8c83a373bd36816aedfc8f96cebda80a075a189d`.

The separate actual eight-game comparison rejected this frozen model: baseline
23 returns versus 11, first returns 7 unchanged, later returns 16 versus 4.
There were 2,958 new cameras and 1,346 changed actions; no 32-game or fresh stage.
Baseline 40069 remains retained. No model, horizon or support variants followed.

One 0.58-second saved-trace check used the first declared game, seed 2915871533,
340 cameras and one hit. Stable-mask one-step sorted-slot position error was
0.40 pixels RMS; across mask changes it was 1.40. Eight-step rollouts following
the actual recorded commands perturbed frozen-tail values by median 0.0247 and
p90 0.1475, compared with selected root gains of median 0.0175 and p90 0.0675.
The event head assigned 0.879 to the actual final end. This supports model–tail
composition error over a blanket failure to learn motion or end events. It does
not identify counterfactual action rankings or isolate critic error from future
policy mismatch. The one-trace report is saved beside the native results as
`one-trace-teacher-forced-review.json`; no fit or new cameras were used.
