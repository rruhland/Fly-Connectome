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
