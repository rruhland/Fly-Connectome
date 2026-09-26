# One causal polarity-return observer experiment

The surface state's noisy failure disappears with exact current visible-image
anchors (16/16 familiar and heavy-noise context top-32), while sparse anchors
remain below gate. This supports prioritizing sensory reconstruction in this
benchmark rather than changing local contextual credit again.

Test one different online teaching principle: an arriving event is supported
when an opposite-polarity event is observed at the same pixel within eight
subsequent camera samples. Keep an eight-sample local credit buffer; apply credit
only once that interval has elapsed. No intensity targets or simulator labels
enter teaching. Features are captured at event arrival; credit uses the existing
local observer update, with no backpropagation. Discard incomplete final buffers.

This is a hypothesis about recurring moving contrast, not physical ground truth:
dropouts and changes that persist longer than the window become false-negative
teaching examples. Static cues, slow motion, and sustained illumination changes
are explicit risks. Do not sweep window lengths after seeing test results.

Use the existing 64 generic corrupted training streams, then freeze this observer
and train the unchanged surface/context state on the original training split.
Evaluate aligned, shuffled-context and frozen-context states on all existing
clean/noisy/heavy-noisy/regression/disappearance gates, plus observer event
credibility when available. Preserve the original teacher/control results.
Pass requires the same full bounded gate, not merely better rejection of noise.
Fail ends this teacher hypothesis without narrow variants.
