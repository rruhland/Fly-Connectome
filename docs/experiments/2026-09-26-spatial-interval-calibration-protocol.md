# Frozen spatial beliefs with observable endpoint calibration

The original belief experiment passes proper-score/coverage/family gates but
fails calibration: central 90% intervals cover 98.3%/96.6% at horizons 4/8.
Preserve that result. No density, neighbor, memory or motion parameters change.

Fit marginal CDF rank intervals from 64 separate causal camera streams, with
the same generic random angle/speed/polarity distribution. Only same-identity
actually observed endpoints supply probability-integral-transform samples.
Bound calibration at 512 outcomes per horizon. Use finite-sample empirical
5%/95% rank endpoints; this is marginal interval calibration, not a claim of
joint or per-context conditional calibration. All controls receive identical
calibration opportunities. No renderer truth in calibration.

Fresh test: 216 L/T/line scenes, phases 41/43, angles 24/66/144 degrees, scales
.85/1.35, independent event/image noise seeds. Keep the previous NLL gates and
require calibrated marginal 90% coverage .85-.95 at both horizons, plus .95
observation coverage. Report per-family coverage, mean errors, and uncalibrated
coverage too. Do not silently waive point-forecast failure. If accepted, retain
this as a probabilistic candidate for broader scene-memory/integration tests;
it still needs an explicit sensor and acceptance revision in a promotion plan.
