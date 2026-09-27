# Probabilistic candidate integration gates

Before additional integration runs: replay the saved checkpoint on all 216
calibrated-transfer scenes and require agreement with offline scores within
1e-5, frozen parameter equality and at least 50 camera samples/second measured
within `step` on this host. Report full measured-connectome execution separately.

Run the saved checkpoint frozen on two-entity crossing scenes: horizontal,
vertical and diagonal, offsets -3/1/5, both polarities. Test grayscale plus
image noise, and the same scene with five unavailable camera samples. Require
at least 35/36 identities retained per condition, no final extra entities, and
at least 95% of unavailable-sample target positions represented by the support
field. Evaluate cached forecast log scores and intervals against hidden scoring
truth during gaps; this score is separate from visible-only calibration and
does not retrospectively teach the learner. Report available forecast counts
and missing forecasts, not just conditional accuracy. No threshold tuning.

Run observationally identical continued/disappeared prefixes with the candidate's
unfitted visual-reidentification prior explicitly exposed as zero calibration
samples. Require identical pre-split mixtures and probabilities, then removal
of contradicted support after visible evidence returns. This verifies honest
ambiguity and revision, not learned conditional existence or domain-general
probability calibration. Retain the separate earlier learned balanced-outcome
fixture as a component capability, not as fitted evidence for this checkpoint.
