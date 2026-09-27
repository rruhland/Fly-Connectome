# Decisive 2x2 successor: local generative dynamics and observation uncertainty

Close stored-displacement prototype variants: broader experience improves their
scores but still fails the fresh per-family/aggregate gates. Preserve the learned
joint contextual representation. Test two independent architectural changes in a
2x2 comparison, with no tuning loops:

1. **Local generative dynamics:** replace each competitive prototype's stored
   displacement with a small regularized linear temporal map. Learn local lag/lag
   and lag/endpoint correlations from actual observations; solve four temporal
   coefficients shared across image axes. Keep 128 prototypes, context admission
   and 4-neighbor interpolation, with unit ridge prior. This can generalize within
   a local regime rather than replay its mean endpoint. No backpropagation.
2. **Statistical observation uncertainty:** use a Gaussian multiple-comparison
   noise bound for image regions (per-image upper bound .01 under that noise
   model), rather than amplifying noise then requiring an event every sample.
   Maintain image-based association with motion variance learned from consecutive
   observed velocity changes (rate .25). Use its likelihood and the existing miss
   branch instead of a second hard distance cutoff. Events remain sensory inputs;
   no named-object or trajectory-family cue is added.

Train each arm on the same 192 observed streams. Reserve phases 17/19, headings
15/75/135 degrees and scales .9/1.4 for comparison. Keep all prior forecast gates,
plus at least 95% observed-history coverage, blank-camera false-track checks,
crossing/outage identity and joint-context regressions for any retained arm.
The statistical model is explicitly not a guarantee for non-Gaussian real-camera
noise. If neither architecture improves the combined gates, reassess at family
level; do not sweep regularization, prototype count or noise thresholds.
