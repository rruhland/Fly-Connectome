# Timing transfer first, margin investigation second

User requests testing whether the E/I regions survive changed motion timing
before doing the margin investigation. No hardcoded gate is added to the model.
These are frozen diagnostics, not locally learned thresholds or general vision.

1. Freeze the existing total/ratio boundary, learned area-matched weights,
   neural dynamics, eight ticks/frame and +8tick forecast horizon. Generate25
   trials each at dwell2,4,6 frames, with matched random blanks seed9026. Positions
   and four cycles/trial remain fixed. Exclude the first complete cycle at each
   dwell; preserve all remaining quiet frames. Report ON/OFF recall, quiet errors
   by phase, branch calls, and the previous fixed current-only neighbor probe.
   Neither boundary nor probe is refitted on altered timing. Check signed current
   forecast correctness separately from timing detection.
2. After reporting the frozen transfer result, run the margin investigation
   regardless of success. Pad the prior two total/ratio rectangles using only
   original seed9023 training/calibration data. Five fixed padding values per
   coordinate:0,.005,.01,.02,.04 times training coordinate SD. Same padding for
   both polarity regions, giving25 candidates, with no new region or mechanism.
   Calibration includes clean currents and five independent1%-training-SD
   Gaussian current perturbations (seeds9040-9044). Require <=5% quiet calls
   in every calibration condition; maximize worst ON/OFF recall across those
   conditions, then mean summed recall, then lower worst quiet error, then
   smaller padding. If no candidate qualifies, record failure without relaxation.
3. Freeze the padding choice before collecting50 original-dwell confirmation
   trials, seed9027. Evaluate clean and five new perturbation seeds9050-9054.
   Also evaluate the padded boundary on the already collected altered-dwell
   data, which were never used for padding selection. Report noise ranges and
   phase errors, not just pooled accuracy. Compare the unpadded boundary on the
   same samples. Do not pick another padding after evaluation.

Transfer success requires >90% detection of both polarities and <=5% quiet
errors separately at each timing, with phase errors also visible. Margin evidence
requires those criteria on clean and each noisy confirmation. These are point
estimate diagnostic criteria, not statistical guarantees or M1A gates.

Different dwell changes event density and warmup/cycle durations; report counts.
This tests tempo transfer for a fixed two-position trajectory, not arbitrary
vision. Failure can implicate the trained representation or the expression
boundary; a frozen original-tempo probe does not establish absence of information
at another tempo. Random unannounced dwell changes are deliberately excluded:
their precise next event could be inherently unpredictable. No training,
backpropagation, topology/sign change, or deployment of offline thresholds.
