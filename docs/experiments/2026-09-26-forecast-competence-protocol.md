# Learned forecast competence, not another endpoint prototype variant

The 2x2 comparison preserves the statistical observation stage: no blank-noise
tracks, 36/36 identities in every camera condition, and adequate fresh history
coverage. The local linear generative alternative does not remove straight-motion
regression. Stop that branch. Curved and varying motion benefit from learning;
simple motion sometimes benefits from a simpler forecast.

Test local predictive-error associations that learn which forecast has been
reliable in a given observed history. Candidate forecasts are the learned direct
model, mean recent displacement, two-step replay and last displacement. These
are generic controls, not named trajectory classes or final representation labels.
Store all issued candidates until the endpoint is actually observed. Only then
associate the original local history with their normalized squared errors. Select
the lowest learned expected error at inference. The base dynamics may continue
learning after credit; never recompute an old candidate using newer weights.

Use the strongest statistical-state/contextual-dynamics checkpoint from the
registered broad training, then 64 new interleaved causal calibration streams
(seeds 171000+, same broad camera distribution). Keep the 256-example local risk
memory and all prior covariance settings. Compare learned competence, shuffled
history/error association and fixed controls. Reserve phases 23/29, headings
18/54/126 degrees and scales .95/1.45 for scoring; add mean recent displacement
to the fixed-control table. Keep prior aggregate, family and coverage gates. No
gate-temperature, error-threshold or expert-bank sweep.
