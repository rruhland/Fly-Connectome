# Review proposal: fixed pairwise visual-feature residual

Status: ready for user review; NOT approved or implemented.

## Why this next

Selected3ac7f87f still has409TRAIN returns and uncertain fresh improvements,
not reliable match-long learning. The64-RBF test did not produce a fitted
candidate. Four-observation velocity smoothing regressed66→52returns. Releasing
ordinary post-burst imitation retained the target's new fifth return but regressed
full160409→394. None was promoted. We stop this sequence of covariance/constraint
variants rather than tuning another preservation weight.

Hypothesis: generic pairwise interactions can distinguish situations the additive
readout couples together. This does not prove that the linear head is incapable.
The RBF basis mostly varied together across whole scenes; a quadratic feature map
has a different representation of joint visual evidence. No game-specific ball,
paddle or interception formula is introduced.

Parent and independent Sol review recommend one fixed centered degree-two
TensorSketch with512outputs. This changes both feature type and feature count;
it is a practical candidate comparison, not an isolated causal test of either.

## Exact scope

* Freeze vision, existing4389input layout, scaler/cap, actuator and3ac linear head.
  Use the original transport, not the rejected velocity filter.
* Derive512 fixed features from the same legal capped visual inputs. Use fixed
  seed831496027 for two independent coordinate hash/sign maps. Divide inputs by
  the existing saved diagonal feature scale, then subtract a frozen equal-episode
  TRAIN mean. Compute the circular convolution of the two Count Sketches using
  FFT. Precompute the sketches of the mean to retain sparse-input efficiency.
* Freeze input centering and output RMS scales from the same2,048label-free TRAIN
  sample used by the RBF comparison. Do not use rewards, donor labels or fresh
  scenes to choose hashes, scales or features. Set the RMS denominator floor to1e-6 times the median positive TRAIN RMS; stop
  if all are zero. No seed/dimension/normalization sweep.
  Weight episodes equally for both mean and RMS, despite12versus13sampled rows.
* Persist and hash both actual hash/sign arrays, mean/scale, RMS denominators,
  base head, residual weights and feature implementation. Compute the branch from
  the already capped raw input; do not recap the combined raw-plus512 vector.
  Preserve the original base Q computation path.
* Add only3×512 output weights, initialized to zero. Exact deployed actions and
  Q values must initially match3ac on saved native inputs. Only these output
  weights learn; no backpropagation, autodiff, teacher actions or privileged inputs.
* Validate the complete848-row own-reward donor104562292 before deriving the
 754-row approach/burst contract (rows0–753, including10forced choices). Keep all
 143known successful opening prefixes hard. The full receipt remains evidence
  of4→5returns; the truncated hard prefix itself contains only4returns.
* Minimize empirical residual score change on the same existing successful TRAIN
  prefixes, with equal-episode weighting, mean-diagonal covariance normalization
  and fixed1%ridge. At most one60second output fit, exact original-row/deployed
  action certificates; no fallback or extra guard/parameter variants.

## Actual outcome and time limits

One10minute experimental budget including calibration, fit, and native checks;
implementation time reported separately. One absolute deadline covers all stages;
incomplete full/new-game evaluations remain explicitly partial. Diagnose centered rank, pairwise feature
behavior, episode-separated linear explanation with a linear control, and inference
cost. These diagnostics inform interpretation; a low effective-rank heuristic
alone must not substitute for the functional fit. Stop for nonfinite/constant or
incorrect features. Check the FFT map against direct small-dimensional products.

A valid fit must produce an actual fifth target return before fixed32TRAIN games,
then full160 if useful, then genuinely unused paired new games if full returns
improve. Report opening/later outcomes and regressions, not just the fitted target.
Use policy-aware exact episode reuse where valid. Keep3ac/d7 immutable; no automatic
promotion. A positive screen is not sufficient to claim reliable learning.

## Limitations and alternatives considered

512sketched interactions are not guaranteed local: collisions and quadratic effects
can still disrupt other states or overfit. Missing visual identity/history cannot
be recovered from identical encoded inputs. This remains an own-hit-reward,
reset-assisted scaffold, not the final continuously learning multi-game agent.

Relative-margin preservation is a plausible smaller alternative, but mainly
reweights already explored soft decision preservation. Adaptive conjunction
selection adds a separate feature-discovery/credit mechanism; we prefer the fixed
map for this next bounded test. Neither alternative is authorized by this proposal.

## Sources and approval

[Pham and Pagh, KDD2013](https://www.rasmuspagh.net/papers/tensorsketch.pdf),
introduction and sections3–4: TensorSketch approximates polynomial interactions
using Count Sketch convolution, avoiding explicit enumeration of all coordinate
pairs. Its kernel approximation results do not guarantee Pong control or retention.
The implementation proposal above adapts that mapping, not its supervised learner.

This is a MAJOR architecture experiment. The user reserved such changes for review.
The completed64-RBF approval does not authorize this512-feature construction.
Request approval for this single bounded comparison; do not implement until granted.

Independent Sol review recommends this fixed test with the asset hashing,
equal-episode calibration and no-recap clarifications now incorporated.
