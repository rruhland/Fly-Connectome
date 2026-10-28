# Review proposal: fixed pairwise visual-feature residual

Status: approved comparison completed in556.11s with modest TRAIN gain, no net later-count gain; separate frozen validation selected under standing continuation mandate.

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
The user explicitly approved this single bounded comparison on2026-10-28.
This is not blanket approval for feature-count/hash-seed/architecture sweeps.

Independent Sol review recommends this fixed test with the asset hashing,
equal-episode calibration and no-recap clarifications now incorporated.

## In-budget review of mixed sentinel outcome

The fixed map had effective centered rank222.16 and verified the14,174-row
contract in7.77s. Actual target4→5 passed. The fixed32TRAIN screen had61→62total,
24→27first,37/58→35/58later, eight games gained16returns and seven lost15.
Its implementation's strict later-return screen stopped after76.26s.

Parent challenged treating this mixed, net-positive small TRAIN screen as decisive
global rejection. Independent Sol agreed to complete the SAME frozen model's
full160 comparison under the ORIGINAL600second deadline. Preserve the initial
sentinel-stop summary and document the narrow screening-gate relaxation. No
refit, feature/basis/hash change, repeated candidate, or budget reset. Full total
and later returns must improve before fresh testing; opening gain alone is not
accepted as reliable rally learning.

## Approved pairwise residual — modest TRAIN gain, qualification pending

Root `pong-rally-pairwise-residual-20261028-r1`; agent157d364→mainf2a4f10.
Fixed centered degree-two TensorSketch512, same capped legal inputs, label-free
equal-episode calibration, frozen vision/base3ac, only3×512outputs fitted withoutBP.
Feature effective rank222.16/numerical rank512; RMS .04331–.08565, no floored
channels; batch/deployed error1.42e-14. Feature36.3microseconds; whole choose
81.7microseconds versus baseline34.3. Heldout linear error1.149 versus linear
control.452 is not standalone nonlinear-capacity evidence.

Seventeen focused tests passed. Zero outputs reproduced92,162saved action/Q rows.
Fit7.77s certified all14,174original hard rows; actual target4→5 succeeded.
Sentinel61→62total,24→27first,37/58→35/58later stopped its strict later gate.
Parent challenged treating this mixed net-positive32TRAIN outcome as decisive.
Sol agreed: full160 ran on EXACT samebundle under ORIGINAL600s deadline,
preserving initial sentinel summary and recording the screening amendment.

Full160:409→412total,143→146first,266/384→266/380later (69.27→70.00%).
No previously successful opening lost.40games gained81returns;44lost78.
At least2returns103→95, at least5returns30→35, timeouts26→34. Net later count
is unchanged: modest TRAIN improvement, not demonstrated reliable later learning.
Exact final action/Q audit passed160receipts/92,316stored rows. No automatic
promotion;3ac andd7 hashes unchanged. Bundle
`eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152`.

Approved experiment ended556.11s,92,106actual cameras+156initials. Implementation
204s separately. Strict full later-count gate meant no fresh games in that test.
Parent now selects a SEPARATE ordinary frozen paired32new-game validation under
the user's standing continuation mandate, at most180s; it is not hidden inside
or a reset of the completed600s test. Same weights/features, no fitting/admission,
no architecture sweep. Motivation: qualify modest gain before any further training.
Use the originally planned unconsumed fresh seed schedule and exclude prior/reserved
streams. Current selected model remains3ac until independent outcomes reviewed.
