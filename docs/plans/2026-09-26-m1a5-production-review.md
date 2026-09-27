# M1A.5 probabilistic visual state: production review

**Decision update, 2026-09-27:** the user accepted the revised hybrid/probabilistic
contract and authorized production promotion. The implementation is now
`fly_connectome.vision`, with a packaged default checkpoint and `vision-run` CLI.
The [current roadmap](2026-09-27-m1a5-to-m1b-roadmap.md) governs further work.
The report below is the preserved pre-promotion review and its limitations; its
statements that production was unchanged describe that historical review point.

The bounded **hybrid, probabilistic candidate is implemented and passes its
registered experimental gates**. It is ready for concrete production review.
It is not deployed. The older event-only and single-point requirements did not
pass. Production `src/` is unchanged.

## Recommended revision

Promote a generic visual evidence/state service with a separately learned
distribution of future spatial outcomes. Its sensor is a grayscale frame at
**every camera sample**, plus events. Its output is a mixture, not a promised
correct single trajectory. This explicitly changes the sensor and forecast
acceptance contract: noisy event-only observation failed, while retaining
competing observed futures yields useful likelihoods and marginal intervals.

Local event credibility, context/outcome associations and motion-history/outcome
associations are learned. Region extraction, association beams, coordinate
normalization and support propagation are engineered. This is not end-to-end
learned object discovery or a learned optic-lobe world model. There is no
backpropagation, reward training, Pong identity, collision rule, wall model,
paddle crop or intercept meaning in inference or local credit.

## Verified evidence

| Check | Result |
|---|---|
| Two learned contextual relations, four rotations | 16/16 each; frozen/shuffled controls much weaker |
| Fresh 216-scene spatial belief, horizon 4 | NLL 2.817 vs strongest calibrated fixed control 3.687 |
| Same, horizon 8 | NLL 3.494 vs control 4.259 |
| Nominal 90% marginal intervals | 92.7% / 91.1% at horizons 4 / 8 |
| Family NLL and observation-coverage gates | Pass, including both horizons |
| Saved checkpoint replay | Max score difference 2.4e-7; all learned state/RNG frozen |
| Noisy two-entity crossings / five-sample outages | 36/36 identities each; zero final extras |
| Missing-evidence support | 180/180 target locations; no missing cached forecasts |
| Identical ambiguous prefixes | Identical complete mixtures and probabilities |
| Visible contradiction | 8/8 continued reidentified; 0/8 vanished reidentified or supported above .5 |
| Full repository suite | 694 passed, four skipped |
| Standalone complete `step`, this CPU | 76.2 samples/s; earlier replay 59.4 samples/s |
| Candidate checkpoint | About 665 KiB; bounded learned memory |

NLL is negative log probability density; lower is better. Coverage is marginal
per coordinate, not joint 90% coverage or a conditional guarantee for every
motion type. Gap coverage is 100% on straight crossing scenes: conservative,
not calibrated hidden-gap uncertainty. The 900 cached-score evaluations include
repeated evaluations of issued forecasts, not 900 independent forecasts.

## Failed requirements and limits

- **Event-only remains unaccepted.** Sparse image anchors also failed earlier.
  Dense images are additional runtime input, not a hidden training teacher.
- **The old point no-regression gate still fails.** On this fresh straight-motion
  split, eight-sample mixture-mean error is 4.17 pixels versus 1.54 for mean
  recent displacement. An exploratory highest-density mixture-center estimate
  also fails (4.52); no point selector is promoted.
- **Reidentification is unfitted in this checkpoint.** `.5` is an explicit prior
  with zero outcome samples. Matched-prefix Brier .25 verifies honest ambiguity,
  not learned existence prediction. Online observable endpoints can fit this
  separate frequency; hidden existence is not its target.
- Synthetic tests change shapes, phases, angles, speeds, polarity and noise.
  They do not establish natural-image/robotics transfer, arbitrary new dynamics
  or unconstrained object discovery.
- Measured T4/T5 inputs have not shown added value on this context task. The
  prior graph audit has a vertical receptive-field confound. Its approximately
  eight samples/s including setup is **excluded** from standalone timings.
  Keep graph execution off this candidate's critical path until justified.
- Loading resumes parameters in a **new scene**, not active identities/pending
  deadlines. Horizons 1/4/8 count camera samples, not milliseconds.

## Integration boundary

`ProbabilisticVisualState.step(events, frame, learn=False)` takes float tensors
`events[2,H,W]` and `frame[H,W]`; `None` declares an unavailable camera sample.
The checkpoint uses H=W=64. Other workspaces require explicit construction and
transfer of shared components, as the 32x64 ambiguity audit does. Anisotropic
scene resizing does not preserve the local receptive field.

Outputs separate observed entities, carried hypotheses and forecasts. Forecasts
contain mixture centers/weights, unit-pixel Gaussian component sigma, mean,
marginal interval, horizon, issue sample and evidence age. A mean can lie between
plausible futures. `learn=True` credits issued forecasts only against actual
same-identity endpoints; unavailable camera endpoints are censored. Propagated
guesses never become training targets.

The review bundle includes the checkpoint, exact experimental runtime dependency
closure, checksums, this report and an isolated smoke test. It requires the
installed Fly-Connectome project and PyTorch. It changes no production defaults
and connects nothing to reward/motor learning.

After review, the proposed cutover is an **optional hybrid visual-state frontend**
that exposes structured beliefs to later brain work and preserves the existing
event/connectome route. Do not silently replace the sensor or activate M1B.

The production decision is concrete: accept this hybrid probabilistic M1A.5
contract, or retain the older requirements and treat it as a successful research
candidate rather than milestone completion. The bundle does not waive old failures.
