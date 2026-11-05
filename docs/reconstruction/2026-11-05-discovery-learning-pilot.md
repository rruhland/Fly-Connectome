# Numerical discovery learning — paused after a promising pilot

Paused at the user's request. All scheduled games finished and no experiments
remain running. No independent qualification has been started.

The new learner discovered a changed three-command motor response from its own
requested actions and public visual displacement. With an unannounced cycle of
the controls, including relocation of the neutral command, it returned **78/82**
balls; the same acquired model with numerical updates disabled returned **0/16**
on matched starts. This is useful development evidence of required adaptation,
not merely a frozen controller tolerating a changed environment.

| Phase/control | Games | First returns | Later returns | All returns |
|---|---:|---:|---:|---:|
| Initial normal controls, learning | 8 | 7/8 | 34/34 | 41/42 |
| Changed controls, learning | 16 | 15/16 | 63/66 | 78/82 |
| Changed controls, acquired coefficients frozen | 16 | 0/16 | No later opportunities | 0/16 |
| Normal controls restored, learning | 8 | 5/8 | 25/25 | 30/33 |

All exploration and recovery losses are included. After restoration the first
three games were losses; the next five returned30/30. That last-five observation
is descriptive, not a selected qualification window. The remapping first-return
rate15/16 is93.75%, so this pilot does not establish95% reliable first returns.
There is only one fresh learning stream and one matched frozen comparison.

## What learned, and what was inherited

The response model has96 continuous coefficients: three anonymous command
channels, each with32 history taps. Recursive least squares updates them from
observed position differences and actual elapsed time. Missing pre-game requests
have zero feature vectors, distinct from pressing any button. No backpropagation,
physical state, intervention label, precomputed mapping catalogue, or expert
action label enters learning. The mapping is changed only at execution.

Updates happen during games; learned coefficients, covariance and exploration
RNG carry across games and both changes. Saved public-stream comparisons found
3,817 command choices changed by within-game numerical updates relative to
game-start coefficients on the same actual history. This is evidence of action
effects; the separately executed frozen control supplies performance attribution.

Vision and the acquired relational object binding/ball-following objective are
inherited. A fixed short-horizon controller evaluates three candidate commands
through the learned response model. The objective, horizon, feature history,
confidence gate and occasional four-frame exploration holds are supplied priors.
This stage learns sensorimotor dynamics through self-supervised visual prediction;
it does not newly discover the reward objective, vision, or Pong strategy.

## Earlier routes informed the choice

- Scalar-command response learning acquired good control and transferred to an
  eight-command delay: live80/83 versus acquired-frozen70/76 and old servo29/44.
  It preserved normal play48/48 before and after the delay.
- Stronger motor gain: live48/48, frozen46/46. Changed gain plus neutral drift:
  live42/43, frozen45/45. These were robustness results, not evidence that ongoing
  adaptation was necessary. We did not continue a scalar parameter sweep.
- Independent Sol challenge proposed full three-command remapping, exposing the
  scalar model's ordered-command assumption. Parent selected the categorical
  response representation; execution implemented it. Both identified why absent
  history must differ from an actual neutral request.
- A harder synthetic combination of additional delay and remapping failed:
  low forecast controllability triggered inherited fallback, which mostly chose
  the newly neutral action. The selected no-added-delay synthetic case passed.
  Combined-delay robustness remains explicitly unproven.

Parent also batched forecasts while retaining their floating-point summation
order: ~12x faster forecast microbenchmark, with exact complete replay on six
saved scalar-model games. This is not a claim of12x total training speed.
Primary RLS/NLMS and eligibility implementation sources are linked in the
[challenge note](../plans/2026-11-05-discovery-learning-challenge.md).

## Validation, cost and saved state

All48 scheduled categorical-pilot games completed in319.81seconds, with27,328
transition cameras and48 initial cameras. Every game passed complete public-input
replay; source/assets remained unchanged. Frozen controls retained ordinary
history and matching continuing-probe probability while disabling coefficient
and covariance updates. Three categorical causal/continuation/synthetic tests
and nine earlier scalar/harness tests passed.

Across the four executed native development allocations this session:
156games,115,096 transition cameras plus156 initial cameras,855.19seconds summed
allocation wall time. No native work was performed for the superseded prelaunch
specifications. The summed wall
time excludes implementation and research time. No qualification was run.

Artifacts: `/workspace/m1b-experiments/pong-categorical-motor-learning-20261105-r1`.
Spec SHA: `d2aa3fbafcf47bae84b7e637495473e0289d70dd20b17fd9fb6b70f610e69506`.
`checkpoint-end-A.json` preserves the acquired pre-change numerical model;
`checkpoint-latest.json` preserves the final restored-control learner. All
per-game public inputs, event tapes and before/after states remain there.
The [committed summary](2026-11-05-discovery-learning-pilot-summary.json) records
aggregate results. Runtime code is `pong_categorical_motor_fir.py` and
`run_pong_categorical_motor_learning.py` under `scripts/experimental`.

## Resume here

Do not automatically restart. On the user's next authorization, qualify this
unchanged mechanism on two fresh independent learning streams with fixed
acquisition/change/restoration windows and a matched acquired-frozen control.
Predeclare reliable first/later-return criteria, minimum later exposure, recovery
misses/time and source/checkpoint checks. The proposed next comparison is larger
than this pilot, but no starts or allocation have been prepared. Do not call the
milestone complete unless that confirmation passes; investigate concrete failures
instead of selecting favorable tails or extending until success.
