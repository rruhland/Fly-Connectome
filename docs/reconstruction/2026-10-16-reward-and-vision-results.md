# Reward-credit and frozen-vision investigation

**No Pong acquisition was demonstrated.** At the fixed12k-camera allocation,
own-hit and survival bonuses did not produce competitive fresh score-only control.
The independent vision assay nevertheless shows substantial localization and
forecasting function on its two tested trajectories. These results narrow the
next learning question; they do not justify replacing the visual architecture.

## Three matched reward arms

The [prospectively registered diagnostic](2026-10-16-reward-credit-diagnostic-protocol.md)
used the same from-zero seed, generic frozen214-coordinate DCT transform/cap256,
true-online SARSA alpha.001/lambda.995/epsilon.2/hold1, .25 opponent and12k cameras
per arm. Only training reward differed. Frozen evaluation was ALWAYS raw-score,
greedy, full vision,24matched DEV rallies with20s limits. No backprop or physics
inputs entered the policy. Contacts entered only the explicitly labeled scalar
hit reward. Every arm retained raw score and separate learning-reward components.

| Training reward | Fresh wins/24 | Own-hit wins | Contacts | Losses/timeouts | Score/rally | Return/physical second |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Score only | 8 | 0 | 2 | 16/0 | −.3333 | −.18648 |
| Score +1 per own hit | 9 | 1 | 7 | 15/0 | −.2500 | −.09872 |
| Score +.1 per physical second | 9 | 1 | 3 | 15/0 | −.2500 | −.13358 |
| Initial/down | 9 | 1 | 2 | 15/0 | −.2500 | −.14469 |
| Constant up | 10 | 2 | 7 | 14/0 | −.1667 | −.06780 |
| Random | 9 | 1 | 8 | 15/0 | −.2500 | −.09407 |
| No-op | 10 | 2 | 9 | 14/0 | −.1667 | −.06049 |

All learned policies had eight natural-serve wins; their remaining wins followed
an own hit. All evaluations completed. Hit shaping's seven contacts versus the
score arm's two are NOT contact competence beyond simple controls: no-op had nine,
random eight and constant-up seven. Neither shaping arm beat the strongest simple
control, much less the descriptive14total/6own-hit screen. No arm is selected or
replicated. These are one-seed exploratory comparisons; trajectories diverge after
actions differ, and the three policies share24serve clusters, not72independent
trials. Negative12k endpoints do not prove asymptotic inability.

Training delivered the declared feedback, without clipping/fading:

| Arm | Raw score | Hit bonus | Survival bonus | Learning reward | Own contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score | −40 | 0 | 0 | −40 | 28 |
| Hit | −51 | 21 | 0 | −30 | 21 |
| Survival | −36 | 0 | 23.93167 | −12.06833 | 26 |

All weights stayed finite, max absolute coefficient below.376, and actual policy
norm² was bounded by256+roundoff. Raw transport still had rare large norms (maximum9030
in the score arm), so the short functional-vision assay does not refute the earlier
rare tracker derivative-corruption problem. Cached1024-row endpoint readouts were
state-varying: action counts[119,431,474]/[269,336,419]/[433,330,261], median state
action-spread.172/.203/.171 versus mean-action bias-spread.079/.059/.042. This
weakens constant-bias collapse as a sufficient explanation; it does NOT establish
that visual information was necessary for the evaluated behavior.

The run committed36,000learning+7456evaluation cameras (43,884images with resets),
zero replay,659.59seconds shared wall;1908.17summed subprocess seconds are a
different measure, not CPU time. Four reward/recovery tests and11allocation/
completion/worker/macro checks passed. The independent parent audit replayed all
43,456recorded actions through physics and verified every reward component and
all36,000TD equations with exactly zero discrepancy. An earlier audit attempted
36k physics rows and stopped only because its naive accumulation differed from
Python3.12's compensated `sum` by~1e−12; the experiment totals were correct.
Frozen learned-hash checks passed throughout score-only evaluation.

Source commit3e79af5; full archive and artifacts:
`/workspace/m1b-experiments/pong-reward-diagnostic-20261016-r1/`.
See `summary.json`, `parent-physics-reward-audit.json`,
`parent-audit-first-attempt.json`, and `cached-endpoint-readout-geometry.json`.

## What the vision assay establishes

The independent [detailed vision report](/workspace/m1b-experiments/vision-integrity-20261016-r1/RESULTS.md)
tested2048cameras each from random actions held ten cameras and a legal privileged
tracking controller, used ONLY to generate diagnostic footage. Ground-truth
annotations never entered candidate features. Both use the same diagnostic seed;
adjacent frames and trajectories are correlated, not independent accuracy trials.

| After five startup frames per rally | Random blocks | Legal tracking |
| --- | ---: | ---: |
| Fresh ball observation match | 88.56% | 89.69% |
| Observed OR predicted ball within3pixels | 99.85% | 99.90% |
| Fresh-match mean localization error | .201px | .195px |
| Missing fresh observations bridged | 219/222 | 207/209 |
| Mean8-camera ball forecast error | 1.944px | 1.790px |
| Matched persistence error | 5.457px | 5.421px |
| Matched encoder-velocity forecast error | 2.881px | 2.947px |
| Own-paddle y90%interval coverage at8cameras | 61.0% | 42.5% |

Missing direct detections therefore usually mean predicted belief, not blindness.
Single-pixel ball appearances disproportionately lose fresh matches. No matched
consecutive identity swap occurred here; public ball-ID changes were three/zero.
This does not exclude rare long-trajectory switches already captured elsewhere.
Localization error is conditional on matching with a3px gate, not unconditional
subpixel accuracy. Forecast comparisons use identical eligible origins, actual
recorded future actions and no reset crossing. Paddle intervals are overconfident
under changing actions; vision receives no future-action input, and interval
bounds are not direct current sparse policy features. This defect is not established
as the cause of reinforcement-learning failure. Forecast improvement here is a
perception diagnostic, not a policy-level forecast-benefit ablation.

4096distinct cameras plus4096exact reprocessed cameras took49.10+45.66seconds;
an8-frame API inspection and final uninspected successors are separately recorded.
Frozen parameter hashes matched. Raw data, executable protocols and figure:
`/workspace/m1b-experiments/vision-integrity-20261016-r1/vision-summary.png`.

## Distinguishing the remaining learning limits

A NEW quarantined cached immediate-feedback test reused4715training and1285
excluded-rally feature rows, the same DCT/cap/SARSA alpha.001/epsilon.2, but made
each row terminal with+1for matching the privileged PD action and−1otherwise.
Three fixed seeds reached83.0–83.7%excluded-rally accuracy after4715updates,
84.75–85.76%after94,300, versus60.1%majority accuracy. Rare no-op labels were
never predicted; balanced recall remained~58%. Thus the rule can learn a useful
immediate action mapping, not all actions or a closed-loop controller. It does
NOT establish linear Bellman-Q adequacy, on-policy coverage or reward-only
acquisition. No coefficients were exported.282,900cached updates/zero new vision,
31.61seconds; `/workspace/m1b-experiments/ORACLE-DIAGNOSTIC-immediate-credit-20261016/`.

Hit feedback remains delayed; survival's dense per-step bonus is nearly action-
independent until termination. At50Hz, an individual action changes the inertial
paddle by much less than a pixel. Hence these negatives do not cleanly distinguish
temporal credit, tiny action advantage, exploration coverage and linear Q-value
misspecification. The vision and bandit results constrain broad failure claims
without choosing one of these explanations.

The next recommended finite comparison is the same DCT condition at hold1 versus
hold10 (~.2s), preserving per-camera vision/journals and proper SMDP physical gamma
and lambda.995^10. This changes both motor persistence and update count; it is not
a pure one-factor credit proof. Parent's quarantined24-independent-serve legal-PD
feasibility check found13wins(5own-hit) at BOTH cadences, but hold10 caused eight
losses versus zero and reduced contacts82→42. Coarser control can still score,
while degrading this fixed controller's rally maintenance. This is a proposal,
not a new allocation. An online immediate-action-feedback ORACLE diagnostic is
a fallback to separate closed-loop distribution from temporal credit, never
candidate imitation or a reward-only learning claim. No additional experiment,
major architecture rewrite, replica or untouched-final evaluation is automatic.
