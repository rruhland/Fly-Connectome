# Native return outcomes as auxiliary reward for hinged-arm exploration

The learned outcome bonus improved the held wrist streaks over shuffled credit. Across the three gravity settings it trailed target reward on the primary mean-streak endpoint, and produced more separately qualified rebounds and more eight-rebound drops. [Full paired data](tt24_outcome_pong.json.gz)

## Source outcome model

Sixteen frozen visual controllers played actual native Pong against the original opponent. They include the zero controller, the four-weight native controller learned by reward, and preset perturbations around both. Forty-eight native starts per controller supply the source fit and 32 other starts per controller supply the source audit. The visual observer samples ball and blade configuration the first time the incoming ball enters a near-blade window during an approach. An actual own return or point lost gives that event its label. The model inputs contain no Pong action IDs or motor commands. [Native parent controller](tt24_joint_pong.md) · [Pinned schedule and units](tt24_outcome_pong-plan.json)

The predictors use normalized tangential gap squared, relative tangential speed squared, the dot product of gap and relative velocity, and speed normal to the blade. The fit adds low-order products of these quantities. Positions are measured in blade radii, with 100 mm for Pong and 76 mm on the arms. Velocities use 0.55 native screen units/s and 1.5 m/s. The cue window begins at 1.55 blade radii of normal separation. The choice of axes and approach window is supplied. A radius-only control uses the same visual or metric observations and rewards smaller tangential separation. The randomized source exchanges true labels within each frozen source actor's observations, preserving actor-specific success rates.

The coherent fit used 1,079 resolved optical events, including 694 returns. The held sample comprised 716 events, including 465 returns. Held AUC was 0.942 for coherent Pong, 0.929 for shuffled outcomes, and 0.908 for the radius-only cue. Resampling whole held native starts gave paired 95% intervals of 0.003 to 0.023 for the advantage over shuffle and 0.016 to 0.053 over radius only. The original code included a convenience stop at a 0.03 AUC margin over shuffle. That threshold blocked the requested physical comparison despite strong native discrimination, so it became a reported source diagnostic before any target evaluation. The registered target protocol remained fixed. [Real source outcomes and fitted weights](tt24_outcome_pong-fit.json.gz)

## Matched receiving experience

All four conditions begin from the same competent donor-trained joint controller. For each of three gravity multipliers, 19 continuous candidate joint policies share 114 short, physically simulated practice episodes. Independent unscored motor probes supply their joint perturbation basis. Practice uses the changed dynamics of three articulated donor bodies and ends after at most 47 action frames. The original bounce judge supplies the number of consecutive qualified rebounds. The practice auxiliary uses the same pre-impact cue window and metrics as the native observer, read from the existing privileged donor metric sensor. The source bonus and both control bonuses use the same candidate-rank scaling, with a maximum of 0.85. The target-reward baseline ranks the same proposals by original rebound count. Each condition freezes one selected policy per gravity. [Experiment code](../../scripts/experimental/tt24_outcome_pong.py) · [Common donor controller](tt24_donor.md)

Full-length evaluation uses new releases on those donors and then four new stock wrist drops per gravity setting. The stock controller receives wrist RGB and its own encoders through the previously calibrated public observer. No stock motor reward or practice affects policy selection. The original judge, collisions and actuation process run on every trajectory. A complete wrist command replay must reproduce all judge events. [Paired target data and action replays](tt24_outcome_pong.json.gz)

## Held physical results

| Selection signal | Sum of longest wrist runs / 12 | Qualified wrist rebounds | Physical wrist blade contacts | Wrist runs of at least eight / 12 | Mean longest donor run / 45 |
|---|---:|---:|---:|---:|---:|
| Native outcome bonus | 67 | 88 | 106 | 6 | 4.02 |
| Shuffled outcome bonus | 34 | 39 | 70 | 1 | 3.49 |
| Radius-only bonus | 60 | 62 | 83 | 2 | 5.73 |
| Target rebound reward | 72 | 77 | 115 | 3 | 5.49 |

The wrist sums for native outcome credit were 15, 25 and 27 at gravity multipliers 0.76, 1.23 and 0.88. At the same settings shuffled gave 15, 8 and 11. Radius only gave 21, 20 and 19. Target reward gave 10, 43 and 19. Native credit led target reward on two settings and fell short by 18 rebounds of longest-streak sum on the third. Native credit led on the secondary count of wrist drops reaching eight, while the longer runs in target reward's strongest setting gave target reward the highest primary mean. The radius-only cue led on the held donors. The three settings provide a limited sample of policy selection under gravity changes.

All selectors used the same 342 finite-force donor practice episodes and their original judge outcomes. All 48 held wrist traces replayed with identical events. The instrumented donor backend also matched all four original judge fields on 27 audit runs. Verification found 155 distinct allocated source and target starts, and reproduced every chosen proposal from its saved practice evidence. [Contact counts and frozen motor policies](tt24_outcome_pong.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_outcome_pong
```
