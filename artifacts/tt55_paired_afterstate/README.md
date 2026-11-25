# Two achieved strokes from the same incoming state

This tests a possible upstream cause of the TT52 whole-rally difference. It compares two recorded, physically landed arm strokes from the same receiving state, then gives both strokes the same six new independent native opponent goals. Different opponent bodies and ball responses after the stroke are part of the measured effect. Goal equality does not require equal realized replies. [Terminal bank cross-play](../tt52_terminal_bank_crossplay/README.md)

The fixed external cohort takes all 17 two-landed first-command forks from the TT52 before/after comparison. Each pair has identical issued commands and native events up to the first altered command, and the specific incoming far impact, bounce, receive line, and target match. A separate planned training cohort has 32 releases, selected from previously mined ordinary play with the same physical-pair checks. It excludes the TT46 season and TT48 official release IDs, the entire 17-release external cohort, and reserved central-motor work. Selection does not read the original reply after either achieved stroke. [Frozen pairs and draw streams](protocol.json)

The harness rebuilds native physics and RGB history from launch before each recorded opponent request. It issues the actual archived motors through the near-racket separation. Every redraw must preserve the prior stroke's native separation and legal landing. The far paddle then physically responds to the assigned goal with its ordinary controller. A single shared, frozen RGB/encoder player receives the next ball, using its observed body, the stock ranker, and the common pre-season bank. Its planner starts with no archived incumbent, while the optical and body observation history extends back to launch. The evaluator's draw seed, physical ball state, and projected receiving geometry never enter this policy.

Physical next legal return, contact, and opponent offer delivery are counted separately. Geometry uses the measured crossing at y = −1.10 m projected to y = −1.53 m. The central-low event is projected height < 1.00 m and |x| < 0.10 m, an evaluator-defined risk band rather than a physical impossibility claim. An early strike can make the crossing unobservable. If its logged postbounce separation supports an isolated native extrapolation, that geometry is labeled extrapolated. The paired difficulty difference uses jointly observed draws only, and the score difference keeps all assigned draws, including far-player failures.

The first three draws can identify a preferred stroke and the second three check that preference, then the halves exchange roles. The summary bootstraps by release, not by individually correlated draws. A causal prediction uses only the two existing nominal candidate readouts at the original common decision tick. It cannot use true departure, the later receiver context, or the opponent draw. A live game is warranted only if the held paired actual-score advantage over the existing risk ranker has a positive lower interval bound.

## The 17 archived forks

All 204 new native replies were delivered and had measured geometry. Both actual strokes scored the next legal return on 82 of 102 common goals. The before stroke alone scored on 11, the after stroke alone on six, and neither on three. Totals were 93 versus 88 next legal returns and 96 versus 94 racket touches. The average difference, −4.9 percentage points for the after stroke, had a release-bootstrap interval of −17.6 to +5.9 points. This is not a uniformly better outgoing bank. [Held pair metrics](held-report.json) · [Native action and event tapes](held.json.gz)

Stroke-dependent changes were large on some receiving states and repeated under different draws. On 11133 the after stroke lowered the mean projected next ball by 139 mm, and the fixed receiver went from five next legal returns to zero. On 5222 it raised the next ball by 45 mm and the receiver went from four to six, even though the original full rally shortened from 23 to three. Height differences between the two strokes correlated .759 across draw halves. The three pairs with a nonzero actual next-score contrast in both halves agreed on which stroke was better. Selecting a stroke on one half gave +4.4 percentage points on the other versus no preference, with a release-bootstrap interval of 0 to +10.3 points. The effect is measurable but sparse.

The central-low geometry appeared twice after the before strokes and four times after the after strokes. The fixed receiver converted **0 of 6** central-low replies. Those six examples support keeping the band as a separate evaluator measure, but are too few to identify a generally reliable avoidance rule. They also do not explain all score differences. On 11133, for example, only one of the six replies after the worse stroke was central-low.

## Can the arm choose the better shot before making it?

The independent training run finished all 32 physical forks. A score preference learned on the first three opponent draws and applied to the other three, then exchanged, gained 4.4 percentage points against choosing at random. This measures repeatability after practicing the same two strokes. Thirty training forks and sixteen held forks recorded a causal estimate for both actions at the decision tick. On these held forks, choosing using only the learned pre-contact estimate gained 2.6 percentage points over the existing risk ranker, with a release-bootstrap interval of −4.7 to +9.9 points. It missed the preset deployment threshold. The ordinary learner retains its previous outgoing policy. [Training physical comparisons](train-report.json) · [Released predictions, comparison and checkpoint](preference.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt55_paired_afterstate held
python -m scripts.experimental.tt55_paired_preference held
python -m scripts.experimental.tt55_paired_afterstate train
python -m scripts.experimental.tt55_paired_preference train
python -m scripts.experimental.tt55_paired_preference learn
python -m pytest -q tests/test_tt55_paired_afterstate.py
```
