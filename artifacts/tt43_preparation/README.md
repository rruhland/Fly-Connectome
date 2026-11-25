# Learning which outgoing shots leave coverable balls

The 38-return TT42 point ended on an unusually low reply. The preceding arm return landed at far y = +0.59 m. The opponent contacted it close to the net, at y = +0.65 m, and placed the next delivery at a projected height of 0.86 m at the learner's deep receiving line. The arm made no contact. The preceding arm strike's accepted private predictions had a very slow far receiving pace of 1.67–1.89 m/s. In the exact RGB-only replay, the ball already projected a short far bounce when the current outgoing ball was near y = −1.14 m, 0.812 seconds before the next far contact. The first incoming optical estimate below 0.95 m arrived with 0.383 seconds left to its projected deep receive. These measurements motivate avoiding difficult future offers by discriminating between already playable outgoing strokes. They do not establish that the proposed ranker would extend the record point. [TT42 physical witness](../tt42/random-38.json) · [Exact optical cue replay](record-terminal-cues.json)

## Learned association

A seven-variable prediction describes the planned stroke's landing, net clearance, and later far flight. For each physically landed and answered practice return, the most recent issued prediction had to come from the eight control ticks before real contact and locate that contact to within four ticks. The two retrospective training labels are whether the next delivery projected below 0.99 m from its measured physical flight and whether the learner failed to land it. The true post-bounce momentum is only an evaluation and practice label. The live ranker never receives the true ball state, next placement or future score. No practice data from the direct assessment releases or the record release 14000 enters the fit.

The training bank covers 626 distinct credited physical strokes from 47 releases. The train-only feature scales and nonlinearities are fixed geometric descriptors. Ridge regression uses one direct linear solve, with no backpropagation. In seven release-group folds, the most risky fifth under the learned ranking contained 18 of the 32 low deliveries and 25 of the 69 next misses, versus 4 and 11 under the foldwise prevalence control. Brier scores improved from .0486 to .0456 for low balls and .0985 to .0937 for misses. Apart from these held-within-practice labels, this is not a claim that modifying a stroke improves a point. [Fit, provenance, and grouped validation](outgoing-risk.json)

## Live comparison

Both arms use the same original elastic direct joint field, frozen physical practice memory, camera and encoder input, private candidate population, robust rescreen, range of continuously randomized TT37 replies, one serve per point, and 240-return stop. The learned ranker only changes the final ranking among the currently generated nominally legal finalists. It may select a candidate only when supported by the practice bank, at least as robust under the original two alternative optical states, and close in the ordinary return quality. It does not force any early preparation, add a control input, change the opponent, or spend additional private rollouts. A later physical offer can change when a different stroke is chosen, even though its target is independently drawn from the same original continuous stream.

Rerank-negative points are checked for bitwise-identical archived actions and events. For changed points, the whole unchanged command prefix precedes the first actual reranking. The output contains both whole-game issued tapes and the paired stock scores for each preassigned release.

The first three changed physical points illustrate why this measure is worth separating from rally length. In each case the altered own stroke was legal on the same physical incoming offer and the following reply was higher. On release 11884 the stock point ended on the 0.974 m follow-up. The altered shot elicited a 1.025 m follow-up on the identical random placement draw and the arm extended the point from four to seven returns. A command-only replay reproduced this point, including its reactive returns, and checked 6,041 solver steps with zero full-CAD penetrations. The other two points got a higher immediate follow-up but ended sooner on later deliveries. [11884 replay](outgoing-11884-cad.json)

## Generating alternatives for coverage

The original finalists can all predict a bad follow-up. A separate pilot uses the same learned objective directly when searching around the stock winner. When enough time and local practice support remain, the controller samples signed changes to early joint displacement and final contact drive, then samples again around the lowest combined current cost and predicted next risk. This search is bounded and uses a separate random stream. The generated choice must be legal under the nominal camera-initialized twin and both original uncertainty settings, lie inside the learned practice support for all three, and improve the next-ball estimate without incurring substantial extra current-return cost. A separate score check is still needed because these are hypothetical screenings. The pilot reuses the first six broad matched starts as a mechanism check, so it is not a fresh release-held confirmation of the earlier comparison.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt43_outgoing_coverage fit
python -m scripts.experimental.tt43_outgoing_coverage run
python -m scripts.experimental.tt43_outgoing_coverage_report --verify
python -m scripts.experimental.tt43_terminal_cues
python -m scripts.experimental.tt43_outgoing_generation
python -m pytest -q tests/test_tt43_outgoing_coverage.py tests/test_tt43_outgoing_generation.py
```
