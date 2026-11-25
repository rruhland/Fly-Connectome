# Learning a region of outgoing flight that leaves easier replies

An achieved outgoing stroke can change the next incoming height even if the same independent TT37 target is drawn. On practice release 6619 the current incoming ball and bounce were identical and both arm strokes landed. A moderate-paced outgoing arc got no next ball below 0.99 m in 12 redrawn continuations, versus six lows for the slow arc. Both strokes received legal opponent answers on all 12 draws. This is an effect of changing the whole achieved stroke. It does not identify one coordinate as the cause or show that every possible target is safe. [Native continuations](physical-replies.json.gz)

## Learn from achieved shots

The observation study starts with distinct scored outgoing departures from 138 historic practice releases. The current assessment release IDs and record release 14000 are excluded. Censored paths do not count as failures. The shared flight descriptor uses only the location and translation velocity at separation and a simple drag-and-gravity projection. Its bounce depth is an index, not the measured table contact location. Average absolute error versus the actual landing is 0.137 m in depth and 0.014 m laterally. Labels use physically measured near arrivals. “Low” means the extrapolation from the original receive line to y = −1.53 m is below 0.99 m. “Hard incoming” uses the same height measure below 1.018 m on the current ball.

On 2,182 tracked replies the release-grouped ridge model using current incoming and the exact next random request scored .0775 Brier for next lows. Adding outgoing reduced it to .0743. Using outgoing alone scored .0751 and put 102 of the 189 lows in the highest-risk fifth, versus 65 when using incoming and the request. The next request is available only for this benchmark. Neither the goal fit nor the actor consumes its value. These are held-release associations, not isolated effects of changing a stroke. [Data, folds and exact-pair assignment](physical-arc-plan.json.gz)

A disjoint training subset learned a broad goal envelope with projected lateral x within ±0.16 m, projected bounce depth 0.34–0.76 m, and forward speed just before projected bounce 2.10–2.65 m/s. It contained 376 real arm returns across 79 releases, four untracked, unanswerable or low follows, and none of these failures after 23 hard incoming balls. A populated neighboring slow interval was substantially riskier. The goal is a smooth center/depth/pace penalty with a wide margin, derived from this group of scored flights. No backpropagation or forced subtype of opponent reply is used. [Frozen goal fit and comparisons](learned-envelope.json)

## Test the causal contrast

Five held-within-practice pairs use corresponding whole actions that actually landed on the same exact recorded incoming impact and bounce. Pairs are assigned by a fitted risk contrast before redrawing any responses. The model used for pair selection excludes those releases. At the old native arm-hit prefix, one evaluator-only restore runs the original arm command tape into separation, then holds the control references and measures the next actual native far hit, near bounce and receive line. The original target must reproduce available recorded measurements exactly before any new draws run. Each of the 12 preassigned independent seeds replaces only the next random request, and both shapes receive the same request. The controller and finite-mass opponent still decide from the physical trajectory. No reply is clipped, fixed, or repeated because of its outcome. These are sampled draws from the full distribution, not exhaustive coverage of a continuum.

All 120 redrawn replies were answered and tracked. The lower-predicted-risk strokes led to zero lows across 60 replies versus eight across the corresponding 60 higher-risk replies. Six of those eight occurred on 6619 and two on 2920. The 36 replies to strokes in the frozen goal envelope had no lows. The lower-risk strokes did not maximize average return height. They averaged 1.049 m versus 1.053 m, but their worst reply was 1.002 m versus 0.953 m. The matrix contains 24 distinct random request vectors across two leg indices. The realized extent covered almost the full field on every request axis. [Paired native draw report](physical-reply-report.json)

## Give the goal to the early search

The live pilot uses the direct-joint controller with the same physical practice checkpoint and normal motor credit after each actual landing, within points and across them. For a proposed movement with a plausible departure, the ordinary short and long trial losses include the same learned smooth flight penalty. It changes which initial joint sequences become parents of transported and sampled proposals. It is not a post-selection rerank and does not add rollouts or modify the stroke geometry. No extra camera, opponent state or true future coordinates enter the actor. Both arms start from the same practice motor memory and use the same unmodified continuous reply law throughout 12 preassigned measured whole games. Labels from these games do not change the frozen flight envelope.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt45_tactical_coverage fit
python -m scripts.experimental.tt45_tactical_coverage sweep
python -m scripts.experimental.tt45_tactical_goal fit
python -m scripts.experimental.tt45_tactical_report
python -m scripts.experimental.tt45_tactical_goal run
python -m scripts.experimental.tt45_tactical_goal compare
python -m pytest -q tests/test_tt45_tactical_coverage.py
```
