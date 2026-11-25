# Early visual intent: two richer RGB readouts did not improve TT50

Keep the deployed TT50 sensor; do not integrate either new readout into a body
controller. Neither calibrated stereo geometry nor a longer dense RGB movie
improved the release-disjoint ordinary test. This study does not establish that
early intent is intrinsically unknowable.

The fixed corpus is 24 actual releases with six physically realized broad random
replies each. The first 16 entire releases train; the remaining 8 releases are held.
No 8056, terminal, or selected difficult-source examples enter these two models.
Labels are the ball subsequently observed in RGB at y=-1.10, including misses.
Native far-contact timestamps are used only to report lead time. Predictor inputs
contain no seed, release ID, requested goal, opponent pose/state or future image.
`protocol.json` records all raw hashes, release IDs and input-model hashes; train
and held release sets are disjoint. Re-extraction from all 24 actual raw files
matches all 532 cached baseline feature/label rows exactly. The frozen TT50 model
has additional practice and difficult-source training, so an equal-data TT50
kernel is also reported.

| Median lead before far contact | Equal-data TT50 lateral MAE | Stereo geometry | TT50 + dense movie | Frozen TT50 |
| --- | ---: | ---: | ---: | ---: |
| 0.598 s | 12.48 cm | 15.13 cm | 12.89 cm | 8.19 cm |
| 0.450 s | 11.33 cm | 12.60 cm | 11.52 cm | 9.18 cm |
| 0.291 s | 10.80 cm | 13.60 cm | 12.02 cm | 10.02 cm |
| 0.117 s | 10.34 cm | 12.47 cm | 11.23 cm | 8.66 cm |

Each of the first three gates has 48 held replies; the last has 45. At the 0.450 s
gate, vertical MAE is 2.96/3.10/2.98/3.23 cm in the same column order. The frozen
model's 90th-percentile lateral/vertical errors there are 21.11/7.06 cm, a material
limit for a large early bodily commitment. A model output near 0.30 is not a
validated confidence guarantee.

There is only **one** held low-left case under TT50's existing definition, so this
test cannot establish reliable low-tail precision. Neither new model captures it
at the unchanged deployed body trigger. Both also emit no positive triggers at
that threshold; zero false positives obtained by abstaining are not a sensing
improvement. The frozen model has one label false positive at the earliest gate
and one at the last gate, without capturing the held positive. Full predictions,
continuous errors, tail ranks and trigger counts are retained.

## Tested information and learning rules

`tt57_visual_intent.py` unblends dark-object edge opacity from ordinary RGB,
triangulates its visible centroid using the installed fixed-camera survey, and
fits a two-view projected circular face by geometric least squares. Radius,
normal outer product, position, motion and image reconstruction error feed the
same fixed kernel regression used by the equal-data comparator. No pose or
orientation labels are provided. The surveyed rig is 384×288 with table-height
shifted camera positions; the earlier compact-court constants would be wrong.
4,006 of 4,070 stereo frames produce valid geometric observations. Extraction costs
50.923 CPU / 93.892 wall seconds. It is an approximate disk model, not proof that
all visual orientation information has been recovered.

`tt57_visual_movie.py` adds the full aligned 14×14 photometry and opacity in both
views over 12 past exposures (up to 0.367 s), beyond the existing short moment
summary. Padding is flagged. Rank 24 PCA is fitted on training-release pixels
only, followed by the same fixed kernel solve. This uses analytic spectral/matrix
solves, with no backpropagation, named stroke classification or held tuning.
Movie extraction costs 7.674 CPU / 20.251 wall seconds. Early raw capture starts at
the prior near return, so it does not contain an arbitrarily long preceding far
follow-through history.

## What the raw images do and do not establish

The cache retains only 60×96 far-court crops per eye. The visible face is about
8–12 pixels wide. In ordinary case 13 / release 4214, the paddle leaves this crop at
the middle/late gates; the latest four cropped frames are byte-identical across
six replies whose later observed lateral positions span 41 cm. This occurs only
0.08–0.12 s before contact. A twelve-frame history still distinguishes those
streams, which argues for retaining longer visual context and avoiding a
latest-blob-only failure. See `ambiguous-crops.png`.

In case 21 / release 14071, even twelve earlier cropped frames match exactly across
six later replies with 55 cm lateral spread. That collision is at 0.73–0.78 s before
contact, **earlier than the requested 0.4–0.6 s window**. At the middle gate there
are no exact history collisions in this corpus. These findings limit the stored
crop/history at particular times; they do not prove whole-scene or target-window
impossibility, nor establish a renderer cache bug. `tt57_visual_audit.py`
reproduces the raw equality checks independently of either trained readout.

The grounded next sensing experiment is to retain a wider far working volume and
continuous RGB from the previous far contact through the next one, then measure
actual ball motion immediately around/after the far hit. Higher resolution could
reduce face-edge quantization but was not tested and is not a demonstrated fix.
Only promote new sensing after an ordinary, release-disjoint new course shows
better lateral tails and captures actual low cases without costly false moves.
Because the stronger-signal condition failed here, no controller integration or
new native qualification course was launched for these models.

Reproduce from the tracked raw camera cache:

```sh
PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt57_visual_intent
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt57_visual_movie
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt57_visual_audit --baseline-parity
```

Feature caches are retained for exact receipts. Remove the two feature NPZ files
to regenerate them from pixels. `executed_geometry_source.py` preserves the
exact measured source; the current script adds a raw fallback when the old
untracked TT50 feature cache is absent, with no feature/model changes. The calibration survey used zero exchanged
control frames and no dynamic opponent state. No native worker remains active.
