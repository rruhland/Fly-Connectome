# Exact speedups for serial native training

The combined opt-in run measured **15.761 to 13.299 process CPU seconds
(15.6% less)** on one complete normal live training game. Every one of its
283 ordinary stereo exposures, 566 issued joint commands, native referee
events, opponent draws and learned motor records matched stock. The final
NPZ checkpoints were byte-identical, including the 22 newly credited
motor fragments. A separate normal game testing the camera change alone
measured **16.255 to 15.283 seconds (6.0% less)**, with all 303 stereo pairs,
605 commands, events and the 23 newly earned fragments identical.
[Combined matched game](combined.json.gz) · [Packed-camera matched game](paired.json.gz)

| Complete matched training game | Stock CPU | Opt-in CPU | Changed components |
| --- | ---: | ---: | --- |
| 10762 | 16.255 s | 15.283 s | Packed stereo |
| 15541 | 15.761 s | 13.299 s | Packed stereo and compiled opponent scoring |

Each comparison starts from the same frozen initial checkpoint and then
learns during one full actual point, using the unchanged wide random
physical opponent and the ordinary 240-alternation safety cap. Reported
CPU covers the whole training loop and excludes only the separately
measured overhead of hashing frames for verification. The reports retain
the full referee and actuator tapes and per-component timing. The two
modified components account for about 8.9% of baseline CPU in the
combined game. Unmodified timing also varied. The whole-loop 15.6% is
an observed result, not a promised speedup on every game. The 63-return
record was the performance motivation, not a matched benchmark here.

## What the profile showed

On a separate live profile of the normal 10762 game, camera acquisition
used 6.02 of 17.38 CPU seconds and near-side native plans used 6.59.
Inside camera acquisition, setting up the scene was only 0.084 s.
Rasterization used 2.83 s and raw readback 2.42 s. Motor retrieval used
0.06 s. This profile motivated keeping the prior exact traced-prefix
reuse and focusing on readback and opponent math instead of rewriting
retrieval or duplicating camera scenes. [Serial training attribution](profile.json) · [Earlier traced-prefix reuse](../tt45_efficiency/README.md)

* The stereo renderer keeps the same MuJoCo draw and cached static
  background. On the reviewed little-endian Mesa/EGL format, it reads
  native packed BGRA and 32-bit floating depth plus stencil, and applies
  the same strict depth comparison while converting moving pixels to RGB
  in a fused host loop. Neither field of view nor pixel sampling changes.
  Unreviewed camera setup, formats or driver, or a missing optional build,
  fall back to the previous camera path. A GPU composition shader matched
  pixels but was slower in a small probe. A raster crop was rejected
  when it changed pixels. Neither is part of this opt-in.
* In the opponent's approximate scoring stage, the compiler advances
  each candidate independently through the same 105 steps. The weighted
  distance and final selection keep the stock arithmetic and interpretation.
  All 20,412 scores and their sort order matched bit-for-bit across six
  real candidate batches. That component fell from 0.458 to 0.040 CPU
  seconds in the separate replay. The fixed opponent remains a legal
  reactive physical paddle, and its exact impact, mechanical actuator
  limits, and random draws have not changed. Unreviewed inputs and
  missing compilation use the stock scorer.
  [Bitwise real-batch test](opponent-kernel.json)

## Opt in and reproduce

The default and earlier pinned runners are untouched. To opt in for an
ordinary serial training entry point:

```python
from scripts.experimental.tt54_training_speedups import enabled
from scripts.experimental import tt46_continuous_season as season

with enabled():
    season.run(args)
```

This does not replace downstream overrides. For a managed benchmark on
the supported software renderer, run:

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt54_exact_speed
python -m scripts.experimental.tt54_exact_speed --benchmark --index 0 --out /tmp/tt54-packed.json.gz
python -m scripts.experimental.tt54_exact_speed --benchmark --fast-opponent --index 1 --out /tmp/tt54-combined.json.gz
python -m scripts.experimental.tt54_opponent_approx_audit
python -m pytest -q tests/test_tt54_exact_speed.py
```
