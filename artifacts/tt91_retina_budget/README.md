# Cheaper actual RGB with the same TT75 viewports

Conservative scene culling lowered actual RGB sensing cost by **24.1%** across the six held physical offers, from **29.67 to 22.52 ms of process CPU per exposure**. All 309,878,784 cropped RGB channel bytes, 2,802 eye-detector results, 1,409 exposure times and triangulated packets, and 375 controller-clock fit packets matched exactly. Both saw 1,314 valid stereo observations and made the same 178 dense estimates and 197 fallbacks.

The benchmark replays the six sealed TT75 physical receiving tapes once. Both sensor implementations see the same simulated scene at exactly the same causal exposure time. They maintain independent optically aimed crop histories and independent downstream DenseFlight fits. Execution order rotates across exposures. Renderer timings exclude the separate comparison and precision work. Nothing changed the prespecified TT75 32-point course. [Frozen six offers](protocol.json) · [Measured RGB and fitted-state receipt](rgb.json.gz) · [Summary](report.json)

The first optimization tried true side-by-side native rendering and one readback. It saved work but changed some raster values in the right eye. Detection and triangulation happened to be exact on 7398. On 13344, 49 of 249 triangulated observations were not bit-identical despite preserved stereo availability. We rejected side-by-side as an exact drop-in. This is recorded as an incomplete two-offer engineering diagnostic, not as the final renderer. [Packing diagnostic protocol](packing.protocol.json) · [Rejected raw receipt](packing.json.gz)

CulledMonoRetina keeps the two original viewport origins and the same actual off-axis projections and exposure clock. Before either eye draws, a renderer-only broad phase removes bounded objects whose conservative spheres cannot reach either view. It applies uniformly to rendered objects. The estimator never uses the culling geometry, object identities, simulator ball coordinates, or future images. The integer chroma gate compares the same integral pixel products and computes the same covariance. Pin the implementation to the tested optics, preview geometry, MuJoCo/EGL configuration and renderer ABI. A different fixture or renderer version needs its own parity check.

## Cycle budget and precision

| Measure | Original | Culled original viewports |
| --- | ---: | ---: |
| Sensor CPU seconds per captured simulated second | 3.561 | 2.703 |
| Sensor CPU seconds per actual simulated second of prefix | 2.600 | 1.973 |
| Sensor CPU seconds per held offer | 6.97 | 5.29 |

The cull omitted 99,382 of 147,105 geometry submissions over the crop scenes. It did not change timestamps, global reacquisition calls, color moments or fallbacks. Equal-release medians of inbound native-truth error remained .212/.311/.310 mm in x/y/z, .00179/.00286/.01037 m/s in velocity and 5.36/4.99/3.05 rad/s in spin. The estimator cost is reported separately. The accelerated software renderer still uses more than one CPU second per simulated second of active 120 Hz sensing.

## Opt in

The alternate class has the same constructor as Retina. In an independent process that uses the TT75 ExposureSession, select it as the constructor's Retina factory. This is not a change to the prescribed TT75 experiment.

```python
from unittest.mock import patch
from scripts.experimental import tt75_observer_adapter
from scripts.experimental.tt91_packed_retina import CulledMonoRetina

with patch.object(tt75_observer_adapter, "Retina", CulledMonoRetina):
    exposures = tt75_observer_adapter.ExposureSession(arena, rig, wide, public_settings)
```

Keep the own-context close before destroying each MuJoCo framebuffer. Otherwise closing one renderer while another GL context is current can leave a subsequent game with frozen images. Installing this optional backend does not change exposure or fit settings.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt91_retina_benchmark plan
python -m scripts.experimental.tt91_retina_benchmark run
python -m scripts.experimental.tt91_retina_action plan
python -m scripts.experimental.tt91_retina_action mono
python -m scripts.experimental.tt91_retina_report
python -m pytest -q tests/test_tt91_retina_budget.py
```
