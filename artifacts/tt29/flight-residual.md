# Learning errors in the incoming flight forecast

The assay predicts where and when an incoming ball will cross the plane at (y=-0.95) m. It triangulates the orange ball from ordinary fixed stereo RGB at 30 Hz. The early forecast starts around (y=-0.20) m and fits up to five recent observations to estimate the current position and velocity. The nominal isolated table model forecasts the crossing with stock damping and zero spin. A local quadratic through later stereo observations estimates the crossing’s horizontal position, height, time, and velocity. The labels use frames captured at least 35 ms after the early state. [Implementation](../../scripts/experimental/tt29_flight_residual.py)

A regularized model learns the error in each forecast quantity after a completed practice flight. The practice table used a hidden damping setting of 34. The final coefficients were frozen and tested on separate measured openings at both 34 and an unseen setting of 42. The held branches reused the same launch roster. The simulator applied the physical settings for evaluation. The fitted correction received early estimates and later RGB-derived crossings. [Source and held observations](flight-residual.json.gz)

## Forecast error

Each cell below gives mean absolute error for the nominal forecast followed by the learned correction. Source errors are scored online before each corresponding label enters the fit.

| Assessment | Paired estimates | Horizontal position (mm) | Height (mm) | Time (ms) | Lateral speed (m/s) | Depth speed (m/s) | Vertical speed (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Practice setting, online | 16 / 46 | 9.06 → 9.64 | 22.48 → 22.34 | 6.25 → 7.06 | 0.054 → 0.062 | 0.066 → 0.072 | 0.168 → 0.168 |
| Matched held setting | 7 / 40 | 3.86 → 2.61 | 21.76 → 13.62 | 4.43 → 3.89 | 0.032 → 0.039 | 0.056 → 0.049 | 0.173 → 0.136 |
| Unseen held setting | 8 / 40 | 5.08 → 4.71 | 30.94 → 21.68 | 4.37 → 5.07 | 0.051 → 0.054 | 0.107 → 0.095 | 0.308 → 0.268 |

The correction improved height on all 15 paired held observations and vertical speed on 13. The median nominal lookahead was about 0.20 s in each held setting. Timing and horizontal-velocity errors moved in different directions across the two settings. Many attempts lacked either a usable early forecast or a later crossing observable in stereo. The scores describe the paired flights. Among the seven launch IDs with both held observations, increasing damping lowered the median observed crossing height by 6.3 mm and vertical velocity by 0.118 m/s. [Paired optical labels and residuals](flight-residual.json.gz)

## Live comparison

The live comparison targets the two vertical quantities after reviewing the component-level held results. It asks the nominal twin to match those two predicted changes within a limited adjustment to its early optical state, while the requested horizontal and timing corrections stay at zero. Both branches use the same frozen motor examples, camera/encoder inputs and fast physical practice carriage. The points use the unseen damping setting. The live openings are disjoint from both the correction practice and the motor checkpoint.

| Measured opening | Nominal alternating returns | Corrected alternating returns | Early adjustments in corrected point |
| --- | ---: | ---: | ---: |
| 4527 | 4 | 4 | 2 |
| 11292 | 4 | 4 | 2 |

Both branches reached the four-return cap on both points, including two learner returns per point. The adjusted points used different joint commands. The forecast improvement did not produce a score difference within this short comparison. [Native action tapes and referee events](flight-residual.json.gz)

Run the forecast assay and then the live comparison from the repository root.

```sh
export MUJOCO_GL=egl PYTHONPATH=.:src:scripts/experimental LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt29_flight_residual /path/to/rallies.json
python -m scripts.experimental.tt29_flight_residual /path/to/rallies.json --live
```
