# MuJoCo gantry source training

The independent source body uses three Cartesian slides and two tilt hinges. The five absolute pose requests come from the unchanged 42-coefficient readout and drive finite-force damped servos. A spinning sphere and a finite-mass cylindrical paddle interact through MuJoCo contacts at 480 Hz. Gravity, drag, friction, contact damping, actuator response, and drops vary by seed. The policy receives clean metric pose and ball motion. The referee requires distinct ascending face rebounds with at least 45 mm of rise and 65 ms airborne. [Native source scene](../../scripts/experimental/tt24_gantry_source_native.cpp)

The analytic readout was the initial policy. Paired coefficient perturbations used only achieved source scores. The single-CPU pass used 4,608 source episodes and took 48 seconds. Gantry development scores selected the exported checkpoint at iteration 12. The SO-101 fixture supplied no motor labels or practice for this fit. [Training and frozen readout](../../scripts/experimental/tt24_gantry_source.py) · [Checkpoint](gantry-source.npz)

| Held body | Drops | Stationary mean | Analytic prior mean | Gantry-refined mean | Analytic prior and refined at least 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuJoCo gantry | 128 | 2.55 | 18.16 | 19.52 | 128/128 and 128/128 |
| MuJoCo gantry, wider dynamics | 128 | 2.04 | 17.77 | 19.29 | 128/128 and 128/128 |
| Original analytic worlds | 500 | 1.97 | 19.55 | 17.76 | 498/500 and 449/500 |
| Original analytic worlds, stress | 500 | 1.15 | 15.70 | 12.65 | 404/500 and 317/500 |

The table counts total verified rebounds. Mean consecutive runs on the nominal gantry increased from 18.16 to 19.04. Average rebound rise changed from 81.6 to 69.6 mm and the mean number of scored impacts with upward paddle-point speed above 55 mm/s rose from 16.27 to 17.15. Recompiling the native source reproduced the first eight held rows exactly per profile and condition. The original analytic rows use the archived fixed seed lists. Refinement improved the physical gantry and reduced the scores on the original body. [Training and paired outcomes](gantry-source.json.gz)

## Clean-state SO-101 transfer audit

Both frozen readouts used the same clean simulated observations and the local differential of the blade point carried by the jaw. The direct Jacobian of the separately welded free paddle has zero SO-101 joint columns. The audit differentiates the jaw-carried point and submits normalized commands to the original position servos.

| Stock SO-101, 16 paired drops | Mean total | Mean consecutive | Maximum consecutive |
| --- | ---: | ---: | ---: |
| Stationary | 2.56 | 2.56 | 4 |
| Analytic prior | 1.50 | 1.31 | 5 |
| Gantry-refined prior | 1.00 | 1.00 | 2 |

The gantry refinement improved the source body and reduced performance under this SO-101 grounding. The ball hit the arm, camera mount, handle or clamp to end six analytic-prior episodes and ten gantry-refined episodes. All sixteen stationary drops ended at the table. The isolated gantry disk lacks those additional collision surfaces. The transfer audit supplied clean state and the jaw differential only during evaluation. Run `python -m scripts.experimental.tt24_gantry_source` for the source comparisons and `python -m scripts.experimental.tt24_gantry_source_carry` for the paired SO-101 audit. [SO-101 outcomes and replay tapes](gantry-source-arm-stock.json.gz)
