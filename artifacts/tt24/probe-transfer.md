# Learning which motor probe to issue from native Pong

This experiment transfers a probe selector from native Pong to identification of a remapped five-joint SO-101 in MuJoCo. The selector chooses between four temporal envelopes. Every envelope uses 12 active samples of amplitude 0.45 with equal positive and negative counts, followed by eight zero samples. Their order differs. The candidate banks, total command energy, sampling horizon and ridge identification estimator are shared. The selector's inputs are dimensionless summaries of the latest observed response. Its first probe is the same square pulse used by the generic controls.

## Genuine native optical source

The live native Pong engine ran 132 one-second point fragments. The experimental interface changed each context's command sign and delay, varied motor gain and lag across three physics profiles, and rotated or reflected the rendered camera. Each of 33 contexts ran the same first probe, then one branch per candidate and a later common query. Optical branching identified the responsive elongated component. The source fitter learns to choose a candidate from the previous optical response's residual, late motion, latency and retained displacement. Its label is the relative reduction in subsequent optical prediction error when the candidate observation is added to the estimator. This conditional label can also reflect motor state carried into that subsequent query.

The source-trained selector averaged 0.489 normalized prediction gain on 12 disjoint raw optical contexts. The fixed square probe averaged 0.440 and the shuffled-label selector averaged 0.406. Shuffling permuted the association between a candidate and its measured gain within each context. The source fit used 21 contexts and no Pong score labels. [Native traces and selector coefficients](probe-transfer.json.gz)

## Physical motor identification

Twenty new SO-101 trials each use unseen dense mixing of all five issued motor channels, one wrist payload scale from 1.55 to 3.8, a second link mass change and a zero-to-three-frame interface delay. The estimator sees only public motor commands and joint displacements at 30 Hz. The true wiring, masses and delay are held by the simulated plant. The first five orthogonal directions in each repeated set come from a fresh QR decomposition and carry no motor-axis information.

The native and shuffled selectors share exactly the same directions as the generic orthogonal arm. The isotropic arm uses independent normalized Gaussian directions and the fixed square probe. Each treatment has eight sequential calibration probes. The budgets of two and four are prefixes. Each fitted model predicts encoder trajectories for 16 new query pulses that begin from the same settled posture. The query results do not update the estimator or choose a probe. No ball score enters this experiment.

| Calibration probes | Pong | Shuffled | Generic orthogonal | Isotropic |
|---:|---:|---:|---:|---:|
| 2 | 45.04 | 44.96 | 44.96 | 46.36 |
| 4 | 32.80 | 31.86 | 31.86 | 36.73 |
| 8 | 23.82 | 22.96 | 22.96 | 24.18 |

Values are held encoder trajectory RMSE in milliradians, averaged over 20 matched plants. At four probes, the Pong selector's error was 0.94 mrad higher than generic orthogonal, with a paired 95% bootstrap interval of 0.07 to 2.00. Its error was 3.94 mrad lower than isotropic, while the same orthogonal direction schedule with the square envelope did better still. The learned selector used a second envelope in 42 of 160 target probes. The shuffled selector chose the square envelope throughout the target qualification.

The game gate required the upper bound of the paired error difference at four or eight probes to show at least a 5% gain over both the generic and shuffled selectors. The main cohort did not qualify.

### Matched sampling period

A sensitivity check cut the arm control period to 1/60 s to match the source camera period. It reused the frozen Pong and shuffled selectors and ran 12 disjoint remaps. It added zero native examples.

| Calibration probes | Pong | Shuffled | Generic orthogonal | Isotropic |
|---:|---:|---:|---:|---:|
| 2 | 29.13 | 28.14 | 28.14 | 29.68 |
| 4 | 27.22 | 24.98 | 23.23 | 28.38 |
| 8 | 18.95 | 18.89 | 18.80 | 23.48 |

At four probes, the Pong selector's mean error was 4.00 mrad higher than generic orthogonal, with a paired 95% bootstrap interval of 1.00 to 7.48. At eight probes the gap was 0.15 mrad, with a paired interval of −1.05 to 1.11. Neither cohort reached the wrist qualification gate. [Matched-period receipt](probe-transfer-matched.json.gz)

## Reproduce

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -m scripts.experimental.tt24_probe_transfer
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python -m scripts.experimental.tt24_probe_transfer \
  --matched-rate-source artifacts/tt24/probe-transfer.json.gz
```
