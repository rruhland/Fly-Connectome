# Local racket effects from live 3D table tennis

A model fitted to 3D table-tennis contacts predicted the normal component of a new bounce's exit velocity. Using that prediction in a bounded reactive controller left its score tied with the shuffled-source controller and below the existing active wrist anchor. All arm comparisons used fresh stock drops. [Clean-state physical control](spatial-carry-clean-control.json.gz)

## Source and geometric assumptions

The source consisted of thirteen previous native SO-101 points with live near-arm commands from release and at least two legal near returns. The points came from the combined S/T and interaction K/L assessment courses. Their exact two-arm command tapes reproduced 52 legal near returns, all referee events and the terminal score. The collector used simulator contact tags and legal-return labels. Simulator geometry supplied the actual racket normal, contact location, racket linear and angular velocity, incoming ball velocity and exit velocity at separation. These labels define a privileged source capture and destination diagnostic.

The fit expresses velocities in the actual racket frame and subtracts the moving contact point's velocity. It chooses the blade face toward the incoming ball, retains face contacts within a fixed normal-alignment tolerance, and scales the contact radius by the known blade radius. The tangent axes are the blade's local geometry axes. The full residual learns the difference between ideal reflection and observed exit velocity. The reduced model estimates the ratio of outgoing normal speed to closing speed from log closing speed, tangential speed divided by closure, and fractional contact radius. It discards tangent direction, absolute paddle motion and ball spin. The no-source control uses unit normal restitution. The shuffled controls permute target residuals or ratios within the same training data and fit. The source-constant control retains only the mean measured restitution. Eight source launches supplied 33 legal face contacts for fitting. Five launches supplied 19 held contacts. [Native tapes and measured effects](spatial-carry-table.json.gz)

## Fresh bounce prediction

Twelve stock drops were played with zero drive and with a fixed wrist controller on identical seeds. The active anchor inherited 266 earlier target attempts. Both controllers acted independently of the new model, and all 24 command tapes replayed. Zero new bounce trials went into either contact fit.

| Contact course | Measured contacts | Median closure, m/s | Median tangential speed / closure | Full feature coverage |
|---|---:|---:|---:|---:|
| Held table replies | 19 | 2.98 | 0.550 | 63% |
| Passive paddle | 49 | 1.45 | 0.061 | 0% |
| Active wrist anchor | 39 | 1.79 | 0.039 | 0% |

Coverage means every feature lies within the middle 90% of its source training values. The median outward racket motion at contact was 1.02 m/s in the held table replies, effectively zero for passive bouncing and 0.05 m/s for the active wrist anchor. Expressing the interaction in local coordinates preserved these physical differences. Three-component coherent fit errors on passive and active bounce contacts were 0.643 and 0.702 m/s, while ideal reflection scored 0.174 and 0.212 and shuffled scored 0.163 and 0.141.

The reduced model still extrapolated at all 88 bounce contacts. Its median input slip ratios were far below the table value. Even there, it predicted the observed normal exits substantially better than all three controls. Its held native performance left the state-dependent effect uncertain because the shuffled and constant fits had smaller errors on those five source launches.

| Normal exit RMSE, m/s | Coherent | Shuffled | Constant source | Unit restitution |
|---|---:|---:|---:|---:|
| Held table replies | 0.180 | 0.159 | 0.162 | 0.266 |
| Passive paddle | 0.075 | 0.197 | 0.144 | 0.285 |
| Active wrist anchor | 0.101 | 0.248 | 0.189 | 0.354 |

[Paired measured positions, velocities, and scores](spatial-carry-contact.json.gz)

## Fresh clean-state control

The independent control course used the existing live wrist anchor for pose stabilization and lateral control. The added rule saw clean position and velocity and used the current simulator Jacobian to convert a desired normal-speed change into bounded joint-rate residuals. It assumed a short constant-relative-velocity projection to check the blade's contact radius, a fixed ball-to-face offset and a desired outward speed of 1.55 m/s. It approximated contact-point velocity by current blade-center translation. The analytic gates and scalar speed goal were shared across the four reactive controllers. Each frame recomputed the residual from its current observations. Source tape replay was confined to data collection.

| Held treatment | Mean consecutive rebounds | Best | Frames with a changed command |
|---|---:|---:|---:|
| Coherent local fit | 1.75 | 3 | 112 |
| Shuffled local fit | 1.75 | 2 | 113 |
| Constant source | 1.67 | 2 | 115 |
| No source, unit restitution | 1.67 | 3 | 111 |
| Existing active wrist anchor | 3.00 | 6 | 0 |
| Passive paddle | 2.50 | 4 | 0 |

Coherent control trailed the active anchor on eight of twelve drops and tied on four. The mean difference was -1.25 rebounds, with a paired bootstrap 95% interval of -1.92 to -0.67. The comparison with shuffled had a mean difference of zero and an interval of -0.33 to 0.33. All 72 tapes reproduced the physical scorer. The physical gate for a separate wrist-decoder test remained unmet. The measured result concerns one-step normal-speed correction with the other control terms held common. [Fresh paired clean-state trajectory tapes](spatial-carry-clean-control.json.gz)

## Reproduction

Use the table-tennis environment from the repository root. Set `PYTHONPATH=src`, `MUJOCO_GL=egl`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

```sh
python -m scripts.experimental.tt24_spatial_carry
python -m scripts.experimental.tt24_spatial_carry_trial
```
