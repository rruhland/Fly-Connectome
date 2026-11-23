# High-arc recovery under slow source actuation

A warm start from `world-prior.npz` was trained in randomized three-dimensional sphere–disk worlds using the same 42-coefficient feedback readout. The readout receives the ball position and velocity, disk position and upward normal from the simulated source. We kept its canonical five-channel command format and bounds. No scheduling, contact phase, trajectory, or disturbance announcement enters the readout. This experiment used no SO101 rollouts or target rewards.

The source body has compliant velocity tracking, one to five frames of random command delay at 60 Hz, cross-axis coupling of 0.20–0.40 with random sign and heading, and a vertical speed limit of 0.22–0.53 m/s. The disturbed training worlds introduce unsignaled horizontal velocity impulses of 0.105–0.22 m/s in random directions. The stronger qualification worlds extend the delay to three to seven frames, add compliance and lower restitution, couple axes by 0.32–0.54, permit vertical speed limits down to 0.16 m/s, and increase impulses to 0.17–0.33 m/s.

A physical return requires a descending sphere–disk collision followed by upward launch, at least 65 ms to a measured apex at least 45 mm above contact and at least 18 mm clear of the disk. The training objective values survival, eligible arcs above a random 75–115 mm threshold, and returns from new contacts after impulses. It penalizes RMS speed and acceleration of requested positions and normals and squared offset from home. Arc credit is clipped above the threshold plus 110 mm. The verification threshold stays at 45 mm in all treatments.

## Frozen source qualification

Each group uses 500 fresh worlds paired between policies. Survival is the fraction reaching the full 5 s horizon. Arc rise averages over verified returns. Slew clamp is the fraction of executed frames where any actuator channel would have exceeded its limit before regulation.

| Worlds | Policy | Returns per drop | Arc rise | Survival | Recovery per impulse | Slew clamp |
|---|---|---:|---:|---:|---:|---:|
| Compliant | Canonical | 17.286 | 8.9 cm | 94.4% | — | 28.6% |
| Compliant | High arc | 11.384 | 21.3 cm | 96.6% | — | 43.4% |
| Compliant | Idle | 2.126 | 11.1 cm | 0.0% | — | 0.0% |
| Disturbed | Canonical | 14.846 | 9.0 cm | 69.8% | 93.2% | 27.2% |
| Disturbed | High arc | 10.938 | 21.5 cm | 91.2% | 89.7% | 42.7% |
| Disturbed | Idle | 2.132 | 11.3 cm | 0.0% | 19.8% | 0.0% |
| Stronger disturbance | Canonical | 4.440 | 10.0 cm | 1.2% | 55.1% | 22.3% |
| Stronger disturbance | High arc | 5.452 | 16.2 cm | 15.6% | 64.3% | 39.7% |
| Stronger disturbance | Idle | 1.572 | 10.6 cm | 0.0% | 8.8% | <0.1% |

The paired rebound difference under stronger disturbances is +1.012, with a 95% paired bootstrap interval of +0.686 to +1.358. On ordinary disturbed worlds the higher-arc policy recovered 3.046 times per drop from 3.394 impulses, compared with 2.802 from 3.008 for canonical. Longer survival exposed the higher-arc policy to additional impulses.

| Worlds | Requested translation speed, m/s | Requested translation acceleration, m/s² | Requested normal speed, 1/s | Requested rate above limits |
|---|---:|---:|---:|---:|
| Compliant | .837 → .781 | 70.38 → 64.13 | .528 → .672 | 22.0% → 18.9% |
| Disturbed | .825 → .781 | 69.23 → 63.77 | .640 → .786 | 20.8% → 19.0% |
| Stronger disturbance | .770 → .792 | 64.52 → 65.09 | .656 → .814 | 23.8% → 32.3% |

The time-domain bandwidth measures are mean per-drop RMS first and second command differences. The normal columns use the x/y components of the commanded unit normal. The policy trades repeated shallow arcs for greater flight height and longer survival. It increases tilt demand and executed clipping in all three groups. These results establish taller, more disturbance-tolerant source play with mixed changes in actuator demand.

## Reproduce

Run `OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt24_world_margin` from the repository. It builds the C++ simulator with OpenMP. The warm-start search ran 176 iterations with 38 antithetic directions and 39 new worlds per iteration, comprising 521,664 simulations and 149,831,716 frames. Checkpoint selection used 6,624 simulations on fixed development seeds and selected iteration 128. Qualification used 4,500 simulations across the three policies and three fresh cohorts. The canonical source exposure is recorded in `world-prior-source.json` and precedes this run. `world-margin-source.json` stores raw counts, per-episode metrics, seeds, and hashes. `WorldPrior.load(Path("artifacts/tt24/world-margin.npz"))` loads the frozen policy.
