# Transferring a one-camera serve forecast

Each camera independently observed 12 practice opponent serves on the unmarked clamp station. The practice records store image-derived ball tracks and camera frame checksums. All 12 tracks from each camera qualified for a tabletop-coordinate position and velocity prior. On eight disjoint starts per target camera, the receiver could use the prior only with a live target ball pixel and a recent history of registered bearings.

| Target | Prior | Starts with a legal return | Points won |
|---|---|---:|---:|
| Outside | Frozen optics | 2/8 | 0/8 |
| Outside | Wrist practice | 1/8 | 0/8 |
| Outside | Outside practice | 1/8 | 1/8 |
| Outside | Wrist practice, shifted timing | 0/8 | 0/8 |
| Wrist | Frozen optics | 0/8 | 0/8 |
| Wrist | Outside practice | 1/8 | 1/8 |
| Wrist | Wrist practice | 1/8 | 0/8 |
| Wrist | Outside practice, shifted timing | 0/8 | 0/8 |

| Transfer | Forecast horizon | Same-image comparisons | Frozen optical median error | Cross-camera median error |
|---|---:|---:|---:|---:|
| Wrist to outside | 100 ms | 56 | 39.8 mm | 21.4 mm |
| Wrist to outside | 200 ms | 44 | 51.1 mm | 31.7 mm |
| Outside to wrist | 100 ms | 38 | 12.8 mm | 25.4 mm |
| Outside to wrist | 200 ms | 31 | 20.1 mm | 52.2 mm |

Wrist practice improved the outside forecast in this sample and reduced first-return starts by one. Outside practice added a wrist return and point win while increasing forecast error. The practice prior with shifted timing also improved the outside forecast, reaching 19.4 mm at 100 ms and 39.7 mm at 200 ms. The aligned transfer had smaller outside errors at the 90th percentile at both horizons. These eight-start samples give limited evidence on receiving.

The prior models the opening between bounces with known gravity and approximate drag. Practice fits table coordinates from frames 7–15 and represents variation in position and velocity at frame 12. Target updates use frames 12–19, with bounds and projection checks on the latest estimate. The forecast comparison uses the same RGB frames within each transfer run, up to the first learner contact. The outside baseline uses the temporal fit and the wrist baseline uses ball size. All conditions used the same frozen factory receiver. This evaluation covers standardized opponent-service openings on the simulated apparatus.

The physical referee counted all outcomes. Replaying all 64 held action tapes reproduced every recorded point result and event trace.

- [RGB-only source tracks and fitted priors](arc-carry-practice.json.gz)
- [Held forecasts, action tapes, outcomes and replay verification](arc-carry.json.gz)
