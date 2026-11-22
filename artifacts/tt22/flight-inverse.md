# Estimating an incoming flight by replaying its optics

A separate ball-and-table MuJoCo model fits initial translation, velocity and spin to up to 21 past stereo observations. A finite-difference least-squares update regularizes weakly visible spin directions. The estimator sees no live ball, joint or contact state. Its force estimate approximates drag from the observed arc, and its table comes from the nominal fixture. In this arena the numerical fit is causal but begins anew on each window. [Estimator](../../scripts/experimental/tt22_flight_inverse.py)

| Eligible archived phase | Accepted fits | Median 3D velocity error, original → fit | 90th percentile, original → fit |
|---|---:|---:|---:|
| Before near bounce | 34 | 0.086 → 0.010 m/s | 0.121 → 0.019 m/s |
| First two frames after bounce | 19 | 0.676 → 0.055 m/s | 4.519 → 0.211 m/s |
| Later after bounce | 14 | 0.055 → 0.039 m/s | 0.160 → 0.079 m/s |

Only this archived diagnostic used actual velocity. In the approach where the near robot chose physical strokes from a factory model and played its stock four-axis partner, its 12 full held points lasted **4, 14, 2, 4, 6, 4, 4, 4, 4, 9, 10, 4**. It reached ten twice (one for the original), and both receivers landed 35 legal near returns. These are the same twelve serves as the original benchmark and a new evaluation of the optics. All twelve new motor and referee traces replayed exactly. A separate near receiver against the original long-cycle rival gained one legal return and one point in four matched starts. [Archived fit](flight-inverse-audit.json.gz) · [Twelve native points](flight-inverse-held.json.gz) · [Alternate player](tt22_cycle_inverse.md)

A closed-loop extension retimed a committed joint swing and moved its blade path based on the fresh fit, keeping the original chosen stroke style. It reached **0, 2, 6, 0, 0** against the original's **2, 4, 0, 4, 6** on five prior development starts. The correction was applied on each point. The original retained more total returns. [Trim pilot](trim-pilot.json.gz)
