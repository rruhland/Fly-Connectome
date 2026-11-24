# Faster native planning for the far-side carriage

The opt-in `FastPartnerArena` ran the existing far-side search **4.19× faster in planning CPU time** and **3.04× faster in total CPU time** across 16 matched releases. Both implementations produced native carriage contacts and legal far returns on all 16. Every event digest and plan report matched, including strike, separation, landing, and aimed receiving windows. [Paired benchmark](tt29_partner_fast.json.gz)

| Input cohort | Releases | Default planning CPU | Native planning CPU | Speedup | Native far legal returns |
| --- | ---: | ---: | ---: | ---: | ---: |
| First eight physical near-side departures from the coached rally | 8 | 7.74 s | 1.72 s | 4.50× | 8 / 8 |
| Rotated measured launches from partner validation | 8 | 6.71 s | 1.73 s | 3.87× | 8 / 8 |
| All | 16 | 14.45 s | 3.45 s | 4.19× | 16 / 16 |

The first group reused the measured ball position, velocity and spin at each native near-side blade separation, with the receiver reset at each release. The second group covered eight previously validated ordinary, fast, oblique, strong-spin, high and slow measurements. Each pair used identical initial states, far targets, model and random seeds. The timing alternated the execution order and used one CPU process. One-time compilation sits outside the timed release loop. Median planning CPU per call fell from 0.895 s to 0.219 s. [Benchmark implementation](../../scripts/experimental/tt29_partner_fast_bench.py)

The native kernels batch quaternion conversions and isolated MuJoCo flight stepping. The search uses the original candidates, trajectory constraints, final score and finite-mass blade dynamics. The normal `PartnerArena` is still the default. Code can select `FastPartnerArena` explicitly using the same constructor parameters. [Opt-in class](../../scripts/experimental/tt29_partner_fast.py) · [Native kernels](../../scripts/experimental/tt29_partner_fast.cpp)

With the near-side commands replayed from the coached point, the accelerated far partner independently computed all 50 far strokes in the live simulation. The replay completed all 100 alternating legal returns in 70.33 simulated seconds using 17.56 seconds of process CPU time after construction. The full event digest and all 4,220 sampled joint and ball state vectors match the earlier replay exactly. This timing measures the far partner and simulator while the near-side commands are already available. It leaves the cost of live near-side planning open. [Continuous replay](tt29_partner_fast_tape.json) · [Original coach trace](rally-physics-100.json.gz)
