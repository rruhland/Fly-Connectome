# Frozen sustained-return validation — 2026-10-29

The fixed 128-game evaluation passed for both independently acquired endpoints.
This establishes the tested sustained-control scaffold, not continuous online
learning or completion of M1B. Policies were frozen throughout evaluation.

| Endpoint | 60-second returns | Later returns, 60-second | Full-minute games | 20-second returns |
|---|---:|---:|---:|---:|
| Acquisition 0 | 543/543 | 511/511 | 32/32 | 180/180 |
| Acquisition 1 | 532/533 | 500/501 | 31/32 | 181/181 |

Combined: **1,436/1,437 returns (99.93%)**, including 128/128 first returns.
The 64 long games produced **1,075/1,076 returns (99.91%)**. Long-game time
bins were 367/367 at 0–20 seconds, 354/355 at 20–40 seconds, and 354/354 at
40–60 seconds. The last bin has exposure from 63 games: the missed rally ended
at approximately 34.975 seconds. It is not counted as surviving late play.
Opponent returns were perfect in both duration strata; opponent misses did not
make these rallies easier by ending them early. Maximum consecutive returns: 18.

Both endpoints exceeded the predeclared 95% first/later-return targets and each
had over 200 later opportunities in long games. Both independently acquired
models have identical bytes; separate training cohorts selected the same structure,
and evaluation used separate fixed starts. This tests reproducible acquisition
and execution of that controller, not diversity of learned solutions.

Runtime: 682.674 seconds (11m23s), four workers; 254,749 transition observations
and 128 initial observations. Every transition passed streaming comparisons of
independently deserialized controllers on the same public visual input. This is
not full archived replay; earlier qualification supplied full replay evidence.
No fitting, model selection, outcome-dependent extension, or baseline rerun.
Source and asset hashes remained frozen. No backpropagation or privileged policy
input was introduced. Simulator-derived opportunity labels are reporting only.

## Proven scope and next target

Reward-based search can acquire this strong relational feedback controller over
inherited frozen vision; existing visual association memory supports reliable
returns throughout the tested one-minute rallies. The search selects from a
small structural family; it is not general learning from raw pixels.

The next substantive target is continuous online acquisition and retention:
policy updates from an ongoing stream of games, carrying learned state across
resets, with progress measured during acquisition. More frozen-performance
polishing does not address that gap. No further experiment was started here.

## Reproduction artifacts

- Script: `scripts/experimental/run_relational_replication_validation.py`
- Artifact directory: `/workspace/m1b-experiments/pong-relational-replication-validation-20261029-r1`
- Spec SHA256: `3e4c609745033ead59d9ad5bae4690617ef9361b35bfa40e8ca74543cf10d60d`
- Model SHA256: `75530410f2c985a3037e9aa264447b8d172e5e9297a9e556bbb74d2c511dc7cc`
- Model: `/workspace/m1b-experiments/pong-relational-categorical-replication-20261029-r1/models/structure_04.npz`
- Full aggregate results: accompanying `2026-10-29-sustained-return-validation-summary.json`.
