# TT21 receiving repertoire from archived demonstrations

An offline count memory selected a compact SO-101 repertoire from previously executed motor tapes. The archive combines a wrist-plus-two-fixed-camera planning teacher, earlier single-eye planners and size-versus-temporal flights. The eight source receipts supply 100 distinct planned strokes across 38 prior point seeds. Fifty-five earned a legal return after that stroke’s learner-side contact. Every attributed plan was checked against the actual recorded motor stream and the shared factory command catalogue. Plans with no matching live contact or legal return supplied negative credit. The source includes deliberate variation in camera and planner, so this tests a mixed teacher archive.

The memory ranks tapes by their observed legal-return counts with a neutral one-success, one-failure prior. The frozen comparison searches all 2,048 tapes, the demonstration comparison searches the best 18 according to real credit, and the shuffled comparison searches 18 according to permuted credit from precisely the same records. The two reduced repertoires share six tapes. The motion templates and dry pose catalogue already existed, so any difference concerns which motions to consider. There is no backpropagation and no new physical practice for selecting the repertoire.

Each evaluation point starts at an ordinary strict SO-101 reset with the same standard opponent serving. The student uses only its left RGB stream, declared lens calibration, own prior commands and clock. The existing flight estimate, intercept gate, physics scorer, recovery, and subsequent-return logic are shared by all treatments. Referee labels qualify the archive and score the held points. Held outcomes do not update the policy. Conditions rotate within each point seed to reduce timing order effects.

| Repertoire | Starts with paddle contact | First legal returns | Starts with two or more returns | Points won | Dry rollouts | Planner CPU |
|---|---:|---:|---:|---:|---:|---:|
| Frozen full catalogue | 11/12 | 5/12 | 0/12 | 4/12 | 988 | 14.93 s |
| Legal-return demonstrations | 8/12 | 3/12 | 0/12 | 3/12 | 496 | 7.58 s |
| Shuffled return credit | 11/12 | 4/12 | 0/12 | 3/12 | 382 | 6.17 s |

The demonstrated subset cut planner CPU time by 49% and dry rollouts by 50% relative to the full catalogue. It lost two of the full catalogue’s first returns and gained none (two-sided paired exact *p* = .50). The taught and shuffled subsets disagreed on three starts. Teaching returned on one that shuffle missed, and shuffle returned on two that teaching missed (paired exact *p* = 1.00). Both won three points. No condition produced a second legal learner return. The pilot demonstrates a search-cost reduction from repertoire compression. Its held results provide no evidence that actual demonstration credit made the smaller repertoire more effective than shuffled credit.

Reading the eight saved receipts and building the count memory took 0.08 process CPU seconds. That time excludes producing the original demonstrations, live teacher interaction and the existing dry catalogue. Every scored action tape reproduced its complete referee events and outcome in a fresh physics instance, 36 replays in total.

`planner CPU` includes the geometric filter and simulated-rollout scoring in each triggered search. It excludes rendering, camera calibration and temporal tracking. A reduction there need not translate into a comparable change in end-to-end action latency. The scope is standard-opponent receiving points on this fitted clamp and lens.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt21_demonstration_trial
```

[Held motion and teacher receipt](demonstration-held.json.gz) · [Memory](../../scripts/experimental/tt21_demonstration.py) · [Reproduction script](../../scripts/experimental/tt21_demonstration_trial.py)
