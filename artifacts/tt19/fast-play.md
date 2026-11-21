# Budgeted native SO-101 receiver search

Both receivers lost the fresh strict game 6–11 with six legal returns and seven legal serves in eight attempts. The adaptive search increased planning CPU time from 21.45 to 23.78 seconds. It fell back on 25 of its 28 calls that reached CAD simulation. [Timing and score summary](fast-play-1221432110-summary.json.gz)

| Receiver | CAD simulations | Planning CPU | Median CPU per call | p95 CPU per call | Whole game CPU / wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full 32-candidate search | 1,112 | 21.45 s | 71 ms | 786 ms | 77.22 / 213.81 s |
| Eight candidates with cached fallback | 1,141 | 23.78 s | 69 ms | 976 ms | 76.60 / 137.59 s |

There were 65 planning calls in each game, including calls with no eligible stroke. The cache reused 313 results. Fallback added CAD work outside the cached overlap and repeated the analytic shortlist ranking. Frame counts differed by three and the sequential wall timings reflect different host contention. The two branches matched exactly on 14 of 17 points. All 34 point tapes passed physics replay. [Full game and timing tape](fast-play-1221432110-full.json.gz) · [Adaptive game and timing tape](fast-play-1221432110-adaptive.json.gz)

The six fixed optical queries produced the same chosen action tapes or abstentions. Four had no CAD candidates. The accepted short search took 330 ms of CPU time versus 694 ms for the full search. The query that fell back took 968 ms versus 791 ms. The accepted case still exceeded one 33.3 ms video frame. [Paired query timings](fast-play-1221432110-summary.json.gz)

The adaptive receiver starts with eight candidates from the existing analytic shortlist. It accepts a stroke when the nominal simulation predicts a legal bounce and all three scenario costs meet the inherited threshold. The three scenarios use the current optical estimate and two perturbations. Otherwise it runs the full 32-candidate policy. It reuses exact repeats of a command tape, duration and perturbation within that one decision. The opening server, full CAD rollout dynamics and native wrist-plus-fixed cameras come from the deployed full-spread player. [Receiver](../../scripts/experimental/tt19_fast_play.py) · [Original rollout receiver](../../scripts/experimental/tt19_model_rollout.py)

The matched games use strict compact-table scoring and the same opening seed stream. No new target interactions train either search. Both draw on the persisted local stroke bank, which spent 36,607 frames on 1,000 contextual search points and 45,734 frames on 1,008 qualifications. Held point seeds are disjoint from both phases. Both receive the same 2,048 factory-generated movement tapes. [Shared stroke bank](consolidated-local-bank.json.gz) · [Factory catalogue build](catalog-build.json.gz)

## Reproduction

Run within the table-tennis simulation environment. The trial plays the two policies sequentially and times receiving search calls with both process CPU time and wall time. It compares the first six baseline query contexts with the same optical input for both policies, then checks that the saved action tapes reproduce each completed game.

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt19_fast_play_trial \
  artifacts/tt19 artifacts/tt19/factory-dry-catalog.npz --seed 1221432110
```
