# Rebuilding the receiving shortlist for the camera load

The 80 g hypothesis previously selected from one 24-frame RGB movement supplied the inertia for a new candidate catalogue. The trial recomputed blade position and face orientation for all 2,048 existing command sequences. This let the inferred load affect geometric shortlisting as well as the independent MuJoCo forecasts. Both controllers used the same gantry priors, 32-candidate limit and native wrist-plus-fixed RGB. The rebuild used 40,960 model frames and zero additional target practice frames. [Earlier optical load inference](../tt19/direct-receive-payload-adapt.md) · [Rebuilt archive](blade-loaded-atlas.npz)

| Receiving starts | Frozen legal returns | Rebuilt legal returns | Frozen points won | Rebuilt points won |
| --- | ---: | ---: | ---: | ---: |
| 12 previously published loaded starts | 7/12 | 7/12 | 3/12 | 3/12 |
| Six fresh paired starts | 2/6 | 2/6 | 1/6 | 1/6 |

Each pair had the same successful starts. On the six fresh starts both branches landed three legal returns in total, including one point with two. The earlier mass-only controller also returned on seven of the 12 published starts. Compared with that controller, the rebuilt catalogue changed all 12 forecasts and selected the same action tapes. It changed one tape relative to the frozen controller on the fresh starts. Neither earlier loss recovered. [Paired receipt](direct-receive-loaded-atlas.json.gz)

Relative to the nominal catalogue, recomputed blade centers moved by a median of 0.50 mm across candidate frames and 10.21 mm at the 95th percentile. All 36 physical point tapes, including the reused frozen baseline, reproduced the event and score records. The fresh starts were disjoint from earlier receiving assays and the first TT20 blade test. [Trial and replay](../../scripts/experimental/tt20_blade_atlas.py)

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt20_blade_atlas artifacts/tt19 artifacts/tt20
```
