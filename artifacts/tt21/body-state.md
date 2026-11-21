# Does a moving-arm estimate improve the next reply?

The learner’s requested joint-rate sums had returned to zero on all six prefixes. When the rival began its return, the blade was still moving at a median **266 mm/s**. A command-driven copy of the arm predicted blade pose substantially better than the HOME initialization. Both initializations chose the same stroke or abstained at every matched decision. An earlier decision produced one extra contact, and both timings made the same one legal next return.

## Matched second incoming balls

Six archived TT20 points supplied complete native opening prefixes, four receiving a serve and two serving. Each prefix ended at the standard opponent’s first contact after a successful own shot. The strict moved-clamp fixture reproduced all six rival contacts exactly. Each neutral continuation confirmed the subsequent legal opponent bounce. Two fixed native RGB streams with natural table markings supplied the same 30 Hz ball observations to both planners. Ball coordinates are in metres, with height zero at the tabletop and negative y toward the learner.

The comparison varies the body initialization and the timing independently. HOME uses the existing settled nominal physics state and actuator targets. The command twin starts at its own factory initialization and advances a second nominal MuJoCo state from its own requested actions, applying servo limits at physics-substep resolution. Its own ball stays parked. Only the dry candidate solver gets the new body initialization. Both ranks use the same RGB ball estimates, HOME-based analytic shortlist, candidate tapes, durations, scoring and perturbations. Target joints and contacts are read by the evaluator to measure error and validate the continuations. [Command twin](../../scripts/experimental/tt21_body_state.py) · [Paired experiment](../../scripts/experimental/tt21_body_state_trial.py)

## Body estimate

Medians over six cases per row. Blade speed error is the magnitude of the three-dimensional velocity residual.

| Snapshot | Actual blade speed mm/s | Center error mm HOME / twin | Velocity error mm/s HOME / twin | Normal error ° HOME / twin |
|---|---:|---:|---:|---:|
| Rival contact | 266.24 | 4.80 / 0.21 | 267.87 / 5.40 | 0.96 / 0.07 |
| Early optical gate | 20.86 | 1.90 / 0.029 | 22.54 / 6.08 | 0.92 / 0.014 |
| Late optical gate | 3.29 | 0.56 / 0.018 | 3.44 / 0.23 | 0.31 / 0.011 |

The largest observed blade speed at rival contact was 783 mm/s. On one prefix, actuator clipping left the actual target 0.626° RMS from HOME even after the requested rates summed to zero. The twin matched the clipped actuator targets on all six prefixes.

## Native scoring

The early gate used the first incoming optical estimate below y = +0.15 m. The late gate used the first below y = −0.24 m. Both demanded y > −0.81 m and measured incoming speed above 0.93 m/s. On these cases, the early gate triggered 3–4 frames earlier, or 100–133 ms. Each condition got one decision opportunity and applied the stock confidence filter. The four conditions continued the same strict physical point independently.

| Decision | Initialization | Applied | Contacts | Legal next returns | Points won |
|---|---|---:|---:|---:|---:|
| Early | HOME | 2/6 | 2/6 | 1/6 | 1/6 |
| Early | Command twin | 2/6 | 2/6 | 1/6 | 1/6 |
| Late | HOME | 1/6 | 1/6 | 1/6 | 1/6 |
| Late | Command twin | 1/6 | 1/6 | 1/6 | 1/6 |

The two body initializations produced identical live action tapes and scoring events for each of the twelve paired decisions. Every one of the 24 continuations reproduced its result and event stream on replay. At the early gate, the unchanged shortlist was empty on two cases and the confidence filter skipped two additional cases. The one-shot comparison isolates body initialization within the existing shortlist. It leaves candidate generation, repeated replanning and dynamic readiness for separate experiments. This branch stayed at the paired-suffix stage after the action tapes tied. [Machine receipt](body-state.json.gz)

Run from the repository root with the MuJoCo environment enabled.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt21_body_state_trial
```
