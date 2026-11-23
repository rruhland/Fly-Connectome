# Visible workspace selection under combined physics changes

The same carried wrist bundle is assessed on a fresh mixed course of 12 stock and 12 altered starts. The experiment compares its original familiar equilibrium with two once-per-drop horizontal adjustments that preserve the recalled body's familiar paddle height and face normal. The readout and both earlier optical calibrations are frozen. The first wrist camera image selects the calibration. [Portable bundle](carried-wrist.npz) · [Previous pose and workspace study](tool-equilibrium.md)

On the altered arm, the scene-based shift reached ten rebounds on 8/12 starts, compared with 4/12 for the fixed shift and 3/12 for the unchanged bundle. Its mean longest streak was 13.83, compared with 9.75 and 4.75. The fixed shift also rescued several starts. The scene-based shift won six, tied two and lost four in that paired comparison.

The two workspace conditions use the same first three consistent wrist ball sightings, within the first ten ordinary preparation frames. Their incoming commands and observations match through the choice tick. The scene-based condition uses the measured horizontal position to choose a reference shift. The fixed condition applies the unweighted mean shift from earlier starts, with separate means for the two recalled optical models. The fixed means use all eight stock and 12 altered prior available sightings. They use no rebound scores. The selected shifts were **(+20.37, −7.33) mm** stock and **(+8.51, +6.35) mm** altered. Both conditions use the same bounded pose solve and then the same live all-joint readout. [Stock development cases](tool-equilibrium.json.gz) · [Altered development cases](tool-equilibrium-local-workspace.json.gz)

The three branches at every held start share the same release from a stock dispenser, independently drawn wide arm angles, and 14–38 charged vision preparation ticks. Ball mass, inertia and drag differ across releases. Gravity switches twice at matched, hidden pre-drawn times, about 3–3.7 and 6–6.9 s after release. These ranges, the 15 s cap, and both body-specific gain and force regimes match the earlier compound stress study. Wrist frames arrive in their ordinary order and timing. The study adds zero target training rewards, motor practice or calibration probes. Only the evaluator observes the ball physics and gravity schedule. [Compound stress protocol](integrated-stress.md) · [New held records](workspace-stress.json.gz)

## Held results

| Body and reference | Mean longest streak | ≥10 | ≥20 | ≥30 | Reached 15 s | Median observed time to loss or cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Altered, unchanged bundle | 4.75 | 3/12 | 1/12 | 0/12 | 2/12 | 0.82 s |
| Altered, scene-based shift | 13.83 | 8/12 | 4/12 | 1/12 | 4/12 | 5.66 s |
| Altered, fixed mean shift | 9.75 | 4/12 | 3/12 | 1/12 | 2/12 | 1.50 s |
| Stock, unchanged bundle | 20.00 | 5/12 | 5/12 | 5/12 | 5/12 | 2.05 s |
| Stock, scene-based shift | 27.75 | 7/12 | 7/12 | 7/12 | 7/12 | 15.00 s |
| Stock, fixed mean shift | 25.92 | 6/12 | 6/12 | 6/12 | 6/12 | 8.69 s |

The scene-based branch and unchanged bundle each scored on the first paddle contact in all 12 altered starts. The fixed branch did on ten. The difference between scene-based and unchanged appeared in sustained play. Five scene-based altered starts produced legal rebounds after each gravity switch, compared with four fixed and two unchanged. Across both bodies the scene-based, fixed and unchanged branches produced qualifying rebounds after both changes in 12, ten and seven starts respectively. The scene-based branch won nine paired releases overall against the fixed shift, tied seven and lost eight.

Every start selected the matching optical model. Both workspace branches accepted the same sightings in 21/24 paired starts, including all 12 altered starts. All 72 physical replays matched exact final ticks, referee events and recorded ball states at the reached gravity switches. Execution resumed after ten checkpointed starts, preserving their case records exactly and retaining the original 24-start schedule. [Commands, timings, and referee data](workspace-stress.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_workspace_stress --output /tmp/workspace-stress.json.gz
```

Add `--resume` with the same output path to continue an interrupted run after the last fully verified start. The runner checks the retained schedule, source data, fixed shifts and bundle before appending cases.
