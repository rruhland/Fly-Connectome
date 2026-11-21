# First-person shaded table tennis

The receiving player observes two native MuJoCo RGB cameras, selects four signed commands, and updates between points. The hall checkpoint includes the optical background, camera calibration, learned arm model, score and timing statistics, and random generator state. Use it at its original station to play a new point without repeating probes.

## Play the saved learner

From the repository root on Linux with EGL, run the following commands. The destination filenames are local outputs.

```sh
python -m pip install -e '.[table-tennis,test]'
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt5_checkpoint play artifacts/tt5/short-calibration.json.gz /tmp/point.json.gz --installation 0 --checkpoint artifacts/tt5/first-hall-player.npz --seed 833294100
python -m scripts.experimental.tt4_replay /tmp/point.json.gz /tmp/point.webp --condition resumed
```

The receiving simulator always serves toward the learner. New serve seeds change ball trajectories. Calibration belongs to a specific pair of cameras, arm placement and wiring. The receipt passed to `play` provides the simulator with the installation settings. The policy receives RGB, its own requested actions and public point rewards.

The checkpoint below carries a player that practiced through first-person cameras on both tables. It is saved after two hall points and resumes at the original installation.

```sh
python -m scripts.experimental.tt5_checkpoint play artifacts/tt5/first-person-source.json.gz /tmp/first-person-play.json.gz --checkpoint artifacts/tt5/first-person-mini-to-hall.npz --seed 833294100
```

## Reproduce the experiments

```sh
python -m scripts.experimental.tt5_hall_trial /tmp/previous.json.gz --seed 86117024 --installations 5 --points 10 --cache /tmp/tt5-probe-cache --conditions temporal:legacy_mini
python -m scripts.experimental.tt5_hall_trial /tmp/hall.json.gz --seed 86117024 --installations 5 --points 10 --cache /tmp/tt5-probe-cache --conditions rally:legacy_mini rally:pong_only
python -m scripts.experimental.tt5_hall_trial /tmp/short.json.gz --seed 86117024 --installations 5 --points 6 --probes 3 --conditions rally:legacy_mini unmapped:legacy_mini
python -m scripts.experimental.tt5_checkpoint build /tmp/short.json.gz /tmp/player.npz --after 2 --installation 0
python -m scripts.experimental.tt5_hall_trial /tmp/games.json.gz --seed 98281405 --installations 2 --points 50 --max-steps 600 --race-to 11 --conditions rally:legacy_mini
python -m scripts.experimental.tt5_hall_trial /tmp/adaptive.json.gz --seed 28498763 --installations 4 --points 6 --conditions adaptive:legacy_mini
```

The cache contains only the actions and camera images from the shared motor probes. Each group is charged for its observation budget. Scored points always run live. The fixed-budget qualification and games use the original native shading. The optional `--fast-hall` flag reduces the floor and box mesh tessellation for quicker development. This changes static lighting and retains the physical model. The [render benchmark](render-speed.json) measures the CPU cost per step and RGB difference on identical commands.

[Primary hall qualification](hall-qualification.json.gz) · [Previous player on the same starts](temporal-baseline.json.gz) · [Short calibration and erased wiring](short-calibration.json.gz) · [Restorable player](first-hall-player.npz) · [New point played after loading](resumed-new-point.json.gz)

Additional comparisons use the same runner.

```sh
python -m scripts.experimental.tt5_hall_trial /tmp/first-person.json.gz --seed 86117024 --installations 3 --points 6 --conditions rally:first_person_mini rally:fresh
python -m scripts.experimental.tt5_hall_trial /tmp/confirmation.json.gz --seed 26419047 --installations 4 --points 6 --conditions rally:first_person_mini rally:pong_only rally:fresh
python -m scripts.experimental.tt5_hall_trial /tmp/studio.json.gz --seed 98034124 --installations 3 --points 6 --venue studio --baseline .16 --conditions rally:legacy_mini
python -m scripts.experimental.tt5_hall_trial /tmp/recovery.json.gz --seed 98034124 --installations 1 --points 6 --venue studio --baseline .16 --conditions fallback:legacy_mini
python -m scripts.experimental.tt5_impact /tmp/mini-impact.json
python -m scripts.experimental.tt5_hall_trial /tmp/impact.json.gz --seed 86117024 --installations 3 --points 6 --conditions impact:legacy_mini
python -m scripts.experimental.tt5_recolor /tmp/recolor.json.gz
python -m scripts.experimental.tt5_retention artifacts/tt5/hall-qualification.json.gz /tmp/retention.json.gz
python -m scripts.experimental.tt4_replay artifacts/tt5/long-hall-games.json.gz /tmp/hall-rally.webp --condition rally:legacy_mini --course 0 --point 7
```

For a long MP4 replay, install FFmpeg with H.264 support and give the replay command an output path ending in `.mp4`.

[Research report, comparisons and replays](../../docs/reconstruction/2026-11-21-first-person-hall-learning.md)
