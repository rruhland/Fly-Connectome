# Table-tennis transfer experiments

The mini table and receiving table can both be played from a fixed stereo rig at the player's end. A second receiving-table camera rig looks down and across the table. The receiving table uses MuJoCo with a separate inlet for actions and an evaluator that records contacts, bounces, legal landings and ordinary point rewards.

## Reproduce

From the repository root, install the table simulation dependencies and set the import path for the earlier experimental scripts.

```sh
python -m pip install -e '.[table-tennis,test]'
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

EGL runs the shaded cameras headlessly on Linux. The projected RGB camera and physics-only benchmark can run without a graphics context. To reproduce the original mini visual component backend, optionally set `TT3_COMPONENT_LIBRARY` to the compiled component library described in the [mini experiment instructions](../tt3/README.md).

### Mini-simulator play

```sh
python -m scripts.experimental.tt4_mini_pov /tmp/mini-course.json.gz /tmp/mini-player.json.gz --matches 2 --points 20 --workers 2
python -m scripts.experimental.tt4_mini_replay artifacts/tt4/mini-first-person-sustained-replay.json.gz /tmp/mini-rally.webp
```

[Player checkpoint](mini-first-person-player.json.gz) · [Matched action hashes and outcomes](mini-first-person-course.json.gz) · [Wider rig](mini-first-person-wide-course.json.gz) · [Post-bounce meeting rule](mini-first-person-rebound-course.json.gz)

### Flight and contact calibration

```sh
git clone https://github.com/cogsys-tuebingen/spindoe.git /tmp/tt4-spindoe
git -C /tmp/tt4-spindoe checkout ebc8a9eac32e42cdbe2978767ef0134d67dfc123
python -m scripts.experimental.tt4_flight_validation /tmp/tt4-spindoe/data/trajectories/trajectories.zip --engine --output /tmp/measured-flight.json
```

The fit takes samples in the first 50 ms of an airborne track and predicts the sample at or immediately after 150 ms. A deterministic hash of the shot ID splits complete shots into 97 for fitting and 37 for evaluation. Each dynamics model estimates initial velocity from the same prefix. Measured spin is supplied to the physics audit only. The separate 155-bounce reference uses short local before/after vertical velocity fits, rejecting off-table and ambiguous impacts. [Flight measurements and source revision](measured-flight.json) · [Serve, bounce, stationary, random and scripted-control checks](arena-audit.json)

### Receiving-table play

All arms of a comparison use the same four-axis hardware, signed wiring, seed sequence, calibration frames, vision pipeline and score-based updates. Run the small native comparison with the following command.

```sh
python -m scripts.experimental.tt4_transfer_trial /tmp/studio.json.gz --seed 912047311 --installations 3 --points 4 --probes 8 --probe-steps 34 --views first_person --mode mujoco --width 192 --height 144 --baseline .28 --conditions first_person_mini pong_only fresh legacy_mini --first-person-mini artifacts/tt4/mini-first-person-player.json.gz
```

Pass `--venue hall` for the hardwood hall, barriers, extra lighting and white/orange balls with red/black rubber faces. Each receipt stores the exact geometry, policy inputs, per-installation wiring and all executed action tapes. The private evaluator counts a return only after the ball legally lands on the opponent's side. Its information is excluded from the policy interface. The calibration is charged separately and shared across each matched set of policies.

The principal matched suite used `--seed 1472928681 --installations 6 --points 12 --probes 10 --probe-steps 36 --width 96 --height 72 --first-person-mini artifacts/tt4/mini-first-person-player.json.gz`. Its default views and conditions test both optics and a timing ablation. Replay an issued action sequence through the same camera and physics with the command below.

```sh
python -m scripts.experimental.tt4_replay artifacts/tt4/tt4-paired-projected-transfer.json.gz /tmp/projected.webp --condition legacy_mini --course 5 --point 7
python -m scripts.experimental.tt4_transfer_source_trial artifacts/tt4/tt4-paired-projected-transfer.json.gz artifacts/tt4/mini-first-person-wide-player.json.gz /tmp/wide-study.json.gz
python -m scripts.experimental.tt4_forecast_probe /tmp/forecasts.json --source-points 24 --test-points 16 --test-seed 921754
```

The forecast diagnostic fits from mini RGB tracks. The ground-truth ball position is used only for identifying and scoring the tracked ball at the receiving table. [Matched projected-RGB games](tt4-paired-projected-transfer.json.gz) · [Wide-donor study](tt4-paired-wide-source-transfer.json.gz) · [Camera precision diagnostic](camera-precision.json) · [Optical forecast results](optical-forecast.json)

```sh
python -m scripts.experimental.tt4_arena_audit --points 40 --output /tmp/arena-audit.json
python -m scripts.experimental.tt4_replay artifacts/tt4/native-hall-transfer.json.gz /tmp/hall-rally.webp --condition first_person_mini --course 1 --point 3
```

The larger shaded-hall check uses `--seed 314592612 --installations 5 --points 8 --probes 8 --probe-steps 34 --views first_person --mode mujoco --venue hall --width 192 --height 144 --baseline .28 --conditions legacy_mini first_person_mini pong_only --first-person-mini artifacts/tt4/mini-first-person-player.json.gz`.

For the native temporal-vision comparison, the player takes every second frame of its existing randomized motor probes and estimates the most frequent RGB value at each pixel. This provides a scene background for tracking moving regions. It adds no observation frames, but uses a fixed camera during a point.

```sh
python -m scripts.experimental.tt4_native_temporal_trial /tmp/temporal-study.json.gz --first-person-mini artifacts/tt4/mini-first-person-player.json.gz
python -m scripts.experimental.tt4_replay artifacts/tt4/native-temporal-transfer.json.gz /tmp/studio-rally.webp --condition legacy_mini --course 1 --point 1
```

The summary figure can be regenerated with Matplotlib installed by running `python -m scripts.experimental.tt4_figure artifacts/tt4 /tmp/returns.png`.
