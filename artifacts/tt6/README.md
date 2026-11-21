# Changing arms, racket response and actuator timing

[Research results](../../docs/reconstruction/2026-11-21-continual-arm-adaptation.md) · [Four real simulator geometries](../../docs/reconstruction/assets/2026-11-21-four-arms.png) · [Watch a two-return point during a mid-rally delay change](combined-rally.mp4) · [Watch the five-axis arm exchange six returns with its opponent](body-offset-rally.mp4)

[continual-player.json.gz](continual-player.json.gz) stores the player after 42 scored points at six installation visits. It includes four camera-fitted motor maps, observed rebound history, shot and delay preferences, stroke speed, the timing posterior, and random state. Load it at the boundary between installations. The player pays for new visual probes before the first serve and may recognize a prior map. The hall, opponent and camera calibration come from the simulator.

## Continue training

Use the project Python environment with MuJoCo, SciPy, NumPy and an EGL-capable renderer. Run from the repository root. For faster component extraction, compile the optional C++ helper as shown below. Omit those two lines to use the Python fallback.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
c++ -std=c++17 -O3 -fPIC -shared scripts/experimental/tt3_components.cpp -o /tmp/tt3-components.so
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components.so
python -m scripts.experimental.tt6_continuous_trial /tmp/my-course.json.gz \
  --resume artifacts/tt6/continual-player.json.gz --checkpoints /tmp/my-player \
  --seed 54782309 --course offset gantry offset-b --predictive --cached-render --reflections
```

Each visit plays seven points by default. Disturbances change within and between points. Each output point contains the public commands, applied commands, optical measurements used by the player, and referee events. A repeated installation name in the evaluator recreates the same wiring and viewpoint. The player only sees the new action width and the camera response to its own commands.

## Reproduce the training and controls

```sh
python -m scripts.experimental.tt6_continuous_trial /tmp/carry.json.gz \
  --seed 3079812 --course gantry offset gantry-b gantry offset gantry-b \
  --mode carry --predictive --cached-render --reflections --checkpoints /tmp/learner
python -m scripts.experimental.tt6_score artifacts/tt6/combined-{carry,frozen,restart}.json.gz
python -m scripts.experimental.tt6_replay artifacts/tt6/combined-carry.json.gz
python -m scripts.experimental.tt6_replay artifacts/tt6/combined-carry.json.gz \
  --visit 5 --point 3 --movie /tmp/first-person.mp4
```

To reproduce a control, use the same training command with `--mode restart` or `--mode frozen`. To reproduce the saved last visit alone, pass `--start-visit 5 --resume /tmp/learner/visit-5.json.gz` with the full original course. The uninterrupted run and the restored run matched on all seven action tapes, applied commands, physical events and scores. The native replay checks every applied command and the final referee accounting. The exported movie uses the original first-person MuJoCo renderer.

## Held-out controls and longer games

Use `--seed 54782309 --course offset gantry offset-b` and `--seed 10932577 --course gantry offset gantry-b` for the two independent forward courses. Start from the varied [player](continual-player.json.gz), the equal-length ordinary [player](ordinary-player.json.gz), or omit `--resume` for the mini-only comparison. To isolate the policy random stream, omit `--resume` and supply `--match-rng artifacts/tt6/continual-player.json.gz`. To recreate ordinary practice, use the original six-visit training command with `--phases ordinary ordinary ordinary ordinary ordinary ordinary ordinary`.

For the longer receiving games, add `--seed 53785301 --course gantry offset --race-to 11 --max-steps 420` to the resumed playing command. `--auto-servo` selects the motor solver from its empirically fitted controllability at each installation. Use it in place of `--predictive` to reproduce the [selection experiment](servo-selection.md).

The [21-point](continual-player-21.json.gz), [42-point](continual-player.json.gz) and [84-point](continual-player-84.json.gz) snapshots can all resume through the same command. To recreate the additional 42 practice points, resume the 42-point checkpoint with `--seed 18204759 --course offset gantry offset-b offset gantry offset-b`.

## Component experiments

- [Transfer to unseen physical joint-axis tilts](axes.md)
- [Different physical arms, reversed motor maps and the fifth-axis intervention](body.md)
- [Hidden latency and command noise, including the independent five-frame confirmation](timing.md)
- [Camera-driven remapping](recall.json.gz) and [independent native-renderer stream](recall-native.json.gz). Use `python -m scripts.experimental.tt6_recall_trial --help` for interface schedules and checkpoint arguments.
- [Racket practice](racket-practice.json.gz), [switched rubber](racket.json.gz), [harder station](racket-confirm.json.gz) and [longer optical reacquisition](racket-reacquisition.json.gz). Use `python -m scripts.experimental.tt6_racket_trial --help` and `python -m scripts.experimental.tt6_racket_confirm --help` for practice and target options.
- [Six-joint ordinary receiving](rotary.json.gz), [hard mixed course](combined-rotary.json.gz) and [matched six-joint stroke experiment](rotary-adaptation.json.gz).
- [Sustained six-joint mixed-condition POV](contact-rotary-rally.mp4), [blocked stroke-selection experiment](contact-blocked.json.gz), [bounded model-predictive control](bounded-mpc.md), [feasibility with known contact timing](contact-feasibility.json.gz) and [RGB-timed pulse](contact-optical-pulse.json.gz).
- [Pong retention and count-only control](pong-retention.json.gz) and [pixel and speed benchmark](render-benchmark.json).
