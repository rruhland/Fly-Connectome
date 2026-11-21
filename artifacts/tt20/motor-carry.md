# Carrying a motor remap between unmarked eyes

Each eye recovered the same hidden wiring on the factory clamp. The target's five public arm ports mapped to nominal axes `[1, 3, 0, 4, 2]` with signs `[+, +, +, −, +]`, drawn from all five-axis signed permutations. The gripper retained its separate public control. The best complete optical assignments cleared their nearest alternatives by 0.499 for the outside eye and 0.279 for the wrist eye in the estimator's normalized loss. [Optical qualification](motor-carry-probes.json.gz)

Each eye used 35 target control frames across five resets and 70 motion frames in an independent nominal CAD renderer. Four frames of small joint-rate requests per port were followed by three settling frames. The estimator compares native RGB changes on unmarked scenes after masking independently seeded balls. It uses the fixed eye's view of blade and arm motion, and the wrist eye's view of how the world moves with the arm. A global assignment considers every canonical port and both signs. The target and nominal CAD share the declared clamp, lens, body and settling dynamics, with different ball seeds and a private target port assignment. No target joint states, exact blade coordinates or wiring labels reach the estimator. [Optical estimator](../../scripts/experimental/tt20_motor_carry.py) · [Independent calibration and scored replay](../../scripts/experimental/tt20_motor_carry_trial.py)

The optical read at frame two of each pulse was sufficient in retrospective truncation on this installation. The executed procedure charges all 35 target frames and five resets. At 30 Hz, those commanded movements take 1.17 seconds of simulated interaction. The nominal model renders its templates separately.

The scored branches use the same unmarked single-eye perception and factory player. Each receives one firmware-style signed inverse, represented by five axis assignments and signs, from its own or the opposite source. The opposite-eye transfer starts with zero new destination calibration frames and inherits the source's paid 35 frames. If two branches produce the same physical input on all six unit basis actions, linearity establishes the same physical input for any program command. The trial physically replays each equivalent branch with its corresponding interface map under the normal strict score. [Scored results and certificates](motor-carry-games.json.gz)

## Strict games

Both destination views played a fresh 11-point game with the opposite eye's map and a second game with the unadapted remap. The games used the same point seeds and alternating serve schedule. Same-eye calibration and the normal interface produced identical physical inputs on every commanded axis and passed separate physical replays of the adapted games. The replayed variants account for 115 point tapes. [Scored tapes](motor-carry-games.json.gz)

| Destination | Motor condition | Game | Legal own serves | Received points with legal return | Total legal returns |
|---|---|---:|---:|---:|---:|
| Outside | Unadapted | 0–11 | 0/5 | 0/6 | 0 |
| Outside | Wrist carry, same eye, or normal interface | 8–11 | 7/9 | 4/10 | 6 |
| Wrist | Unadapted | 0–11 | 0/5 | 2/6 | 2 |
| Wrist | Outside carry, same eye, or normal interface | 1–11 | 3/6 | 3/6 | 3 |

On the first 11 matched starts, the outside-eye player with the carried map won six points, served legally on three of five and returned legally on four of six. The wrist-eye player with the carried map won one point, served legally on two of five and returned legally on three of six. Both uncompensated players lost all 11 points. The outside-eye game had one point with multiple legal returns. This experiment covers one new wiring installation and one game seed. Its known body, camera geometry and factory player isolate changes to the public motor interface. [Scored tapes](motor-carry-games.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt20_motor_carry_trial
python -m scripts.experimental.tt20_motor_carry_trial --verify
```
