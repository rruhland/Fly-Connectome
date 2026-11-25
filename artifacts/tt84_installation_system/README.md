# Designing the whole end-clamp installation

This study asks whether the whole stock six-joint arm is poorly placed for the actual spread of receiving balls. It jointly varies lateral and depth placement, pitch, roll and yaw. The original table and clamp shoes, arm and active joints, racket, camera hardware, motor limits, ball, continuous opponent law and scoring are retained. It is explicitly a different rigid installation, not the original benchmark or a manufactured load certification.

## Workspace and impulse screen

The source is 884 measured ordinary deliveries across 128 uninterrupted stock training points. A fixed random sample retained 72 each from low, transition and routine heights. The screen used mostly continuous measured lines, not a fixed center or a selected successful donor. It projected each recorded receiving-line state to four depths and asked for two useful outgoing pace bands. A precomputed stock joint atlas transforms rigidly with the proposed mounting pose. It checks blade-face plane, contact offset, approximate travel time and local blade-patch velocity in both the normal and upward tangent directions. Mixed directions prevent the same joint from being counted twice at full rate. The geometry predicts support, not a legal return. It ignores loaded motion, complete route clearance and later rebound errors.

Of 192 five-dimensional proposals, 27 fit the original CAD base footprint on the supported deck with the declared height bound. Ranking required ordinary and transition coverage broadly comparable to stock, then traded low and regular support across the complete sample. A no-wedge +12 cm translation is included as a simpler reference.

| Installation | Low / 72 | Transition / 72 | Routine / 72 |
|---|---:|---:|---:|
| Stock | 36 | 45 | 69 |
| +12 cm translation only | 47 | 45 | 66 |
| Combined installation | 49 | 43 | 70 |

The combined design is +119.72 mm lateral and 5.61 mm farther inboard, with +17.96° pitch, −7.60° roll and −4.00° yaw in the declared extrinsic XYZ convention. The taper has at most 61.94 mm thickness. Two relieved lower pads bear on the original end plate. CAD measured a base-to-deck gap below 8 nm and 1.00 mm and 4.00 mm clearance from the upper jaws and playing table, respectively. The rig was separately commissioned to the original ready blade center within 2.2 mm. It retains 24 of 25 ordinary routed atlas sites. The screening advantage over lateral-only remains small. [Protocol and source checksums](protocol.json) · [Joint workspace screen](screen.json) · [Compiled support and calibration](build.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt84_installation_system plan
python -m scripts.experimental.tt84_installation_system screen
python -m scripts.experimental.tt84_installation_system build
python -m scripts.experimental.tt84_installation_system physical
python -m scripts.experimental.tt84_installation_compare
python -m scripts.experimental.tt84_installation_play plan
python -m pytest -q tests/test_tt84_installation_system.py
```
