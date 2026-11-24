# The tested handle reseatings do not open the low branch

Every alternative that held the same cylinder as well as stock and cleared the wrist moved the moderate low balls **farther** toward, or beyond, the blade rim. Moving the handle inward would help in the pertinent signed branch, but the blade then enters the gripper envelope. This is a separate geometry falsification. It does not change the continuous-opponent benchmark, the paddle, or the SO-101 range of motion.

Eight configurations were declared before the atlas and game scores. We varied only seating distance and small rotation about the unchanged handle, around a pivot at the actual gripper site. For each one, full CAD triangle slices measured the cross-section of both jaws around the cylinder at the existing jaw command. The stock handle presented 5.0 mm of continuous bilateral contact-band span, below our initial absolute 6.0 mm engineering proxy. Before checking alternate reach, we added a stock-equivalent qualification which requires at least the measured 5.0 mm span without increasing blade crowding. An alternate also had to maintain at least 1 mm of positive blade-to-wrist signed distance. The nominal blade carries a −3.4 mm conservative convex-hull reading against the fixed jaw and is retained strictly as the existing bench control. The nominal geometric fit and this CAD proxy do not measure real jaw friction or holding force.

| Grip | Shaft band | Closest blade/wrist | Local low-ray radii 7038 / 5864 / 4563 | Regular centered sites |
|---|---:|---:|---:|---:|
| Stock | 5.0 mm | −3.4 mm | 60 / 83 / 77 mm | 25/30 |
| Out 8 mm | 5.0 mm | +4.6 mm | 68 / 91 / 85 mm | 25/30 |
| Out 16 mm | 5.0 mm | +12.6 mm | 76 / 99 / 93 mm | 24/30 |
| In 8 mm | 5.0 mm | −10.8 mm | Rejected before reach | — |
| Out 12 mm, tilt −7° | 5.5 mm | +8.6 mm | 71 / 94 / 87 mm | 22/30 |
| Out 12 mm, tilt +7° | 5.0 mm | +8.6 mm | 72 / 96 / 89 mm | 25/30 |
| Out 12 mm, roll −15° | 5.0 mm | +8.6 mm | 72 / 95 / 89 mm | 25/30 |
| Out 12 mm, roll +15° | 5.0 mm | +8.6 mm | 72 / 95 / 88 mm | 25/30 |

The low-ray fits start from the original measured arm angles and require the same deadline, achieved axial range, face alignment and clear links as the original [rim study](shallow-middle.md). The 30 regular-height sites span both sides, three depths and two heights. The qualified configurations retained 22–25 of the original 25 sites, gained none, and already worsened all three low rays. In this joint branch the outward slide increases radial error about one-for-one with extension. No config clears the declared +4 mm progress on two low shots with at least 85% regular-height coverage, so no altered grip was sent into the reserved live course. There is no motor-memory transfer claim. [Declared configurations and live cards](shallow-grip-plan.json) · [Jaw geometry](shallow-grip-holds.json) · [Stock-equivalent reach comparison](shallow-grip-geometry-relative.json)

The auxiliary configuration changes the weld transform at construction and nothing else. The model checks original local geom transforms, individual arm inertials and servo ranges, and the untouched ball, court and camera parameters. Mounted and ordinary scenes are isolated. Tests confirmed the alternative reapplies the correct paddle pose on reset and leaves later ordinary simulations on the original weld. Seven relevant unit tests passed, including the baseline SO-101 tests.

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_shallow_grip plan /tmp/shallow-grip-plan.json
python -m scripts.experimental.tt41_shallow_grip holds /tmp/shallow-grip-holds.json --plan /tmp/shallow-grip-plan.json
python -m scripts.experimental.tt41_shallow_grip geometry /tmp/shallow-grip-geometry.json \
  --plan /tmp/shallow-grip-plan.json --holds /tmp/shallow-grip-holds.json --stock-equivalent
```
