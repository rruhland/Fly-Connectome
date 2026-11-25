# Sliding the entire end clamp moves the central-low reach hole

Moving the stock clamp and SO-101 12 cm to the right increased safe, answered returns in the fixed development assay from one of seven to five of seven. Three of the five struck inside 58 mm of the blade center. The stock coach had no interior answer. This is evidence for the **translated fixture**, not a result on the original bench or an RGB-only rally milestone.

The complete steel end clamp and stock arm base translate together parallel to the table edge. The move does not alter their height, depth or rotation. The same two screw shoes support the table underside, the same plate supports the arm, and the original regulation table, net, blade, actuators and continuously randomized opponent remain in place. An older opening-only study moved this clamp by up to 13 cm and found that replacing a single stored endpoint by a static retarget made play worse. The current comparison instead recenters normal ready using geometry and searches a moving stroke with the current joint model. [Earlier mounting experiment](../../tt27/trajectory-learning.md)

| Clamp offset | Regular covered sites / 30 | Central-low interior sites / 36 | All low interior sites / 84 |
|---|---:|---:|---:|
| 0 cm | 25 | 3 | 27 |
| −12 cm | 24 | 13 | 37 |
| +12 cm | 24 | 13 | 38 |
| −18 cm | 21 | 20 | 46 |
| +18 cm | 20 | 20 | 45 |

These are routed candidate poses rather than return probabilities. The 18 cm shifts covered more low centers but lost too many ordinary sites under the fixed 85% threshold. The preassigned selection picks +12 cm over −12 cm on the one additional all-low site. Its surveyed ready blade center is about 2 mm from the original center. Its encoders and kinematics are different at that same site. [Frozen comparison](protocol.json) · [CAD atlas and joint-ready results](geometry.json)

On all seven development releases, both fixtures preserved the exact original far contact and near bounce. The actual interior coach hits on 9254, 8831, and 15456 measured 16.51, 38.99 and 43.84 mm radially. The two other shifted answers were outside the 58 mm threshold. The simpler test that replayed the exact archived **failed** command sequences already rescued 8831 and 15456 on the shifted body. Those two were ordinarily answered, the latter at a 42.23 mm contact radius. Their stock counterparts failed. The coach knows the physical ball state and is not the deployed actor. The opponent never reads its selected action or changes its draw law. [Full physical development tapes](development.json.gz)

At the same encoder pose on both fixtures, the fixed cameras measured a red-paddle translation of 121.04 mm, 1.04 mm relative to the nominal x shift. Both base plates had the same support margin and the closest non-support clearance was 18.0 mm. In seven neutral paired physical flight prefixes, 198 usable RGB position/velocity estimates were exactly equal on both fixtures. Each camera produced 112 detections per condition. The calibrated control saw a 2.70 mm median position error and 0.056 m/s median velocity error on both. All 8,336 total CAD samples were nonpenetrating. No simulated ball coordinate reached the optical estimate. [Stereo and support survey](optical-calibration.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt63_lateral_clamp plan
python -m scripts.experimental.tt63_lateral_clamp geometry
python -m scripts.experimental.tt63_lateral_clamp development
python -m scripts.experimental.tt63_lateral_clamp assessment
python -m scripts.experimental.tt63_lateral_optics
python -m scripts.experimental.tt63_lateral_report
python -m pytest -q tests/test_tt63_lateral_clamp.py
```
