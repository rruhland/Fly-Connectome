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

These are routed candidate poses rather than return probabilities. The 18 cm shifts covered more low centers but lost too many ordinary sites under the fixed 85% threshold. The preassigned selection picks +12 cm over −12 cm on the one additional all-low site. At x = −6 cm, 0 and +12 cm, surface coverage changes respectively from 1→7, 1→5 and 5→1 of 12 cells. The weakness moves to the right instead of disappearing. The surveyed ready blade center is about 2 mm from the original, with a different encoder pose. [Frozen comparison](protocol.json) · [CAD atlas and joint-ready results](geometry.json)

On all seven development releases, both fixtures preserved the exact original far contact and near bounce. The actual interior coach hits on 9254, 8831, and 15456 measured 16.51, 38.99 and 43.84 mm radially. The two other shifted answers were outside the 58 mm threshold. The simpler test replayed the exact archived **failed** command sequences from each fixture's geometric ready pose. It already rescued 8831 and 15456 on the shifted body. Those two were ordinarily answered, the latter at a 42.23 mm contact radius. Their stock counterparts failed. The coach knows the physical ball state and is not the deployed actor. The opponent never reads its selected action or changes its draw law. [Full physical development tapes](development.json.gz)

At the same encoder pose on both fixtures, the fixed cameras measured a red-paddle translation of 121.04 mm, 1.04 mm relative to the nominal x shift. Both base plates had the same support margin and the closest non-support clearance was 18.0 mm. In seven neutral paired physical flight prefixes, 198 usable RGB position/velocity estimates were exactly equal on both fixtures. Each camera produced 112 detections per condition. The calibrated control saw a 2.70 mm median position error and 0.056 m/s median velocity error on both. All 8,336 total CAD samples were nonpenetrating. No simulated ball coordinate reached the optical estimate. [Stereo and support survey](optical-calibration.json)

## Independent real-ball checks

On three frozen low misses from release IDs separate from development, stock scored none. The +12 cm coach rescued two with the exact original far contact and near bounce. Both were CAD-clean and drew an ordinary legal answer. The interior return on 14429 measured 54.23 mm. On 10762 the cleaner of two successful strokes was at 71.73 mm, outside the interior threshold. The receiver still missed 15541, whose incoming x was about +9 cm near the shifted body's low hole. The geometry advantage therefore generalizes mechanically, without meeting the registered two-held-interior criterion for the formal TT63 rollout.

Both bodies preserved all four preassigned routine positives with unchanged archived joint-rate tapes from their respective geometric ready poses. The coached counterpart also earned a clean interior answer on every one of these stock-scoring controls. An independent command replay reproduced all 31 scored tapes across the development and held studies, checked all 40,896 CAD solver states without penetration, and rechecked every requested opponent draw against the standard law. This tests preservation at fixed opportunities. It does not establish that a current RGB actor will choose the newly supported movements during an uninterrupted point. [All held and routine tapes](assessment.json.gz) · [Command-only replays and independent referee recount](verification.json)

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
