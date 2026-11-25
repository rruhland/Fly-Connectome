# Separate SO-101 fixture and neck-pitch preflight

None of the tested fixture changes qualified for an independent or full-game comparison. The stock body produced the only actual answered rescues, all on one practice release and outside the 58 mm interior threshold. This is a separate mechanical experiment, not a change to the continuous-opponent benchmark.

The first family is a pitched blade neck, distinct from turning the blade around its handle. The cut axis is perpendicular to the handle and the blade normal through a point physically embedded in both original shapes. The seated handle is straight in the existing jaws. The plate retains the original 76 mm radius, thickness and mass. The full rigid paddle retains its 115 g mass and its new COM and inertia are recomputed. Across all four variants the jaw grasp span was at least as long as stock and observed blade-to-wrist separation was above 1 mm. This is CAD fit, not a physical holding-force test.

The second family bolts a twin-web steel shelf onto the outside of the original two clamp spines. The unchanged base is 98 mm farther from the net and its supported deck is 26–66 mm lower. All three shelf geometries maintained the base contact convention and positive clearance to the table. The original joint ready collided on all three shelves. No alternate pose met the commissioning constraints near the surveyed stock face and center. An exploratory farther-away safe posture recovered at most nine of the original 25 covered regular-height sites. This traded away too much ordinary workspace to justify prospective play.

| Installation | Covered regular sites out of 30 | Three legacy low-ray fitted radii in mm |
|---|---:|---|
| Stock | 25 | 60 / 83 / 77 |
| Neck +17° | 24 | No eligible fits |
| Neck −17° | 25 | 77 / 112 / 105 |
| Neck +22° | 23 | No eligible fits |
| Neck −22° | 24 | 86 / none / 115 |

The local low-ray comparison is not a physical impossibility result. The physical development assay therefore admitted the best neck variant that preserved regular-site coverage even though it lost the mapped low-ray comparison. Seven untouched or touched low first offers were frozen independently of new outcomes. The stock and −17° fixture matched every historical far contact and near bounce. Each signed short coach was allowed three rounds of 144 native forecasts and at most three actual candidates. Stock matched the real coach's three legally landed and ordinarily answered alternatives on 15456 with no full-route CAD penetration. Their impacts were all outside the interior threshold. Nothing modeled legal on the new neck, so no surrogate touched forecast is called a save. Stock archived controls and the altered fixture running the same archived controls scored zero on the seven frozen offers.

[Frozen fixture and offer roster](protocol.json) · [CAD dimensions, mass and local atlas](geometry.json) · [Full matched controls and coach tapes](development.json.gz) · [Qualification decision](selection.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt63_fixtures plan
python -m scripts.experimental.tt63_fixtures geometry
python -m scripts.experimental.tt63_fixture_qualification development
python -m scripts.experimental.tt63_fixture_qualification select
python -m pytest -q tests/test_tt63_fixtures.py
```
