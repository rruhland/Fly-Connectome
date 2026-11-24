# A pitched clamp spacer exposes more of the low blade

Small handle reseatings made the low workspace worse. A separately declared six-degree adapter on the same table-edge clamp improves the same-signed branch’s local reach without changing the SO-101, its joint limits, the paddle, or the opponent. It retains 22 of stock’s 25 regular-height sites and reduces radius on all three moderate terminal rays. At the explicit 71 mm training-support line, one previously outlying ray moves into support and another still does not.

## Isolated geometry

The rigid tapered spacer screws onto the unchanged top clamp plate. The top and base change angle together. The stock base’s original position in the tabletop plane does not change. A six-degree pitch adds 4.13 mm at the frame origin and slopes from 1.5 to 11.1 mm across the supported base footprint. The fixed fixture changes, not the moving links, joint limits, motor settings, link and blade masses, grasp, table edges, or opponent. The CAD validator checks original local geometry and inertia arrays and tests the base’s physical support and tabletop clearance. The native convex base support reads between −6 and −13 μm on the alternate spacers, under the same 0.15 mm support convention. The baseline seating reads −1 nm.

| Base pitch | 7038 | 5864 | 4563 | Normal centered sites |
|---|---:|---:|---:|---:|
| Stock | 60 mm | 83 mm | 77 mm | 25/30 |
| +4° | 55 mm | 79 mm | 72 mm | 23/30 |
| +6° | 52 mm | 76 mm | 69 mm | 22/30 |
| −4° | 72 mm | 95 mm | 88 mm | 26/30 |

The three ray entries are minimal fitted ball-center tangents with the existing elbow/shoulder limits, collision screen and optimistic travel deadline. These are not claimed physical returns. The six-degree candidate wins by predeclared geometry alone. It loses three specific normal/deep atlas cells and gains none, so it does not preserve all ordinary coverage. [Frozen adapter plan](shallow-clamp-plan.json) · [CAD seating and ray census](shallow-clamp-geometry.json)

## Frozen memory comparison

The independent regulation points use the same six measured launches and continuous random replies, with identical opening and reply seeds and at most 24 total legal returns per point. There is no reset or partner catch within a point. Both conditions use the same existing RGB/encoder field and the same optical fit. The raw condition uses learned joint memory without prediction, while the frozen search condition also gives the independent camera-initialized predictor the calibrated fixture geometry. Neither branch learns or uses the true incoming ball state to select a command.

Raw motor memory produced two genuine returns on the alternate fixture, both answered, but fell from 9 to 2 own returns overall. It lost both stock returns on 11600 and five of seven on 13221. A native replay verified the alternate's two learned returns on 13221 with unchanged events and zero body or table penetrations in 2,760 solver states. That is partial frozen memory transfer but no measured improvement without recalibration. [Stock raw tape](shallow-clamp-raw-stock.json.gz) · [Spacer raw tape](shallow-clamp-raw-alternate.json.gz) · [Native raw replay](shallow-clamp-raw-cad.json)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_shallow_clamp plan /tmp/shallow-clamp-plan.json
python -m scripts.experimental.tt41_shallow_clamp geometry /tmp/shallow-clamp-geometry.json --plan /tmp/shallow-clamp-plan.json
python -m scripts.experimental.tt41_shallow_clamp raw /tmp/stock.json.gz --condition stock
python -m scripts.experimental.tt41_shallow_clamp raw /tmp/alternate.json.gz --condition alternate
python -m scripts.experimental.tt41_shallow_clamp frozen /tmp/frozen-stock.json.gz --condition stock
python -m scripts.experimental.tt41_shallow_clamp frozen /tmp/frozen-alternate.json.gz --condition alternate
```
