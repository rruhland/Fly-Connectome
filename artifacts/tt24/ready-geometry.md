# Choosing a wrist ready reference from the visible release

This experiment keeps the five-joint, donor-reward wrist controller frozen and changes how it chooses its ready reference. The same joint reference sets the controller's motor equilibrium and the public observer's paddle-centered spatial origin. The familiar baseline uses the existing fixed donor-average pose. The conditional branch selects a posture from public wrist sightings of the ball before release. [Frozen wrist skill](tt24_donor.md) · [Public observer](public-body.md) · [Ready preparation](../../scripts/experimental/tt24_ready.py)

The familiar pose restored ten-rebound streaks in 16 of 28 paired starts. Selecting the reference from release geometry matched the familiar pose on ten-rebound counts in the fixed course and yielded fewer in the roving course. The posture learned from successful donor scenes did not improve reliability at the wider release locations.

## What the source provides

The fit uses one re-rendered 192 × 144 wrist image and five own encoder readings at each of 24 archived held launches across three physically different hinged donors. All 24 had a usable public ball sighting. The fit includes the 22 releases on which the earlier donor pupil made at least five consecutive physical rebounds. It uses the rebound score to select examples and uses the initial joint readings as posture targets. It uses zero receiving action sequences and no target motor examples. The existing source policy was rewarded for bouncing with clean donor observations. The new selector uses the stock public wrist model to infer the ball coordinates in these source images. [Archival source play](tt24_donor-refined.json.gz) · [Released selector coefficients](ready-geometry-source.npz)

The selector fits a regularized linear relationship from two horizontal launch coordinates to deviations from the familiar joint pose. The public wrist model supplies the shared coordinate axes and uses a color-defined sphere sighting. It centers examples by donor for the slope fit and limits requested posture shifts using the source encoder spread. A control independently permutes the pairing between launch coordinates and joint postures within each donor. The fixed familiar pose and starting-pose modes make no launch-based adjustment.

## Held physical comparison

The stock learner starts in independently randomized `WidePosture` configurations. A separate simulated dispenser places the ball according to the standard release distribution in the fixed course and a wider `WidePosture` release distribution in the roving course. Dispenser and receiver are reset with independent seeds in both courses. The dispenser is relocated between drops and holds its ball still during each preparation interval. Only the evaluation harness has its true position, release velocity and spin.

The four paired conditions have the same 14–38 preparation ticks at 60 Hz and the same true launch and learner start. Conditional and shuffled start with the familiar reference and switch once when they have three consistent public ball estimates during the first half of preparation, within 11 ticks. They retain familiar if they lack sufficient sightings. A switch preserves the absolute commanded servo targets by expressing the accumulated offsets in the new coordinates. All four conditions issue fresh feedback commands across the preparation and active drop. [Assessment runner](../../scripts/experimental/tt24_ready_geometry.py) · [Paired physical data](ready-geometry.json.gz)

| Reference | Fixed mean longest streak | Fixed ≥10 | Roving mean longest streak | Roving ≥10 |
| --- | ---: | ---: | ---: | ---: |
| Familiar donor-average pose | 9.83 | 6/12 | 11.19 | 10/16 |
| Geometry-conditioned pose | 10.25 | 6/12 | 9.63 | 8/16 |
| Association-shuffled pose | 8.83 | 5/12 | 9.50 | 8/16 |
| Starting pose | 0.83 | 0/12 | 0.69 | 0/16 |

On fixed releases, the conditional reference won four paired drops, tied six and lost two against familiar. On roving releases it won five, tied seven and lost four. Its losses were larger than its wins. Against shuffled geometry the roving conditional branch won five, tied six and lost five. Its longest streak was 21 in the fixed course and 18 in the roving course. The starting reference never exceeded one rebound in either course.

The true fixed release positions spanned 108 × 65 mm horizontally. The roving course spanned 139 × 135 mm. Every paired case supplied enough public sightings for the conditional estimate. The median reference switch was at tick two in each course, and median horizontal error against the withheld simulator position was 2.46 mm fixed and 2.42 mm roving. The two fitted branches matched the familiar branch's commands up to their reference switches. All 112 resulting command tapes reproduced the full physical referee events and final control ticks. [Raw releases, inferred references and physical replay records](ready-geometry.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_ready_geometry \
  --output /tmp/ready-geometry.json.gz --fit /tmp/ready-geometry-source.npz
```
