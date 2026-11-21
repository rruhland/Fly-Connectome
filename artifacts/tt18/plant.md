# TT18 physical table and arm variants

The two mixed fixtures compile different MuJoCo models for the existing TT8 service and TT17 scored-game interface. The unchanged fixture uses the original TT17 factory. The mixed fixtures change the ball, the receiver's mechanics, and the actual table contacts together. The net, markings and supports follow the resized table. Both fixtures keep the same tabletop height, ball diameter, arm actuators and point rules. The learner's seeded toss shifts with the end line. The scripted rival's serve launch scales with table length and its rally controller retains its original approximations.

| Physical setting | TT17 | Compact | Extended |
| --- | ---: | ---: | ---: |
| Table width × length (m) | 1.525 × 2.740 | 1.408 × 2.640 | 1.626 × 2.860 |
| Ball mass (g) | 2.700 | 3.375 | 3.618 |
| Receiver link mass and inertia scale | 1.00 | 1.22 | 0.86 |
| Receiver actuated joint span | 100% | 86% | 94% |
| Left and right table solver damping magnitudes | 24, 24 | 43, 17 | 17, 40 |
| Arm camera mount | — | 45 g, wrist | 65 g, proximal link |

Each camera housing contributes mass and moment of inertia to its carrier. The mounted view is a native RGB camera whose position and orientation follow that link. The fixed stereo pair is available for the inherited player. On the gantry, the proximal mount translates along the bridge without rotating. Its wrist mount rotated about 10° over the first two probe commands. The two table materials are coplanar. Both can contribute to a contact at the panel seam, which the point scorer counts as one tabletop bounce.

## Matched physical probes

- **Ball.** The 34% heavier ball used the same absolute aerodynamic force and spin-decay torque at the same initial speed and spin. After 0.20 s it retained 2.926 m/s along the table versus 2.843 m/s for the original ball. A matched blade impact at 3.700 m/s normal approach separated at 3.308 m/s with the heavier ball and 3.318 m/s with the original ball.
- **Surface.** Vertical drops on the regulation-size split surface hit at 2.919 m/s. The two sides left at 2.270 and 2.781 m/s. Mirroring the materials swapped those speeds and the identical center drop produced one scored bounce. The compact table dropped a ball at lateral coordinate 0.740 m onto the floor. The original and extended tables returned it from the tabletop.
- **Mass placement.** Adding 50 g to the proximal link or wrist gave each arm the same total mass under the two treatments. The same 25-command tape produced blade trajectories that differed by up to 21.1 mm on the gantry and 29.8 mm on the five-axis arm. Adding an actual 50 g camera at either mount also changed the compiled arm mass and the observed blade trajectory.
- **Opponent service.** An idle learner received a legal opponent opening on 100 of 100 regulation starts, 99 of 100 compact starts and 95 of 100 extended starts. Those starts span seeds 0 through 99. The extended misses included four subsequent rival double hits and one ball reaching the floor before a legal reception bounce.
- **Scored commands.** All 29 archived TT17 point tapes reproduced their original score, events and applied actions on the unchanged table. Each of the two mixed fixtures changed the events on all 29 tapes. These were independent point replays in the recorded service roles, capped at the last recorded command. Eight to twelve tapes per case ended within that cap on the mixed fixtures.

The audit uses evaluator state for controlled launches and measurements. The mounted camera, ball, arm and table all belong to the compiled simulator model. The plant settings and the evaluator measurements stay outside the player's RGB and reward interface.

Use `make_arena(spec)` with an optional evaluator `plant` field containing `compact`, `extended`, or a settings mapping. For a same-size two-material fixture, set `left_damping` and `right_damping`. `link_mass_scales` takes named learner link multipliers such as `bridge` and `wrist` on the gantry. [Plant module](../../scripts/experimental/tt18_plant.py)

The audit receipt includes the physical settings, body masses and moments, camera frame hashes, action traces and contact events. [Audit receipt](plant-audit.json.gz)

```bash
MUJOCO_GL=egl python -m scripts.experimental.tt18_plant_audit artifacts/tt18/plant-audit.json.gz
```
