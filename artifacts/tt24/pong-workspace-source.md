# Can official Pong serves teach a mover-dependent workspace shift?

Zero shift gave the strongest terminal point score on both native Pong pilot courses. The gate for fitting a signed gain from reward therefore stayed closed. The planned reward fits, within-serve shuffles and wrist tests were left unevaluated. [Registered protocol](pong-workspace-source-plan.json) · [Native diagnostic outcomes](pong-workspace-source-pilot.json.gz)

The frozen actor received actual official serves from the native engine. These naturally vary the ball's starting row and couple it to launch angle. The wide roster balanced 72 starts whose first visible ball position differed from the familiar mover position by at least 0.15 screen units. The ordinary roster drew 32 unselected seeds. Native player starts, both players' actuation, serve velocities and physics came directly from the engine. Actual terminal point score was the sole gate measure. A 12-second timeout scored zero. [Source runner](../../scripts/experimental/tt24_pong_workspace.py)

The controller kept its learned motor weights and adjusted the equilibrium used by its pose feature and servo. It first identified the compact mover in the interior raster and took the median of three consistent positions. Along the shared actuator and mover row axis, it subtracted the original paddle center and the familiar ball-to-paddle offset measured from 90 earlier training rasters. It capped the displacement at 0.75 apparent paddle half-heights, then multiplied it by the signed gain. This is the one-axis counterpart of the wrist reference update, which applies a bounded horizontal translation while preserving height and face normal. The earlier native origin experiment instead changed how far the controller reused its familiar body pose after a displaced paddle start. [Physical wrist operation](workspace-stress.md) · [Earlier native origin experiment](tt24_pong_origin.md)

| Signed position gain | Wide 72, net score | Ordinary 32, net score |
| --- | ---: | ---: |
| −1.0 | −27 | −9 |
| −0.5 | −16 | −5 |
| 0.0 | **−8** | **−1** |
| +0.5 | −12 | −4 |
| +1.0 | −17 | −10 |

Every launch yielded a consistent query on camera tick two. All 72 wide cues reached the cap of 0.078125 screen units and all four nonzero settings changed every command tape. The zero setting reproduced the original actor's commands and outcomes on all 104 starts. At +0.5, 70 of the 72 wide starts registered at least one return, compared with 67 at zero. It also lost 13 points, compared with nine at zero. Each gain won one wide point and no ordinary points. The pilot used 520 actual points across the fixed controls. The gate required an improvement of at least four points over zero and at least three over the opposite sign on the wide roster, better results on at least three individual wide starts, and an ordinary score no more than two points below the zero setting. All four nonzero settings scored below zero on both rosters. [Per-start command digests, settled queries and scores](pong-workspace-source-pilot.json.gz)
