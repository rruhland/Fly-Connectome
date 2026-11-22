# TT22 causal inverse flight fit with the physical cycle rival

The stereo SO-101 kept its frozen swing library and command-driven arm model. A causal fit to up to 21 recent stereo observations estimated the ball center, velocity and spin in a nominal MuJoCo model containing only the ball and table. The fitted path also identified a rebound near the tabletop on the learner’s half. That inferred phase was latched for the current incoming leg. When a fit passed its optical residual limits, these estimates initialized the near player’s dry model. Missing estimates fell back to the existing optical receiver. The live near player received RGB, its clock and its own issued commands. The far player remained the separate privileged physical cycle controller. [Inverse observer](../../scripts/experimental/tt22_flight_inverse.py) · [Cycle integration](../../scripts/experimental/tt22_cycle_inverse.py)

| Ordinary start | Original optical state | Inverse flight state | Accepted fit frames | Fits used in planning |
|---|---:|---:|---:|---:|
| 2119291602 | 4 | 4 | 8 | 0 |
| 265865819 | 1 | 1 | 0 | 0 |
| 509321423 | 2 | 2 | 10 | 1 |
| 1213749908 | 2 | 3 | 16 | 2 |

On seed `1213749908`, the inverse fit entered the dry solver at frames 9 and 14. The revised exchange produced an extra legal learner return at frame 84 and changed a loss to a win. The other three seeds produced identical command tapes. Across the four starts, near legal returns rose from five to six and far returns stayed at four. Both policies peaked at four alternating returns. The comparison used the original verified tapes on those same starts as the control. No target practice or held updates were added.

All three planning decisions initialized by the inverse fit occurred while the observed ball was still approaching its first near bounce. Optical evidence for a bounce did appear on other frames, between planning decisions. The 34 accepted fit frames therefore offered little direct evidence about post-bounce decisions. These four points establish one improvement in actual play and a limit in how often the inverse estimates reached the planner. All four new action tapes reproduced native RGB hashes and strict referee events. [Paired native tapes](cycle-inverse.json.gz)
