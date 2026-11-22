# Frozen ready-pose validation

Use two fixed blocks of six fresh receiving points against the same physical anchor rival. The player receives calibrated stereo RGB, prepares its blade and then runs the existing late bounded-value stroke policy. The fixture uses .30 of standard launch spread and .82 of speed. Each point ends under the normal strict referee. The scoring excludes the serve.

Both branches use the policy and motor parameters from the finished three-start pilot. The stationary branch uses the fixed factory ready pose. The Pong branch uses the best native point-score coefficients to choose a continuous lateral ready position. Both keep the same late ranker, timing, blade geometry, candidate generators, duration-based proposal variants and edge options. Coefficients, motion caps, preparation window, ranker and both opening blocks remain fixed through the comparison. [Runner](../../scripts/experimental/tt23_value_ready.py)

On four fixed starts, play a mirrored control as well. It flips the requested lateral ready position about the court center. The magnitude cap, step bound, body solver and late ranker match the Pong branch. The body and later point can move differently, so log actual preparatory joint commands and their absolute effort for both.

| Block | Fixed launch order | Mirrored control starts |
|---|---|---|
| 1 | 2130715543, 399331003, 659743118, 1520097002, 949556383, 2134479342 | 399331003, 949556383 |
| 2 | 1999740997, 781303606, 1321024455, 358697341, 131612783, 464758626 | 781303606, 131612783 |

The launch seed lists were drawn using NumPy SeedSequence 56769413 and two independent child generators before either block was played. The Pong and stationary branches alternate which runs first. The first mirrored control in each block runs before its pair, and the second runs afterward.

Record 10-return and 20-return counts, longest alternating lengths, totals for each arm, point wins and the number of paired rallies that lengthen, shorten or tie. Report each block and the aggregate. Measure readiness by the number of frames from the first preparation action to the late stroke selection. Record wall time spent computing preparation and the sum of absolute joint-rate commands over the preparation frames. Evaluator blade and ball positions are stored separately. Replay every point from the archived controls and check referee events and terminal outcome.
