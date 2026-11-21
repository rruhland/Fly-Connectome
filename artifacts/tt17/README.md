# TT17 · Learning altered arm controls from stereo vision

The table player started with its previously trained camera and stroke models. After 24 commands on newly weakened or cross-coupled motors, a fit to the observed blade movement improved returns on both the gantry and five-axis arm. A controller that applied its travel limit in the learned arm coordinates converted one of the four losses into an 11–9 win. Software supplied the exploratory movement, camera tracking, stroke planner and travel rule. The player estimated the controls from RGB without access to the simulator's motor matrix or a backpropagated loss.

| Question | Evidence |
| --- | --- |
| Can the player infer more than a motor's sign and position? | [Motion errors, stable controls, in-game comparator and Pong retention](motor.md) |
| Does the better estimate help the arm play a whole game? | [Controller comparisons, continuation and downloadable player](allocation.md) · [Watch the coupled gantry win](gantry-coupled-full-game.mp4) |
| What happens when order, sign and coupling all change together? | [Four additional installations and 12 matched games](compound.md) |
| Can the player learn where to intercept an opponent serve from experience? | [Point reward and an optical rally cue](strikes.md) |
| Why do second returns fail on the six-joint rotary arm? | [Full games, matched serves and measured stroke speeds](rotary.md) |
| Does experience with this arm help on a different physical task? | [Accelerate a puck through a gate and into a dock](payload.md) |

A return is counted when the arm's shot lands legally. A multiple-return point requires at least two such shots on the same point. Every complete game alternates service, plays to 11 and requires a two-point lead. New hardware games include changing transport delay and noise. See the individual comparisons for calibration cost, opponent seeds and evaluator replays.
