# Returning from air hockey to changed Pong

The learner carried information about wall rebounds from air hockey back to Pong. Every comparison started with the same saved Pong actuator timing and the same wall estimate learned in elastic Pong. A camera flight tracker measured the ratio of outgoing to incoming lateral speed at a visible wall bounce. Early estimates used five outgoing camera frames. A later estimate used eight. The longer estimate replaced overlapping short estimates. The learner used the current estimate to project the ball's crossing at its paddle. [Pong actor](../../scripts/experimental/arcade_hockey_pong.py) · [Shared wall tracker](../../scripts/experimental/arcade_wall_learn.py)

Four practice blocks gave the learner evidence about how often the wall response changed and what responses to expect. A block needed at least two observed wall bounces to inform that distribution. The learner received public camera images, requested buttons and, during hockey practice, ordinary goal rewards. It received no restitution coefficient or physical position. The trainer supplied block boundaries. The image decoders, rail geometry, projection formula and confidence threshold were supplied in code. The numeric bounce estimates and the distribution of familiar and unfamiliar responses came from the camera history. The estimators used weighted moments and discrete probability updates.

All extra-practice conditions used 3,360 camera transitions. One played air hockey with a guard opponent. Two played native Pong with either elastic walls or the same varied wall coefficients as the hockey practice. A fourth condition went straight from the shared checkpoint to the test. A fifth could detect a change and select the hockey practice mean after a surprise, then held that mean fixed. A sixth kept the original Pong forecast throughout.

The test used 16 paired courses with distinct held-out wall coefficients. Each played six bank openings and two ordinary native serves. The evaluator created bank openings with shallower wall incidence to limit the visible displacement per camera frame. Bank openings used ball speeds from .74 to .99 compared with .55 in the ordinary starts. The native physics applied the damped wall response on contact. The same starts and physics were used by every condition. The evaluator checked the first incoming path and used physical contact and wall coefficients to compute outcomes after the controller acted. The first 16 requests in the first life matched across all comparison arms. [Experiment runner](../../scripts/experimental/run_arcade_hockey_pong.py) · [Native Pong physics](../../src/fly_connectome/pong.py)

## Held-out results

The checkpoint began at a wall estimate of .991. The additional hockey practice yielded estimates of .458, .592, .725 and .825 from 37 visual bounces. The first opening of each target course gave all six conditions identical pre-contact evidence. The independent audit checked 62 short or full bounce measurements per condition.

| Preparation | Revised before first full bounce | Mean error after first partial bounce | Mean error after first full bounce | Final mean error | Mean cameras to stable estimate, capped at 800 |
|---|---:|---:|---:|---:|---:|
| Shared Pong checkpoint | 5/16 | .319 | .100 | .020 | 216.1 |
| Equal extra elastic Pong | 3/16 | .335 | .100 | .020 | 221.4 |
| Equal varied Pong | 12/16 | .163 | .105 | .021 | 197.3 |
| Air hockey | 12/16 | .165 | .106 | .021 | 197.3 |
| Hockey mean held fixed after surprise | 12/16 | .182 | .160 | .127 | 465.8 |
| Frozen Pong | 0/16 | .405 | .405 | .405 | 800.0 |

Hockey practice prompted an earlier revision than the shared checkpoint in 13 of 16 courses and earlier than extra elastic Pong in 14. The remaining courses tied in both comparisons. Its median first revision was at 30.5 camera frames compared with 32.5 for both elastic-Pong controls. The mean first partial-bounce error was lower by .154 against the shared checkpoint and .171 against extra elastic Pong. The respective 95% paired course-bootstrap intervals were [−.250, −.063] and [−.270, −.073]. The first full-bounce estimates and final errors converged to similar values across the live controls.

A stable estimate stayed within .07 of the actual coefficient through the remaining recorded evidence. With an 800-camera cap, the hockey-trained estimate took 18.8 fewer camera frames than the shared checkpoint and 24.1 fewer than extra elastic Pong. The descriptive paired intervals were [−49.5, −0.4] and [−54.8, −0.9]. Equal varied Pong revised and reached the stable threshold at the same times on all 16 courses. Experience with physics variation explained the measured speedup across these preparations. Live target observations also improved on holding the hockey average fixed, especially by the end of the course.

The four learners that updated their coefficient during the test each returned 120 of their 128 first incoming balls and finished with three goals, ten losses and 115 timeouts. The 32 ordinary serves per condition ended in four losses and 28 timeouts. With the clear camera and this interception controller, the earlier physics estimates yielded no score improvement.

The steeper four-course pilot gave all live controls enough camera evidence to reach a stable estimate on the first full bounce. The shallower eight-course pilot informed the later confirmation. The archive contains both pilot plans and summaries, the 16-course preregistration, all confirmation rows, and the audit. It preserves the exact registered air hockey source before a later one-line floating-point tolerance change. Elastic Pong matched the previous implementation bit for bit for 900 physics ticks across six seeds, including 300 rendered camera batches. [Experiment receipts](../../artifacts/m2/2026-11-20-air-hockey-pong-wall-transfer.zip)
