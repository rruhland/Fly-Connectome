# Online play and practice with the clamped SO-101

The arm plays on the unmarked compact TT20 table. The standard and stronger rivals execute physically limited paddle motions. The strict referee counts a return after it lands legally across the net. A second learner return on a point means the same arm landed two shots after service. The experiments distinguish that from a legal serve followed by a single return.

## Learning during the game

An unexplained four-frame motor delay appeared after four normal points. From one eye and its own ordinary commands, the player inferred the new delay at the end of the first affected point on all four games (34–44 control frames). It adjusted both service timing and its receiving forecast. It used an independent nominal rendering of its own motions to estimate response lag. The physical game continued to supply all actual observations and scores.

| Single policy camera | Frozen → adaptive, game 1 | Frozen → adaptive, game 2 |
|---|---:|---:|
| Outside | 3–11 → 5–11 | 0–11 → 0–11 |
| Wrist | 1–11 → 5–11 | 1–11 → 4–11 |

On 33 post-change starts reached in both branches, the learner won 7 points versus 0 for the frozen player. It made 6/18 legal first returns and 13/15 legal serves, compared with 0/18 and 0/15. On a new outside-camera game the same learner detected a two-frame delay and its later removal, each after one affected point. Its score in that game was 0–11 versus 1–11 frozen. [Game receipts and per-point costs](game-delay.md) · [Watch both wrist players take the same serve](game-delay-wrist.mp4)

In a separate handoff, only the delay estimate from ordinary play moved between cameras. On 12 matched new opponent serves per direction, outside experience improved wrist first returns from 0 to 3, and wrist experience improved outside first returns from 1 to 2. Point wins tied in both comparisons. [Cross-eye carry receipts](game-eye-carry.md)

We also removed the extra nominal camera renders from delay identification. A quieter, cheaper cue measures when the arm first moves relative to its own requested stroke in the live image. On a fresh schedule with one, five and two frames of delay it identified each change after its first ordinary point with either camera. The wrist score moved from 0–11 to 3–11 and the outside score from 1–11 to 0–11. On the original two wrist games this estimator reproduced every command, RGB digest and referee event over 1,834 frames. Delay estimation took 8.2 milliseconds across both games, compared with 35.31 seconds and 1,143 extra renders for the independent-arm method. Live vision and stroke planning are additional costs. [Model-free lag measurement and costs](motion-onset.md)

## Preparing the arm for the reply

Fifteen exchanges supplied enough optical observations to fit a small predictor of where the rival’s next shot would pass the end of the table. On 32 fresh points, preparing the shoulder from the incoming ball added three legal answers after a rival return (2 → 5), three total legal returns (16 → 19) and three point wins (9 → 12). On all three changed results the learner won the exchange. The two arms each made 14/16 first returns on rival serves; the improvement came later in the point. One gain was a true second return after receiving, and the other two followed a learner serve. A zero-practice straight-line predictor with the same shoulder motion answered 4 replies and won 11 points. One additional reply and win on this bank distinguished the fitted predictor from that stronger baseline. In two fresh complete games the online fit carried 15 → 22 → 32 observations; centered, frozen and online players each finished 2–11 and 4–11. The fit uses an explicit strike location, optical tracking and a known shoulder model. [Watch the shoulder prepare and win the rally](ready-position-rally.mp4) · [Matched starting states and exact replays](ready-position.md)

The same frozen lateral fit scored no gain through either single camera on 12 fresh starts per eye. It issued three outside-camera pans and one wrist-camera pan across the 24 starts. Camera-independent motor latency transferred more readily than this visually estimated spatial skill. [Single-eye trial](ready-mono.md)

## Learning the second stroke

The timing learner tried three ways to recover after the first stroke. It scored them from its own point rewards and the forward trajectory seen in RGB. Its practice curriculum cost 15 points and 1,166 target control frames including physically replayed openings. The unmarked left and right fixed cameras supplied the observations. The robot carried its body geometry, gantry-derived physics, optical tracker and stroke repertoire into practice. Its learned component was the choice of when to use a stroke. On faster incoming balls the trained selector attempted a stroke earlier. [Training, held points and replication](recovery.md)

On the first 12 fresh points, learned and frozen players each returned five of six rival serves, won two points, and served legally three times. The learner reached a second legal landing on one point. The frozen policy reached none. Both lost the separately scored whole game 2–11 with five first returns. The paired clip shows the two identical openings, then the earlier strike, legal return and eventual loss on the learned side. [Watch the comparison](recovery-paired-638131348.mp4) · [RGB and referee verification](recovery-paired-638131348.json)

The independent 32-serve replication produced one gained and one lost second return. Both policies returned a second ball on 1/15 genuine reply opportunities. A narrower timing window left all 24 fresh command tapes identical. We kept the factory timing as the default. [Matched trials](recovery.md)

## What held back the next return

| Question | What the comparison established |
|---|---|
| Does a more accurate model of the returning arm change its shot? | A nominal arm driven by the player's own commands reduced the blade-center error at rival contact from 4.80 to 0.21 mm. It left all 12 matched action choices identical. [Six physical replies](body-state.md) |
| Is ball reacquisition the main wrist-camera problem? | On 14 fresh starts a better estimate generated an earlier second swing on one point. Both arms returned six opening serves, landed zero second returns and won two points. [One-eye trial and occlusion audit](reacquire.md) |
| Does a replayed opening help practice? | Each practice condition saw six real rival replies after physically replaying an opening, versus two when starting normally. The guarded timing fallback left held wins and legal returns tied across 16 points with each opponent. [Continuation training and controls](continuation.md) |
| Does imitating archived successful strokes shorten planning? | The 18-stroke subset cut planner CPU by 49% but returned three of 12 held serves. The full bank returned five, and a shuffled-credit subset returned four. [Distillation trial](demonstration.md) |
| Does more point-reward practice improve the inherited serve? | Continuous exploration across four games scored eight points to the fixed serve's eleven. A separate 280-point serving drill also lost to the incumbent on new ordinary tosses. [Continuous checkpoints and held comparisons](serving.md) |
| Can a coach give the player useful second shots? | A cooperative rival aims its physical strokes toward the end-clamped arm. An evaluator with full state independently verifies whether any rate-limited counterstroke can return a particular ball. This validates the practice distribution. It supplies no evidence of independent learning unless the RGB player improves on its own held points. [Matched competition and practice rivals](coach.md) |

A privileged teacher supplied three legal stroke demonstrations in a separate experiment. A camera-driven player could adapt those strokes to new incoming balls using the known arm model. Across four disjoint archived opportunities, stock, taught and generic stroke styles produced one, two and three legal continuations. On six fresh starts only the generic style gained a legal continuation and a point. This expands the verified set of possible strokes. The three demonstrations gave no independent advantage over the generic style. [Teacher and generic-style comparison](apprentice.md)

## Adapting to a changed bounce

We tilted the physical tabletop four degrees sideways together with its markings and net, while leaving the clamped arm and fixed cameras in place. The learner watched four real simulated serves, compared its predicted lateral crossing with the later optical track and learned an offset. On ten fresh tilted serves the median crossing error was 259 mm for straight-line extrapolation, 251 mm for the original fit and 67 mm after adaptation. On eight further fresh serves, moving from the straight-line predictor to the updated fit raised actual contacts from 2/8 to 6/8. Both landed zero legal returns. The learned spatial correction got the paddle to the ball, while stroke angle and outgoing speed remain the limiting skills for that fixture. [Changed surface, held forecasts and scored strokes](changed-bounce.md)

## Receiving practice

On a restricted distribution of legal incoming serves the outside-camera learner practiced 24 actual points from an empty target-stroke memory. It learned associations between RGB trajectories, its own commands and optical outcomes. On 20 fresh tosses it returned 12 first balls versus 10 for the cold planner, gaining six starts and losing four. The strong opponent legally answered all 12 of its returns versus seven of the cold player's ten. The learner won zero points versus three for cold. Point-level scoring therefore distinguished a more frequent first interception from a harder shot to return. [Practice stream, held points and controls](receive-lesson.md)

## Returning to earlier games

The 67 KB checkpoint carries the SO-101 camera-drive state alongside the prior policies. Reloading it reproduced the earlier gantry game, 7–11 with 18 legal returns and 8/8 legal serves. On Pong all 180 routed action and outcome tapes matched the earlier run. Each SO-101 camera also restored a legal return and continued the delay updates. This measures retention through explicit task and body selection. It does not establish that learning on the arm improved Pong or gantry. [Checkpoint and resume commands](carry-checkpoint.md) · [Player bundle](carry-checkpoint-player.json.gz)

## Compute

The receiving planner first rejects stroke trajectories whose swept bounding boxes cannot intersect any eligible ball segment. The original detailed contact and landing calculation runs on the remaining trajectories. Across 36 optical histories this reduced the number of strokes scored from 73,728 to 35,694. Shortlist calculation fell from 2.57 to 1.23 CPU seconds (2.09×). All 14,643 feasible rows had identical indices, timings, positions and costs. The original full outside-camera game reproduced all 786 frames, including RGB hashes, control tapes and scorer outcomes. The timing covers shortlist calculation only. [Benchmark receipt](planning-speed.json) · [Run the comparison](../../scripts/experimental/tt21_planning_speed.py)

## Next research target

The next learner needs to decide which return leaves this specific arm time and reach for the following ball. The stronger rival supplied ordinary competitive tests, while the cooperative rival made six of eight matched openings physically returnable by the exact evaluator, versus three for the stronger rival and two for the regular rival. That provides a way to teach the player to choose between its stock and retargeted strokes on real incoming replies. Credit the choice using the subsequent visible landing, continued rally and final point. Carry the same policy into unassisted games. Count distinct points with two legal learner returns, point wins and time spent on the teaching fixture. Record wins on unreturned serves separately so the rally count measures actual play. The goal is to increase both sustained returns and points won.

The hierarchical repertoire and the separation of practice opponents from match evaluation draw on work with a physical table-tennis robot. Its hardware, data and learning method differ from this simulator. All TT21 online updates use direct scores, optical observations, or model fitting without backpropagation. [Achieving Human Level Competitive Robot Table Tennis](https://arxiv.org/abs/2408.03906)
