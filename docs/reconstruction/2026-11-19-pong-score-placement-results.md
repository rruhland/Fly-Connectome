# Pong score placement, first checkpoint

The score learner practiced on 96 native rallies against the .35 tracker
after restoring the independently qualified timing state. It won 53 rallies
after its own return and 18 on the original serve. Two ended in losses and 23
timed out. It returned 141 of 143 opportunities, including all 78 opening
opportunities. The 78 terminal updates reconstruct the frozen score table to
within 1e-12. The 18 natural wins carried no eligible score assignment.
[Run receipts](/workspace/m1b-experiments/pong-score-placement-20261119-r1/)

## Fresh paired starts with timing held fixed

| Opponent and arm | Own-hit wins | Natural wins | Losses | Timeouts | First returns | All returns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| .35 learned score, 48 starts | 32 | 6 | 1 | 9 | 42/42 | 63/64 |
| .35 uniform eligible offsets | 24 | 6 | 2 | 16 | 42/42 | 90/92 |
| .35 inherited interception | 0 | 6 | 0 | 42 | 42/42 | 154/154 |
| .35 centered forecasting and braking | 0 | 6 | 0 | 42 | 42/42 | 162/162 |
| .65 learned score, 24 starts | 0 | 0 | 2 | 22 | 24/24 | 65/67 |
| .65 inherited interception | 0 | 0 | 0 | 24 | 24/24 | 82/82 |

The paired learned gain in own-hit wins was 66.7 percentage points against
either center control, with a 95% bootstrap interval of [54.2, 79.2]. The
gain against the uniform placement control was 16.7 points [0.0, 33.3].
At .65, the change in all-return rate was −3.0 points [−7.8, 0.0].
[Matched outcomes](/workspace/m1b-experiments/pong-score-placement-20261119-r1/summary.json)

The independent comparison that kept timing and motor adaptation live also
qualified scoring at .35. It recorded 24/32 own-hit wins, versus 3/32 for
inherited interception and 14/32 for fixed exploration. At .65 the score
policy returned 19/20 first balls and 30/33 later balls, versus 20/20 and
49/49 for inherited interception. This missed the 95% later-return gate.
[Independent run](/workspace/m1b-experiments/pong-score-qualification-20261119-r1/)

The practice and internal evaluation used source `6477a97` and the actual
restored Stage 2 checkpoint from the qualified v3 run. Score checkpoint
SHA-256 is `484059d03112dc06b18c946d8da244e0d229d5e63ee3da08ebef20fd94c762eb`.
Stage 2 timing updates were held fixed for this experiment. The visual motor
observer still tracked paddle motion.
[Input pins](/workspace/m1b-experiments/pong-score-placement-20261119-r1/input.json)

The portable archive contains all 96 practice and 240 paired evaluation
receipts, the frozen score table, upstream checkpoint copies and lineage, and
an audit script that reconstructs the terminal-score state without new camera
observations. The audit passed after a fresh extraction, and every content
hash matched the archive manifest.
[Portable receipts](../../artifacts/m1b/2026-11-19-pong-score-placement.zip)

## Camera facing-edge projection development check

The optional contact estimate fits the ball's recent public visual motion
and projects to the facing edge of the rendered paddle. A separate option
shrinks the offset when the trajectory fit or its agreement with the inherited
projection is weak. Each active variant used the frozen first score table.
There were eight fresh paired starts per condition and no score updates.

| Variant | .35 own-hit wins | .35 returns | .65 returns | Mixed first after switch | Mixed later after switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original placement | 5/8 | 12/13 | 20/22 | 5/6 | 4/7 |
| Camera facing-edge projection | 5/8 | 13/13 | 23/23 | 5/6 | 4/7 |
| Projection with uncertainty shrinkage | 5/8 | 14/14 | 21/22 | 5/6 | 4/7 |
| Inherited interception | — | — | 27/27 | 5/6 | 12/12 |

The mixed starts carried the learned 12-request delay, started with ordinary
buttons, and changed their order at 2.4 seconds. Two starts per arm ended
before any post-switch opportunity. Camera projection preserved wins in this
small scoring sample and failed to improve mixed retention. Source
`65211df`. [Development receipts](/workspace/m1b-experiments/pong-contact-geometry-dev-20261119-r1/)
