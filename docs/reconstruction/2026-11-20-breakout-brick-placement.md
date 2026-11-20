# Learning brick-directed shots from score

After 20 ordinary practice lives, a frozen brick scorer cleared 205 bricks after paddle returns on 48 unseen nine-brick boards. Its frozen, untrained visual prior cleared 185. Both controllers returned every ball they faced. Center pursuit cleared 194 bricks. The learned advantage over the identical prior was positive in both ordinary cohorts. A paired bootstrap across the two cohorts gave a 95% interval of −0.02 to +0.88 bricks per life for the pooled difference. With a hidden 12-command delay, the learned scorer cleared 52 bricks versus 56 for the prior and 43 for either centered controller.

## Visual placement and credit

All arms started with the sealed post-course Pong motor and arrival checkpoint. The role adapter found the paddle by its visible response to requests and expressed the scene along the paddle's motion axis. The brick scorer read the public camera for rectangular targets. It used the inherited arrival and reach estimates to choose among nine paddle-center offsets. A uniform bank of 24 rebound hypotheses initially covered both direction signs and three possible contact biases. The policy ranked offsets by the chance of a folded ray crossing a visible target and deducted a small cost for larger offsets. The time forecast and braking used the existing Pong aim hook. [Pong state](../../artifacts/m1b/2026-11-19-pong-after-mixed-course.json) · [Brick placement code](../../scripts/experimental/breakout_brick_placement.py)

Positive native brick score or a completed scoreless flight changes the model weights by a finite likelihood update. A new visual approach closes the preceding flight. A loss or time cap can cut a flight short, so terminal flights earn credit only when they already have a brick score. The update receives public camera time and score. The arena records actual points of contact and brick removals for evaluation. Practice used 320 seconds of game time and yielded 94 model updates, 80 with brick score. Its 97 scored bricks came after 108 successful returns. The frozen prior cleared 95 after 107 successful returns in 318 seconds of practice. [Score checkpoint](../../artifacts/m2/2026-11-20-breakout-brick-checkpoint.json)

## Matched held-out lives

The ordinary scout was capped at 16 seconds per life. The separate confirmation was capped at 12 seconds. The delay scout introduced 12 queued camera commands, approximately 240 ms, and kept the 16-second cap. Each arm saw the same seeds within a cohort. The 20 practice boards and the 64 test boards had distinct occupancies. The scorer and prior stayed frozen across all tests. The timing and motor updates continued within each life from the same Pong timing checkpoint. The camera-derived paddle role carried across lives within each arm. Initial ordinary setup took 0.133 seconds in either cohort. The delay cohort took 0.333 seconds on its first life. Median setup across all lives of each arm was 0.058 seconds. All time and lost lives enter the denominators. [Experimental runner](../../scripts/experimental/run_breakout_brick_placement.py)

The untrained prior uses the same visual targets, reach limits and forecast path as the frozen trained policy. Raw center uses the inherited pursuit with a zero offset. Forecast center activates the same forecast and braking path as brick placement whenever a reachable center choice has sufficient visual support. Ordinary center forecasts often return the ball to the same vertical path as the bricks thin out. The frozen Pong arm keeps the original score model and shot guard.

| Cohort | Policy | Bricks / lives | Bricks / minute | Returns / opportunities | Lives lost |
|---|---|---:|---:|---:|---:|
| Ordinary scout | Trained brick scorer | 82/16 | 19.22 | 83/83 | 0 |
| Ordinary scout | Frozen visual prior | 73/16 | 17.11 | 82/82 | 0 |
| Ordinary scout | Raw center | 74/16 | 17.34 | 82/82 | 0 |
| Ordinary scout | Forecast center | 27/16 | 6.33 | 85/85 | 0 |
| Ordinary scout | Unchanged Pong aim | 73/16 | 17.11 | 66/67 | 0 |
| Ordinary confirmation | Trained brick scorer | 123/32 | 19.22 | 130/130 | 0 |
| Ordinary confirmation | Frozen visual prior | 112/32 | 17.50 | 126/126 | 0 |
| Ordinary confirmation | Raw center | 120/32 | 18.75 | 124/124 | 0 |
| Ordinary confirmation | Forecast center | 57/32 | 8.91 | 129/129 | 0 |
| Ordinary confirmation | Unchanged Pong aim | 116/32 | 19.70 | 93/98 | 5 |
| Hidden delay | Trained brick scorer | 52/16 | 19.92 | 44/52 | 8 |
| Hidden delay | Frozen visual prior | 56/16 | 21.66 | 41/51 | 10 |
| Hidden delay | Raw center | 43/16 | 16.23 | 50/57 | 7 |
| Hidden delay | Forecast center | 43/16 | 16.23 | 50/57 | 7 |
| Hidden delay | Unchanged Pong aim | 43/16 | 17.72 | 45/53 | 8 |

All scored bricks followed an own return. In the ordinary confirmation, unchanged Pong aim cleared fewer bricks per life than the learned policy and more bricks per minute. That run ended five lives early through misses. The ordinary paired differences versus the untrained prior were +0.56 bricks per life in the scout and +0.34 in the confirmation, with 95% intervals of [−0.38, +1.69] and [−0.09, +0.78]. Delay yielded −0.25 per life against the prior, with a 95% interval of [−0.81, +0.13]. The audit reconciled native scores and returns across 400 lives and 260,221 camera transitions and verified matched starts and frozen shot states. [Schedules, raw receipts and audit](../../artifacts/m2/2026-11-20-breakout-brick-placement.zip)
