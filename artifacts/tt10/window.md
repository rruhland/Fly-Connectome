# Choosing when and where to strike a service reply

The variable contact window landed 3 of 10 replies in paired replays, compared with 2 of 10 at the inherited hitting depth. It lost ground on the same measure in two fresh alternating-service games. The practiced five-axis arm won 13–11 with the window and lost 9–11 at the inherited depth. The gantry lost 9–11 and 8–11 respectively. The window rescued one reply in the paired replays and spoiled two already-legal replies in the full games. Keep the inherited stroke for service recovery. [Paired points A](window-paired-a.json.gz) · [Paired points B](window-paired-b.json.gz) · [Five-axis games](window-game-offset.json.gz) · [Gantry games](window-game-gantry.json.gz)

## Same serve and prior history

Each fixed stroke reproduced every command in the earlier complete game on its five-axis installation. For all ten legal replies to a registered serve, the evaluator forked the adaptive player immediately before the point. The two branches requested identical commands through the rival's contact. The window changed nine shots and selected depths 7.5, 15 or 22.5 cm nearer the net. Each controller made 8 of the 10 first follow-up contacts.

| Archived installation | Stroke | Legal first follow-ups | Later legal returns | Points won |
| --- | --- | ---: | ---: | ---: |
| Five-axis A | Inherited | 0/5 | 0 | 0 |
| Five-axis A | Window | 1/5 | 0 | 1 |
| Five-axis B | Inherited | 2/5 | 2 | 1 |
| Five-axis B | Window | 2/5 | 3 | 0 |

The change on A turned a short follow-up into a far-side bounce and a point win. On B, one held exchange gained three later legal returns and another lost two later legal returns and the point. Every baseline tape matched the original games, including points outside the paired set. The twenty paired tapes were physically replayed with identical applied commands, events and scores. [Five-axis A baseline](../tt9/precontact-match-a.json.gz) · [Five-axis B baseline](../tt9/precontact-match-b.json.gz)

## Full games from the practiced player

Both controllers started from the same [five-axis B checkpoint after a complete game](../tt9/precontact-match-b-player.json.gz). They commissioned their body from RGB motor probes, updated the inherited stroke, timing and service registration from RGB and point reward, and stayed active across each full game. Each installation had a fresh, ordered set of point seeds including ordinary, soft, noisy and delayed conditions. The installation, starting server, disturbances and game outcomes were available to the evaluator.

| Installation | Stroke | Final score | Legal own serves | Legal first follow-ups | Later legal returns after own serve | Total legal returns |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Five-axis B | Inherited | 9–11 | 5/10 | 1/2 | 1 | 13 |
| Five-axis B | Window | 13–11 | 7/12 | 0/2 | 0 | 14 |
| New gantry | Inherited | 8–11 | 4/9 | 2/4 | 4 | 16 |
| New gantry | Window | 9–11 | 2/10 | 1/2 | 0 | 12 |

On the five-axis arm, the two runs used identical commands through the rival's contact on the first changed point. The inherited stroke landed two returns and lost the point. The window landed zero and lost the point. On the gantry, the first changed point also matched through the opponent's contact. The inherited stroke won after five legal returns. The window missed and lost. Once points changed, subsequent control updates and, on the five-axis arm after deuce, serve order could diverge. The higher final scores therefore came with fewer legal follow-ups and fewer won points on answered own serves. The window activated on three legal-reply points and on zero rival-service points. All 83 full-game tapes replayed. [Five-axis end state](window-game-offset-player.json.gz) · [Gantry end state](window-game-gantry-player.json.gz)

The gantry end state reproduced all 180 reference native Pong action tapes and their 41 earned points. [Pong retention](window-pong.json.gz)

## What the model forecast

The controller recognizes its toss and watches for an observed ball to reverse depth on the far side. During the first approaching reply it projects the ball through the existing flight model. For each proposed contact depth and wrist pitch, it predicts what the current visual arm model can do with the queued commands, estimated actuator delay, and three subsequent rate commands within the usual motor limits. It compares the estimated face crossing, blade velocity and pitch, and projected net clearance and landing to those of the original stroke. It keeps the inherited stroke if the alternatives score too close or fall outside the available time. The servo keeps recalculating commands during the selected stroke.

At selection time the predicted contacts were within a median 0.37 frames and 5.6 cm of the actual impact times and depths on eight paired contacts. Forward speed differed by a median 0.025 m per frame, about 0.74 m/s, from blade displacement across nearby RGB observations. The flight model forecast one legal shot across the nine selected windows. Three landed legally, and the single forecast legal shot bounced on the player's side. In two legal cases the initial estimate gave the moving blade less than 0.008 m per frame of forward speed, while nearby RGB observations later showed more than 0.050 m per frame. The short rollout holds the initial three-move solution, while the live servo replans every camera frame. Both that difference and errors in the optical approach and impact estimate can change the forecast departure. [Referee and RGB forecast audit](window-forecast-audit.json)

## Reproduce

Use the [hall environment](../tt6/README.md). The runner defaults to the practiced five-axis B game checkpoint for fresh games. The output of `--checkpoint` can be passed back to this runner with `--resume` to carry its state to the next installation.

```sh
python -m scripts.experimental.tt10_window_trial /tmp/window-paired-a.json.gz \
  --seed 558279631 --paired artifacts/tt9/precontact-match-a.json.gz \
  --resume artifacts/tt9/precontact-registration-a-player.json.gz --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt10_window_trial /tmp/window-paired-b.json.gz \
  --seed 926138271 --paired artifacts/tt9/precontact-match-b.json.gz \
  --resume artifacts/tt9/precontact-registration-b-player.json.gz --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt10_window_audit /tmp/window-forecast-audit.json \
  /tmp/window-paired-a.json.gz /tmp/window-paired-b.json.gz
python -m scripts.experimental.tt10_window_trial /tmp/window-game-offset.json.gz \
  --seed 839061427 --installation-seed 317557501 --probe-cache /tmp/tt8-cache \
  --checkpoint /tmp/window-game-offset-player.json.gz
python -m scripts.experimental.tt10_window_trial /tmp/window-game-gantry.json.gz \
  --seed 544710629 --installation-seed 741720081 --body gantry --probe-cache /tmp/tt8-cache \
  --checkpoint /tmp/window-game-gantry-player.json.gz
python -m scripts.experimental.tt10_window_trial /tmp/window-game-gantry.json.gz --verify
python -m scripts.experimental.tt8_pong_retention /tmp/window-game-gantry-player.json.gz /tmp/window-pong.json.gz
```
