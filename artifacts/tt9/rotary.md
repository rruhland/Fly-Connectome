# Moving a learned receiving stroke into a full six-joint game

A lift timing policy trained on matched receiving practice transferred to the current predictive arm controller. It took 5 points on the first six-joint installation and 1 on the second. Starting the same lift policy with empty timing memory took 5 and 2. The trained phase preference gives no game-level advantage on these installations.

Every full game started from the same [four-game player](../tt8/span-after-four.json.gz). The receiving overlay used the earlier mixed-condition practice set to choose an early, aligned or late lift from incoming ball motion and camera-estimated actuator delay. It updated those choices from outgoing flight and point score during the game. The service controller recognized the toss from stereo. Its own stroke skipped the lift, and both service and receiving memories continued to update. Each four-point block included the ordinary arm, soft racket, delayed control and combined disturbance. [Source practice and earlier receiving results](../tt7/contact-mixed.md)

| Receiving policy | Station A score | Station B score | Legal player returns A / B |
|---|---:|---:|---:|
| Inherited receiver | 3–11 | 2–11 | 3 / 2 |
| Practiced lift with online updates | 5–11 | 1–11 | 4 / 1 |
| Empty lift memory with online updates | 5–11 | 2–11 | 3 / 0 |
| Fixed early lift | 5–11 | 2–11 | 4 / 1 |

No player made two legal returns on a point. A separate check on station A cleared only the inherited service scores and kept the original receiver. That player lost 1–11 and landed 3 of 6 own serves, compared with the intact player's 3–11 and 5 of 9. Games differed in length and lets changed which side served later starts. The six-joint checkpoint after the practiced lift game on A preserved all 180 reference native Pong command tapes, 41 earned points and 195/206 later returns. [Station A control](rotary-game.json.gz) · [Station B control](rotary-control-b.json.gz) · [Learned A](rotary-lift-carried-a.json.gz) · [Learned B](rotary-lift-carried-b.json.gz) · [Cold A](rotary-lift-cold-a.json.gz) · [Cold B](rotary-lift-cold-b.json.gz) · [Fixed A](rotary-lift-early-a.json.gz) · [Fixed B](rotary-lift-early-b.json.gz) · [Service ablation](rotary-fresh-service.json.gz) · [Native Pong receipts](rotary-lift-pong.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md). For the second installation use point seed 287261843 and installation seed 531806597. For the fixed control use only `--receive-lift early`. For the empty learner use only `--receive-lift online`. For the inherited receiver omit both receiving options.

```bash
python -m scripts.experimental.tt8_service_trial /tmp/rotary.json.gz --body rotary \
  --seed 184618765 --installation-seed 498610233 --limit 50 --phases ordinary soft delay mixed \
  --resume artifacts/tt8/span-after-four.json.gz --receive-lift online \
  --receive-practice artifacts/tt7/contact-mixed-matched-practice.json.gz \
  --probe-cache /tmp/tt8-cache --checkpoint /tmp/rotary-player.json.gz
python -m scripts.experimental.tt8_pong_retention /tmp/rotary-player.json.gz /tmp/rotary-pong.json.gz
```
