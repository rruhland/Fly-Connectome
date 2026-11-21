# Alternating-service games and cross-game transfer

[Watch the full 11–6 game from the player's camera](full-game-first-person.mp4). The 35-second replay includes the toss, strikes, misses, let and score. The controls and physical result were checked against the played game.

The [integrated player](span-after-four.json.gz) keeps both native Pong and table-tennis score evidence while training continues. Across four table games it reproduced all 76 physical action tapes from the corresponding arm-only controls, including after a 32-point return to native Pong. Frozen Pong reproduced all 180 earlier tapes. [Paired retention experiment](motor-span.md)

## Serving from the camera

A launcher releases the ball upward behind the end line. The learner has to identify the near-side toss from stereo, forecast its descending path and move its physical racket into it. A legal serve requires contact on the descent after at least a 16 cm rise, then a bounce on each half. Net lets repeat the service. The opponent has the same seeded opening used in the receiving experiments. Service switches every two scored points and every point from deuce. The simulated launcher supplies the toss. The arm learns how to strike it. [Service rules, section 2.6](https://db.ittf.com/sites/default/files/public/2025-02/2025_ITTF_Statutes_clean_version.pdf)

A no-action control missed all 64 tosses on four bodies. On 18 practice tosses at a gantry with ordinary racket response, the player made 12 legal serves and won ten points. The arm, hall receiver and visual tracker came from the previous 42-point player. A separate service learner selected among six designed strokes that vary in target height, forward speed and blade inclination. It learned from public point outcomes and a camera estimate of whether the struck ball moved forward. Its only state carried into the matched test was its per-stroke reward and use counts. Neither the server flag nor the referee's legal-serve flag was passed to the player.

| Player at entry | Gantry game | Legal own serves | Offset game | Legal own serves |
| --- | ---: | ---: | ---: | ---: |
| Previous receiver, no service stroke | 3–11 | 0/8 | 5–11 | 0/8 |
| Service learner, empty preferences | 4–11 | 2/8 | 2–11 | 0/6 |
| Same learner after 18 practice serves | **11–6** | **7/10** | 1–11 | 0/6 |
| Fixed initial stroke | 11–9 | 2/11 | 3–11 | 0/6 |
| Fixed stroke favored by practice | 11–6 | 7/10 | 1–11 | 0/6 |

These are complete simulated games with a changing mix of ordinary, soft, delayed and noisy racket conditions. The gantry served first and the offset arm received first. Games of different length expose players to different numbers of serves. The fixed initial stroke won nine receiving points on the gantry, showing why match score alone is a poor measure of serving. On that gantry, fixing the stroke to the learned favorite reproduced every command from the practiced learner. Prior practice selected an effective stroke for one new installation. It did not solve serving with the new five-axis body.

[Practice](service-practice.json.gz) · [Service memory](service-player.json.gz) · [Practiced gantry](service-warm-gantry.json.gz) · [Cold gantry](service-cold-gantry.json.gz) · [Fixed favorite](service-bestfixed-gantry.json.gz) · [Practiced offset](service-warm-offset.json.gz) · [Fixed offset](service-bestfixed-offset.json.gz)

## Keeping one player across three games

| Consecutive game | Score | Commissioning probes | Legal own serves |
| --- | ---: | ---: | ---: |
| New four-axis gantry | 11–6 | 8 | 7/10 |
| New five-axis offset arm | 3–11 | 8 | 0/6 |
| Return to the first gantry with new tosses and serves | 11–8 | 2 | 6/12 |

Each game loaded the preceding game's full body library, shared reward and timing statistics, RNG state, racket response and service preferences. The arm-specific command permutation was hidden. The player had to verify its recalled motor model against public camera probes on re-entry. The replay checker independently verified 206 practice and game points across these runs and their controls on the final arena geometry. The native Pong evaluation after the first game reproduced all 180 original action tapes, with 129/160 first returns, 195/206 later returns and 41 earned points. After three games, the same check found a regression. First returns fell from 129/160 to 126/160, later returns from 195/206 (94.7%) to 188/215 (87.4%), and earned points from 41 to 36. Of 180 action tapes, 156 changed. The published tolerance check passed, but the earlier native player performed better. Its role weights stayed identical. New table-tennis rewards had moved a different shared interception relation to the top of the score table. [Native Pong after three games](pong-after-three-games.json.gz)

[Game one checkpoint](after-full-game.json.gz) · [Offset continuation](service-carried-offset.json.gz) · [Gantry return](service-gantry-revisit.json.gz) · [Three-game checkpoint](after-three-games.json.gz) · [Pong retention after game one](pong-after-service.json.gz)

## Physics transfer and motor control

Table-tennis bounce estimates transferred to a different RGB game in which a cart must catch a ball behind a screen. When the same physical material continued and the evaluator handed over its learned prior, the player caught 300/384 falls versus 289/384 for a strong adaptive target-only control. Its advantage was greatest on the very first fall, 36/48 versus 24/48. With hidden material switches and both candidate materials in memory, the table-tennis mixture caught 251/384 versus 273/384 for the fixed generic hard/soft mixture with the same online inference. The first result tests transfer of a physical property. The second tests whether the learner can identify when to use it. [Results, paired courses, replay and extraction method](rebound-transfer.md)

A stroke that explicitly controlled forward blade velocity at impact made 7/8 soft or noisy gantry serves versus 2/8 for the earlier fixed stroke. On the comparable offset course it made 2/8 versus 0/8. Additional adjustment of pace using optical feedback did not improve those runs. In full alternating-service games on two new stations, the trained pace frozen or adapting online both served legally on 3/7 offset attempts. The fixed favorite from the original stroke bank served legally on 1/7. The frozen drive, online drive and fixed stroke finished 2–11, 0–11 and 2–11 respectively. On the new gantry they finished 5–11, 3–11 and 5–11. The two new stations were evaluated with identical initial state for each comparison. [Service-only control and replays](service-drive.md) · [Online offset game](drive-game-online-offset.json.gz) · [Frozen offset game](drive-game-frozen-offset.json.gz) · [Fixed offset game](drive-game-hard-offset.json.gz)

On a further six-parameter search, 30/35 practice attempts on seven distinct tosses were legal, falling to 2/10 on untouched tosses where the fixed controller made 7/10. Constraining all planned motor commands helped sustained play at one of two new five-axis stations. [Serve search](service-search-offset.md) · [Bounded motor control](bounded-mpc.md)

Optional compiled material commands produced identical pixels on 288 matched stereo frames and identical full-game receipts in six runs. Across two warm-cache measurements per mode, mean full-game CPU time went from 27.46 to 26.74 seconds. Compiling the entire shape initially changed two pixel channels by one unit on a specific toss and eventually changed the score. The optional implementation caches material state only. The original rendering path remains the experiment default. [Timing measurements](render-timing.json)

## Reproduce the alternating-service game

Use the [MuJoCo environment](../tt6/README.md). The portable service preferences can be tested alone, or the entire player can be resumed from a full checkpoint.

```bash
python -m scripts.experimental.tt8_service_trial /tmp/practice.json.gz \
  --seed 83531471 --practice --limit 18 --service-out /tmp/serve.json.gz --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt8_service_trial /tmp/first-game.json.gz \
  --seed 128753901 --service-in /tmp/serve.json.gz --checkpoint /tmp/full-player.json.gz \
  --phases ordinary mixed soft delay --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt8_service_trial /tmp/next-game.json.gz \
  --seed 219706341 --body offset --first-server -1 --resume /tmp/full-player.json.gz \
  --phases ordinary mixed soft delay --checkpoint /tmp/after-offset.json.gz --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt8_service_replay /tmp/first-game.json.gz /tmp/first-game.mp4
```
