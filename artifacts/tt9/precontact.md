# Five-axis service from the observed blade–ball approach

Learning lateral offset and stroke advance from the five-axis blade’s observed approach raised held serve legality from 15/24 to 23/24 on two offset-arm installations. The continued player lost both alternating-service games 5–11. It used the carried player after four alternating-service games, the existing 0.132 m/frame velocity objective, and a separately fitted RGB motor model for each installation. Learning used stereo RGB, issued commands, and public point reward. The simulator’s contact events, serve flags, and outcome reasons were used for evaluation. [Starting player](../tt8/span-after-four.json.gz) · [Fixed drive](../tt8/service-drive.md)

The last several visible ball positions define a short ballistic continuation into the region where the blade can hide the toss. The observer locates the crossing of that continuation with the moving blade face, accounting for the ball radius and disc thickness. It interpolates between measured blade positions and inclinations. At that crossing it records the signed horizontal and vertical displacement on the face and the rate at which ball and blade close. An observed crossing can lie outside the blade radius. The player can therefore measure the direction of a miss even if it receives no departing ball track.

A ridge fit learns stroke advance from the signed vertical displacement and measured toss descent. The fit can condition on the ball’s lateral position and the latency inferred from RGB blade movement. Losses receive more weight than successes. A separate fit uses signed horizontal displacement to shift the approach laterally. The swing speed objective remains fixed in both experiments.

## Timing correction

Each of two offset-arm installations received 16 distinct practice tosses and 14 held tosses. Practice updated the contextual correction after each attempt. The three held policies shared the same RGB motor models and had the same soft and mixed conditions, tosses, command disturbances, and starting rally state. The pooled correction fit only one lead across the contextual learner’s practice evidence.

| Installation | Held policy | Racket contacts | Legal serves | Serve points won |
| --- | --- | ---: | ---: | ---: |
| A | Contextual timing | 14/14 | 11/14 | 8/14 |
| A | Pooled timing | 14/14 | 11/14 | 8/14 |
| A | Fixed velocity | 12/14 | 12/14 | 6/14 |
| B | Contextual timing | 9/14 | 7/14 | 7/14 |
| B | Pooled timing | 10/14 | 6/14 | 6/14 |
| B | Fixed velocity | 10/14 | 6/14 | 5/14 |

Across the 116 attempts, the approach estimator recovered a blade-face crossing on every serve, including all 24 complete misses. A four-frame departure fit was available on 72/73 legal serves and 12/43 failures, with zero of the complete misses. The median distance from the observed blade center was 5.5 cm on legal serves and 10.9 cm on complete misses. Among 92 actual strikes, the median absolute difference between optical crossing time and the simulator contact timestamp was 0.135 video frames (4.5 ms). The optical crossing is a geometric estimate from sampled images.

The contextual correction won four more held serve points than fixed velocity across the two installations. They tied on legality. On B the seven complete misses during practice had a median horizontal gap of 10.0 cm and median vertical gap of 1.2 cm. This motivated adding lateral correction on a new set of tosses. [Timing tape A](precontact-offset-a.json.gz) · [Timing tape B](precontact-offset-b.json.gz) · [Timing checkpoint A](precontact-offset-a-player.json.gz) · [Timing checkpoint B](precontact-offset-b-player.json.gz)

## Lateral and timing correction

The two installations each received 12 new practice tosses, followed by 12 distinct held tosses. During these isolated serving trials the service fit carried between practice points. Each toss began with the same inherited rally, racket and latency state, and the service fit froze for the held tosses. In the full games below, the entire player carried forward after each point. The complete model and the timing-only ablation used the timing fit learned by the complete model. The ablation set lateral displacement to zero. All three held policies shared the same starts and disturbances.

| Installation | Held policy | Racket contacts | Legal serves | Serve points won |
| --- | --- | ---: | ---: | ---: |
| A | Lateral and timing | 12/12 | 12/12 | 5/12 |
| A | Timing only | 12/12 | 12/12 | 4/12 |
| A | Fixed velocity | 11/12 | 10/12 | 6/12 |
| B | Lateral and timing | 12/12 | 11/12 | 9/12 |
| B | Timing only | 6/12 | 4/12 | 4/12 |
| B | Fixed velocity | 6/12 | 5/12 | 5/12 |

Compared with fixed velocity, adding the two corrections turned eight illegal held serves into legal serves and lost none. Points won on serve rose from 11/24 to 14/24. A already made contact on most tosses and won one fewer held serve point after the correction. B had six additional contacts and four additional serve wins. B’s gains occurred with both undisturbed soft serves and mixed delay and command noise. Its one nonlegal serve with the complete correction was a let.

The estimated center displacement had different signs on the two installations. At the center of the normalized inputs, the learned offset was −2.6 cm on A and +5.0 cm on B. With both corrections enabled, the median optical distance from the blade center decreased from 5.4 to 4.3 cm on A and from 8.6 to 4.2 cm on B. The lateral ablation on B had a 7.7 cm median distance and made six fewer contacts on the same tosses. [Registration tape A](precontact-registration-a.json.gz) · [Registration tape B](precontact-registration-b.json.gz) · [Practiced player A](precontact-registration-a-player.json.gz) · [Practiced player B](precontact-registration-b-player.json.gz)

## Alternating-service games

The frozen held test and the following games use disjoint toss seeds. Each game started from its installation’s practice checkpoint. The player recalled its saved body map after one new RGB motor probe per installation. The player then faced an opponent on the full table in ordinary, soft, mixed, and delayed conditions. During the game it continued fitting the approach on its own tosses and carried its usual rally and motor-latency updates. The fixed control used the same practice checkpoint and receiving policy with the 0.132 m/frame serve and no optical registration correction. Both controllers faced the same sequence of point seeds and conditions within each installation.

| Installation | Serving policy | Final score | Legal serves | Serve points won | Legal returns | Sustained points |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Continuing registration | 5–11 | 5/9 | 0 | 6 | 1 |
| A | Fixed velocity | 4–11 | 4/8 | 2 | 6 | 1 |
| B | Continuing registration | 5–11 | 6/8 | 2 | 11 | 3 |
| B | Fixed velocity | 6–11 | 1/8 | 1 | 10 | 2 |

Sustained points contain at least two legal player returns after service. On B the opponent returned five of the corrected serves legally. The player made a legal return in two of those exchanges and sustained one of its own serves. On A the opponent returned five corrected serves legally and the player made no subsequent legal return. The larger gains in held legality therefore reached B’s full game, but neither game produced a win. A’s held test also favored fixed velocity on points won from the serve. [Game A](precontact-match-a.json.gz) · [Game B](precontact-match-b.json.gz) · [Continuing player A](precontact-match-a-player.json.gz) · [Continuing player B](precontact-match-b-player.json.gz)

The after-game checkpoints carry 21 approach observations on A and 20 on B, up from 12 apiece. Both full player checkpoints reloaded successfully. All 278 point tapes across the six receipts reproduced the applied commands, physical events, and scores. The recorded raw observations and public rewards also reproduced every stored crossing and chronological correction update.

## Reproduce the experiments

```bash
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt9_precontact_trial /tmp/precontact-a.json.gz \
  --seed 398512109 --installation-seed 501137399 --probe-cache /tmp/tt8-cache \
  --checkpoint /tmp/precontact-a-player.json.gz
python -m scripts.experimental.tt9_precontact_trial /tmp/precontact-b.json.gz \
  --seed 83041723 --installation-seed 317557501 --probe-cache /tmp/tt8-cache \
  --checkpoint /tmp/precontact-b-player.json.gz
```

```bash
python -m scripts.experimental.tt9_precontact_registration_trial /tmp/registration-a.json.gz \
  --seed 635723681 --installation-seed 501137399 --calibration-seed 398512109 \
  --checkpoint /tmp/registration-a-player.json.gz --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt9_precontact_registration_trial /tmp/registration-b.json.gz \
  --seed 1004386721 --installation-seed 317557501 --calibration-seed 83041723 \
  --checkpoint /tmp/registration-b-player.json.gz --probe-cache /tmp/tt8-cache
```

```bash
python -m scripts.experimental.tt9_precontact_match /tmp/registration-match-a.json.gz \
  --seed 558279631 --installation-seed 501137399 --probe-cache /tmp/tt8-cache \
  --resume artifacts/tt9/precontact-registration-a-player.json.gz \
  --checkpoint /tmp/registration-match-a-player.json.gz
python -m scripts.experimental.tt9_precontact_match /tmp/registration-match-b.json.gz \
  --seed 926138271 --installation-seed 317557501 --probe-cache /tmp/tt8-cache \
  --resume artifacts/tt9/precontact-registration-b-player.json.gz \
  --checkpoint /tmp/registration-match-b-player.json.gz
```
