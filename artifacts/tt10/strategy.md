# Sideways placement on the serve

In matched practice, a rightward blade sweep moved the opponent's contact a median 7.9 cm on five-axis installation A and 5.3 cm on B. The learned placement choice landed 3 of 11 legal replies on fresh throws, compared with 5 of 14 for the registered serve. The contextual player took more points in both full games and made fewer first follow-ups in each. [Practice A](strategy-practice-a.json.gz) · [Practice B](strategy-practice-b.json.gz) · [Qualification A](strategy-qualification-a.json.gz) · [Qualification B](strategy-qualification-b.json.gz)

## Method and matched practice

Each installation began with its TT9 registration checkpoint. Three strokes crossed the same planned impact center with the inherited 0.132 m/frame forward drive, height and pitch targets. The two sweeps requested lateral velocity of ±0.052 m/frame through the blade's otherwise registered contact. The camera-calibrated servo planned both the displacement and the lateral velocity near impact. Joint coupling means the achieved forward motion and pitch can change as well. The receiving stroke was shared across all conditions. [Control and learning implementation](../../scripts/experimental/tt10_strategy.py) · [TT9 registration](../tt9/precontact.md)

Each of nine practice tosses per installation was repeated for all three options in shuffled order using the same pre-point body, rally, latency and registration state. The placement memory used public point reward and a 0.5 bonus when the RGB track showed the ball come back from the rival and cross the table center again after the learner's reply. The memory fit separate ridge regressions by option using the first available toss x and visual latency estimate. A pooled selection from the same observations picked the fixed rightward sweep at both installations. Service legality, opponent contacts and legal follow-ups were evaluator-only. [Trial and replay](../../scripts/experimental/tt10_strategy_trial.py)

| Installation | Placement | Legal serves | Legal rival replies | First legal follow-ups | Later legal returns on own serve | Points won on serve |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Registered | 8/9 | 6 | 1 | 0 | 5 |
| A | Left | 4/9 | 2 | 2 | 3 | 3 |
| A | Right | 8/9 | 3 | 1 | 0 | 7 |
| B | Registered | 8/9 | 5 | 3 | 1 | 5 |
| B | Left | 5/9 | 4 | 1 | 0 | 1 |
| B | Right | 8/9 | 5 | 3 | 4 | 5 |

The rightward sweep increased the camera-measured sideways slope in all 11 matched tosses with comparable outbound tracks. The median within-throw slope change was +0.031 m/m on A and +0.059 m/m on B. The opponent's actual contact moved right on all eight matched tosses with opponent contact in both conditions. A's practice gain on points won coincided with three fewer legal rival replies, and B tied on points won with three more later returns. The RGB continuation detector can miss legal return events. It saw one of three legal first follow-ups for the registered control on B and all three for the rightward sweep. [Practice A](strategy-practice-a.json.gz) · [Practice B](strategy-practice-b.json.gz)

## Fresh tosses

Qualification used 12 new throws per installation and froze the placement and registration memory. Each policy reset its inherited rally and latency state to the same checkpoint before every throw. The contextual comparison used three registered serves and nine rightward sweeps on A. It selected the rightward sweep on every start on B. Identical within-throw placement choices reproduced identical action tapes across policies.

| Installation | Placement | Legal serves | Legal rival replies | First legal follow-ups | Later legal returns on own serve | Points won on serve |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Registered | 10/12 | 8 | 3 | 1 | 5 |
| A | Fixed right sweep | 10/12 | 6 | 1 | 0 | 5 |
| A | Contextual | 9/12 | 6 | 1 | 0 | 4 |
| B | Registered | 12/12 | 6 | 2 | 2 | 8 |
| B | Fixed right sweep | 8/12 | 5 | 2 | 2 | 3 |
| B | Contextual | 8/12 | 5 | 2 | 2 | 3 |

Across the ten held starts where both registered and contextual serves drew legal replies, the registered player made four first follow-ups and the contextual player made three. The contextual player gained three wins and gave up nine on the same set of 24 throws. [Qualification A](strategy-qualification-a.json.gz) · [Qualification B](strategy-qualification-b.json.gz)

## Alternating-service games

Each policy started a new first-to-11 game. Both arms played ordinary, soft-racket, delayed and mixed conditions. The placement regression stayed frozen, and the player carried its registration, racket, rally and delay updates through every point. The controllers faced the same ordered point and disturbance schedule on each installation. The service role could diverge when one game had a let.

| Installation | Placement | Final score | Legal own serves | First legal follow-ups / legal rival replies | Later legal returns on own serve | Own-service points won | Own-service wins after a legal reply |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Registered | 6–11 | 7/9 | 3/7 | 2 | 4 | 4 |
| A | Fixed right sweep | 8–11 | 7/9 | 3/6 | 0 | 3 | 2 |
| A | Contextual | 9–11 | 8/10 | 2/6 | 2 | 4 | 2 |
| B | Registered | 6–11 | 7/8 | 2/5 | 2 | 4 | 2 |
| B | Fixed right sweep | 8–11 | 6/9 | 1/2 | 0 | 5 | 1 |
| B | Contextual | 8–11 | 6/9 | 1/2 | 0 | 5 | 1 |

On A, all three extra contextual points came on the rival's serve. On B, it won one more own-service point and one more rival-service point. It drew three fewer legal replies on its own serve. The B contextual and right-sweep players produced identical commands across all 19 game points. [Qualification A](strategy-qualification-a.json.gz) · [Qualification B](strategy-qualification-b.json.gz)

All 238 practice, held and match point tapes replayed with identical commands, events and scores. The two contextual checkpoints preserve placement practice, the body library, and registration and rally state from the completed games. [Experimental player A](strategy-player-a.json.gz) · [Experimental player B](strategy-player-b.json.gz)

Both contextual checkpoints also reproduced all 180 native Pong reference action tapes, with 41 points earned, 129 of 160 first returns and 195 of 206 later returns each. [Pong retention A](strategy-pong-a.json.gz) · [Pong retention B](strategy-pong-b.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) and set `LP_NUM_THREADS=1` for the EGL renderer. The two stages use separate point-seed streams. Keep the calibration seed fixed within each installation.

```sh
python -m scripts.experimental.tt10_strategy_trial /tmp/strategy-practice-a.json.gz \
  --seed 1736889021 --installation-seed 501137399 --calibration-seed 398512109 \
  --resume artifacts/tt9/precontact-registration-a-player.json.gz \
  --practice 9 --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt10_strategy_trial /tmp/strategy-qualification-a.json.gz \
  --seed 1736889021 --installation-seed 501137399 --calibration-seed 398512109 \
  --resume artifacts/tt9/precontact-registration-a-player.json.gz \
  --training /tmp/strategy-practice-a.json.gz --held 12 --game 55 \
  --probe-cache /tmp/tt8-cache --checkpoint /tmp/strategy-player-a.json.gz
```

For B, use point seed `2017089031`, installation seed `317557501`, calibration seed `83041723`, and the [B registration checkpoint](../tt9/precontact-registration-b-player.json.gz). The same script accepts `--verify` to replay a receipt's applied commands, events and scores. The qualification receipts also archive the RGB ball tracks and verify the computed placement evidence. To resume the saved experimental player, pass its checkpoint as `--resume` and omit `--training`.

```sh
python -m scripts.experimental.tt8_pong_retention /tmp/strategy-player-a.json.gz /tmp/strategy-pong-a.json.gz
```
