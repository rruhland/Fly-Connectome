# Cursor-trained body map after a service reply

Switching to the cursor-trained motor predictor after the opponent struck raised first legal follow-ups from 5/12 to 7/12 and later legal returns from 10 to 15. Both first-follow-up gains came on installation A. The branches preserved the serve and the commands through the opponent's contact on every point. [Forks A](cursor-reply-a.json.gz) · [Forks B](cursor-reply-b.json.gz)

The full cursor-to-table games used the updated motor geometry before every point. That allowed the serve, the trajectory the opponent received, and the player history available on later points to change. In this comparison, the original TT9 player carried its visual, registration and reward updates throughout the baseline games. Each branch began from that player's exact pre-point state. It installed the cursor-trained linear translations in the predictive servo after TT9's RGB detector recognized the opponent's reply. Referee contacts defined the evaluated set and verified the shared action prefix. The intervention preserved the body map's pitch, origin, quadratic terms and actuator dynamics, and kept the existing strike planner, delay estimate and command queue. [Trial](../../scripts/experimental/tt10_cursor_reply.py) · [RGB reply detector](../../scripts/experimental/tt9_recovery.py) · [Cursor fit A](target-back-a-fit.json) · [Cursor fit B](target-back-b-fit.json)

| Installation | Predictor | First legal follow-ups | Later legal returns | Total legal returns | Points with at least two legal returns | Points won |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Original | 2/7 | 2 | 4 | 2 | 3 |
| A | Cursor fit after reply | 4/7 | 5 | 9 | 4 | 5 |
| B | Original | 3/5 | 8 | 11 | 3 | 2 |
| B | Cursor fit after reply | 3/5 | 10 | 13 | 3 | 1 |

The RGB gate switched maps on 11 of the 12 forks, two to six camera frames after the rival contact and before the learner's first follow-up strike. The undetected reply reproduced the original action tape throughout. Both new legal follow-ups on A turned original own-half faults into won exchanges, with the learner making contact on the same frame in each pair. B returned the same number of first follow-ups, made two more later returns, and lost one previously won point. These forks estimate the map's effect among the original games' legal service replies. [Forks A](cursor-reply-a.json.gz) · [Forks B](cursor-reply-b.json.gz)

The replays reproduced the requested-action hashes, applied commands and scorer events of all 37 baseline game points. Each branch matched the requested and applied commands up through the opponent contact and until RGB activation. All 24 fork tapes replayed with identical physical events and scores.

## Prospective alternating-service games

Each installation played a fresh first-to-11 schedule twice, with and without the predictor switch. Both players shared the original full checkpoint and exact RGB motor commissioning. The paired receipts certified the saved translation coefficients. The camera gate ran on every point, including serve faults and rounds the opponent served. Each player carried its own visual, registration and reward updates through the complete game. The point seeds were disjoint from the earlier replay, placement, and full-transfer tests.

| Installation | Predictor | Score | Legal own serves | First legal follow-ups | Later legal returns on own serve | Total legal returns | Own-service wins after a legal reply |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Original | 5–11 | 7/8 | 3/7 | 1 | 8 | 3 |
| A | Cursor fit after reply | 5–11 | 7/8 | 3/7 | 3 | 10 | 3 |
| B | Original | 5–11 | 6/8 | 1/4 | 3 | 12 | 1 |
| B | Cursor fit after reply | 6–11 | 6/8 | 1/3 | 3 | 9 | 1 |

The fresh games produced four first follow-ups and four own-service wins after a legal reply under either policy. The gate added two later legal returns on own serves, both on A. B gained one whole-game point and made three fewer total legal returns. All ten map changes followed the actual rival strike by three to five camera frames. All 65 point tapes replayed with the recorded actions, events and scores. Both saved players restored their exact gated maps and original default maps. [Match A](cursor-reply-match-a.json.gz) · [Match B](cursor-reply-match-b.json.gz) · [Player A](cursor-reply-match-a-player.json.gz) · [Player B](cursor-reply-match-b-player.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) with `LP_NUM_THREADS=1`. Substitute `b` for `a` on all four input paths and the output path for the second installation.

```sh
python -m scripts.experimental.tt10_cursor_reply /tmp/cursor-reply-a.json.gz \
  --baseline artifacts/tt10/target-back-a-baseline.json.gz \
  --trained artifacts/tt10/target-back-a-trained.json.gz \
  --source artifacts/tt9/precontact-match-a-player.json.gz \
  --updated artifacts/tt10/target-back-a-player.json.gz --probe-cache /tmp/tt8-cache
```

Use `--verify` with a saved receipt to replay its forked commands and check the action prefixes.

```sh
python -m scripts.experimental.tt10_cursor_reply_match /tmp/cursor-reply-match-a.json.gz \
  --reference artifacts/tt10/cursor-reply-a.json.gz \
  --source artifacts/tt9/precontact-match-a-player.json.gz \
  --updated artifacts/tt10/target-back-a-player.json.gz --seed 629456259 \
  --probe-cache /tmp/tt8-cache --checkpoint /tmp/cursor-reply-match-a-player.json.gz
```

Use seed `824371753` and the B files for the second match comparison. A saved match player contains the original default body model and the gated replacement. Pass that player as `--source` and omit `--updated` to resume it on another point schedule. [Prospective trial](../../scripts/experimental/tt10_cursor_reply_match.py)
