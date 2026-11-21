# Recall from the player's own table-tennis strokes

Recalling a prior stroke raised first legal follow-ups from 6/16 to 7/16 on matched incoming replies. Later legal returns stayed at six and reply points won fell from five to three. Recall added one first legal follow-up in a separate game and two later returns on common service-reply starts in a new installation. Both policies lost all three alternating-service games.

The repertoire contains ten distinct first follow-ups from eight TT10 receipts. The learner fitted short incoming and outgoing ball arcs from first-person stereo, estimated blade velocity from at least two measured poses, and kept the stroke only when it saw a coherent far-half bounce. The ten examples came from nine distinct point seeds. The referee subsequently confirmed all ten as legal returns. Replacing the archived referee and reward fields in a copy of a source receipt preserved its optical samples exactly and changed only the external audit. [TT10 receipts](../tt10/README.md) · [Player and optical extraction](../../scripts/experimental/tt11_stroke.py) · [Trial and independent audit](../../scripts/experimental/tt11_stroke_trial.py)

The player recognizes its own toss with the inherited service planner. The reply detector requires two consecutive inward movements on the far side before retrieval can begin. Near impact it selects the closest example by lateral position, height and two components of the incoming velocity. The servo keeps the new ball's intercept as its target, adjusts the example's forward pace and blade pitch for the changed flight, and adds a short velocity objective around contact. A rejection threshold leaves the inherited stroke in place when the library lacks a nearby example. Actual pitch, blade velocity and outgoing flight are estimated from the camera. The controller receives images, previous commands and the usual public point reward. Example collection, nearest-neighbor lookup and the existing bounded rate controller require no backpropagation.

## Matched incoming replies

Each established five-axis installation ran 20 fresh serve points with the inherited controller, carrying its earlier registration and normal updates across the sequence. Every legal serve that the opponent returned generated a fork of that same pre-point state. The 16 forks shared the requested and applied commands through the opponent's contact and up to the recall decision. The 12 decisions occurred 16 to 20 frames after the opponent struck. All 56 requested-action tapes replayed with the recorded physical events and outcomes. [Matched A](stroke-paired-a.json.gz) · [Matched B](stroke-paired-b.json.gz)

| Installation | Policy | Hit the reply | First legal follow-ups | Later legal returns | Reply points won |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Inherited | 7/8 | 3/8 | 2 | 3 |
| A | Recall | 7/8 | 3/8 | 2 | 2 |
| B | Inherited | 8/8 | 3/8 | 4 | 2 |
| B | Recall | 8/8 | 4/8 | 4 | 1 |

The extra legal follow-up on B began a two-return exchange that the player lost. The inherited stroke won that same point when the rival volleyed before the ball landed. Across the 12 activations, the median difference between requested and measured forward blade speed dropped from 0.0083 to 0.0066 metres per camera frame. For this audit the referee supplied the time window for measuring blade motion, which still came from stereo. [Matched A](stroke-paired-a.json.gz) · [Matched B](stroke-paired-b.json.gz)

## Alternating-service games

Both policies played new point schedules with service alternating and a two-point margin required at deuce. The recall player could add camera-confirmed strokes at the end of a point. All 111 full-game tapes replayed, and every recall activation followed a legal service reply. [Game A](stroke-game-a.json.gz) · [New five-axis games](stroke-game-offset.json.gz) · [New gantry games](stroke-game-gantry.json.gz)

| Installation | Policy | Score | Legal own serves | First legal follow-ups | Later legal returns on own serve | Reply points won | Total legal returns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five-axis A | Inherited | 4–11 | 2/7 | 0/1 | 0 | 0 | 4 |
| Five-axis A | Recall | 4–11 | 2/7 | 1/1 | 0 | 1 | 5 |
| New five-axis | Inherited | 16–18 | 14/18 | 5/8 | 5 | 3 | 23 |
| New five-axis | Recall | 11–13 | 8/12 | 4/4 | 6 | 3 | 15 |
| New gantry | Inherited | 0–11 | 0/5 | 0/0 | 0 | 0 | 1 |
| New gantry | Recall | 0–11 | 0/5 | 0/0 | 0 | 0 | 1 |

The new five-axis game with recall ended ten scored points sooner. In the four common starts that produced legal replies for both policies, each landed all four first follow-ups. Later legal returns rose from four to six, and wins on those same starts rose from two to three. Each five-axis checkpoint gained one camera-confirmed stroke. The gantry's service results gave it no recall opportunities. The three checkpoints restored their exact repertoires. [A checkpoint](stroke-a-player.json.gz) · [New five-axis checkpoint](stroke-offset-player.json.gz) · [Gantry checkpoint](stroke-gantry-player.json.gz)

## Reproduce

Set `LP_NUM_THREADS=1` and use the native stereo component library for consistent tracking. [Hall environment](../tt6/README.md)

```bash
archive_strokes=(artifacts/tt10/{cursor-reply-a,cursor-reply-b,cursor-reply-match-a,cursor-reply-match-b,window-paired-a,window-paired-b,window-game-offset,window-game-gantry}.json.gz)
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-paired-a.json.gz \
  --seed 1402850814 --limit 20 --reference artifacts/tt10/cursor-reply-a.json.gz \
  --resume artifacts/tt9/precontact-match-a-player.json.gz --archives "${archive_strokes[@]}" --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-paired-b.json.gz \
  --seed 1640624713 --limit 20 --reference artifacts/tt10/cursor-reply-b.json.gz \
  --resume artifacts/tt9/precontact-match-b-player.json.gz --archives "${archive_strokes[@]}" --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-game-offset.json.gz \
  --mode game --seed 1530682997 --installation-seed 883502071 --body offset --limit 52 \
  --archives "${archive_strokes[@]}" --probe-cache /tmp/tt8-cache --learn-strokes \
  --checkpoint /tmp/stroke-offset-player.json.gz
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-game-gantry.json.gz \
  --mode game --seed 1500752929 --installation-seed 780847903 --body gantry --limit 52 \
  --archives "${archive_strokes[@]}" --probe-cache /tmp/tt8-cache --learn-strokes \
  --checkpoint /tmp/stroke-gantry-player.json.gz
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-game-a.json.gz \
  --mode game --seed 1861430429 --reference artifacts/tt10/cursor-reply-a.json.gz --limit 52 \
  --resume artifacts/tt9/precontact-match-a-player.json.gz --archives "${archive_strokes[@]}" \
  --probe-cache /tmp/tt8-cache --learn-strokes --checkpoint /tmp/stroke-a-player.json.gz
python -m scripts.experimental.tt11_stroke_trial /tmp/stroke-game-gantry.json.gz --verify
```

Pass a saved player to `--resume` and omit `--archives` to recall its saved strokes on another schedule.
