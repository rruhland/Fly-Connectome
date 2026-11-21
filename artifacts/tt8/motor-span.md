# Keeping Pong and the arm's learned scores in one checkpoint

One table-tennis player completed three alternating-service games, played 32 new native Pong practice points, and returned for another table-tennis game. It continued to update its shared and local score estimates throughout this sequence. On a separate frozen 180-point Pong test, the trained player reproduced every action taken before those three hall games.

The older first-person mini player already had separate score evidence for movement along one observed direction and movement spanning several directions. Its port to the realistic hall had only imported the common totals. After three games that common table favored interception relation 1. The native Pong experience favored relation 0. Using a single ranking lost five native earned points on the reference course. A separate public-score refresher found relation 0 on its first set of practice starts, but an independent score test did not confirm it. Using the refresher's single new ranking also cost the arm four points on its return fixture. [Refresher experiment](native-refresher.md)

The integrated player restores the differentiated score banks from the earlier trained checkpoint. A source bank that duplicates the common evidence starts from the current shared totals when it is encountered again. During arm play, a motor model fitted from commanded movement and calibrated stereo estimates the span of reachable movement. In native Pong, observed paddle movement following trial commands identifies a single axis. Both use the earlier mini player's singular-value criterion. A scored action updates the common bank and the bank for the observed span. No game name, ground-truth joint axis, or preferred relation index enters the choice. The spatial hypotheses, optical tracker and rule for grouping movement spans are designed components. [Source mini player](../tt4/mini-first-person-player.json.gz) · [Hall entry checkpoint](span-before-games.json.gz)

| Chronological stage | Outcome | Comparator |
| --- | --- | --- |
| New gantry | Won 11–6 | All 18 point tapes matched the original full game |
| New five-axis arm | Lost 3–11 | All 14 point tapes matched |
| Return to first gantry | Won 11–8 | All 21 point tapes matched |
| Frozen native Pong after the three games | 129/160 openings, 195/206 later returns, 41 earned points | All 180 tapes matched pre-hall Pong. The single score bank earned 36 points |
| 32 disjoint native Pong practice points | All four numeric hypotheses tried eight times and credited in the one-axis and common banks | Multi-axis score history stayed intact |
| Frozen native Pong after the new practice | 129/160 openings, 195/206 later returns, 41 earned | All 180 tapes matched |
| Back to gantry on new serves | Lost 10–12, landed 23 legal returns | All 23 point tapes matched the arm-only continuation on the same fixture |

A final frozen Pong evaluation after the fourth arm game again matched all 180 pre-hall action tapes. [Final native game records](span-native-final.json.gz)

The three arm games used the exact same probes and hidden physical conditions as the original sequence. The restored profiles kept the arm's real adaptation to its own scores. In particular, the multi-axis bank changed its preferred relation during the third game. Directly replacing the single score ranking in the refresher control produced a 6–11 game on the fourth fixture. Using the camera-estimated response preserved the existing 10–12 performance without requiring extra table training.

An effect of prior experience on a new game remains a separate question. This test establishes simultaneous retention and continued updating for two distinguishable motor spans. Both test arms occupied the same multi-axis bank, and the persistent five-axis service failure remains.

[Game one](span-gantry.json.gz) · [Game two](span-offset.json.gz) · [Game three](span-revisit.json.gz) · [Pong before further practice](span-native-before-practice.json.gz) · [Chronological native updates](span-pong-practice.json.gz) · [Pong after practice](span-native-after-practice.json.gz) · [Fourth game](span-after-pong-game.json.gz) · [Resumable full player](span-after-four.json.gz)

## Reproduce the switch between games

The first three games use the installations, full-state continuation order, and service source from the [alternating-service experiment](README.md), starting the first game with the output of the first command. The saved checkpoint after the third game is provided for reproducing the switch directly.

```bash
python -m scripts.experimental.tt8_motor_span artifacts/tt6/continual-player.json.gz /tmp/span-entry.json.gz
python -m scripts.experimental.tt8_pong_retention artifacts/tt8/span-after-three.json.gz /tmp/native-before.json.gz
python -m scripts.experimental.tt8_span_practice artifacts/tt8/span-after-three.json.gz /tmp/after-pong.json.gz /tmp/pong-practice.json.gz
python -m scripts.experimental.tt8_service_trial /tmp/fourth-game.json.gz --resume /tmp/after-pong.json.gz \
  --seed 327046119 --installation-seed 128754010 --limit 35 --first-server -1 \
  --phases ordinary mixed soft delay --no-compiled-geometry --checkpoint /tmp/after-four.json.gz
```
