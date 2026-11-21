# Choosing which motor experience to reuse

A player used eight dry swings and first-person stereo to choose between two estimates of its actuator timing. One came from table tennis and one from playing a cursor game on the same arm. On two new installations the visual choice won four of six alternating-service games. Always using the table-tennis estimate won two, and always using the cursor estimate won three. These were two new motor wirings and scene appearances of the same simulated five-axis design. Each fixture included normal, softened, noisy and delayed points. [Watch the full 11–7 game](timing-d-full-game.mp4) · [Methods and qualifications](timing.md)

| On the two new installations | Game wins / 6 | Points scored | Legal returns | Sustained rallies |
|---|---:|---:|---:|---:|
| Choose timing from first-person stereo | 4 | 63 | 81 | 23 |
| Always use table-tennis timing | 2 | 48 | 76 | 21 |
| Always use cursor timing | 3 | 62 | 70 | 18 |

Across all ten fixtures, including four games on previously studied installations, the visual choice won four games and both fixed alternatives won three. Its 34 sustained rallies compare with 28 and 31 for the fixed alternatives. The third fixture on each new installation was a disjoint replication with the timing choice frozen. On C, both alternatives won 11–8, with 13 legal returns for the selected model versus nine for the other model. On D, the selected model won 11–4 and the other model lost 10–12, although the other model landed more returns in its longer game. A sustained rally counts two legal player returns on rival service or a legal follow-up on own service. These totals are sensitive to the length of each game. [Comparison audit](timing-family-audit.json.gz) · [Replicated D player](timing-d-replication-player.json.gz) · [Native Pong after replication](table-pong-retention.json.gz)

## Continuing into the next game

The comparisons above started each fixture from the common commissioned state and allowed learning within the game. We also took the winning checkpoints immediately after fixture two and continued directly into fixture three. D went from an 11–7 win to an 11–3 win, versus 11–4 if restarted before fixture three. C went from an 11–9 win to a 9–11 loss, versus an 11–8 win if restarted. C made 16 legal returns when carried versus 13 after restarting, and sustained five rallies in either case. D made 10 legal returns and sustained two rallies when carried, versus 11 and three after restarting. The D player retained all 180 native Pong action tapes and all 41 earned points after its two consecutive wins. Carrying the online updates between table-tennis games has yet to show a consistent advantage. [C game](carried-c-game.json.gz) · [D game](carried-d-game.json.gz) · [Continuous D checkpoint](carried-d-player.json.gz) · [Pong after consecutive games](carried-d-pong.json.gz)

## Which findings transferred

- A higher-rate arm controller spent 46% of play within 4.5 cm of a moving 3D goal across familiar and randomly rotated camera axes, versus 20% under the slower controller on the same courses. A separate high-rate learner with identical initial probes benefited from the learned table-tennis map on all three installations in both view conditions. When the camera moved during play, transporting previously learned visual evidence yielded 493 on-target frames after the changes versus 264 for promptly re-learning it. These are projected, color-coded markers and a supplied tracking objective. [Tracking, change detection and controls](pursuit.md)
- The average actuator response that worked best for smooth pursuit landed just 1/24 serves in four complete table-tennis games. Each task rewards a different part of the arm's response. [Return-to-table trials](pursuit.md)
- Retrieving the player's own successful strokes raised first legal follow-ups from 6/16 to 7/16 on matched incoming replies, while dropping reply wins from five to three. The retrieval player and its baseline lost all three independent complete games, including on the gantry where neither landed a serve. [Optical stroke memory](stroke.md)

## Reproduce the continued games and video

Use the [hall environment](../tt6/README.md) from the repository root. Repeat the first command with the C files to reproduce C's carried game.

```sh
python -m scripts.experimental.tt10_series_trial /tmp/carried-d.json.gz --resume artifacts/tt11/timing-d-selected-player.json.gz --seed 182395613 --games 1
python -m scripts.experimental.tt8_service_replay artifacts/tt11/timing-d-cursor-games.json.gz /tmp/d.mp4 --game 2
```
