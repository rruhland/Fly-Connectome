# Carrying match practice on the same arm

Across five fresh fixtures on the practiced five-axis arm, the standard player scored 37 points when it carried practice forward and 41 when it restarted from the original commissioned state. Restarting won one game. Carrying practice won none. Restarted scores varied from 2 to 12 points across fixtures. Keeping match experience changed the learner's score by one to four points on a given fixture after the first game. [Carried five-axis games](series-b-carry.json.gz) · [Restarted five-axis games](series-b-fresh.json.gz)

| Five-axis game | Carried score | Restarted score | First follow-ups carried | First follow-ups restarted | Sustained points carried | Sustained points restarted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12–14 | 12–14 | 2/2 | 2/2 | 2 | 2 |
| 2 | 7–11 | 6–11 | 1/2 | 1/2 | 1 | 2 |
| 3 | 9–11 | 11–7 | 1/2 | 3/4 | 3 | 6 |
| 4 | 6–11 | 10–12 | 3/4 | 3/3 | 3 | 4 |
| 5 | 3–11 | 2–11 | 2/3 | 1/4 | 3 | 2 |
| Total | 37–58 | 41–55 | 9/13 | 10/15 | 12 | 16 |

The five-axis player carrying practice landed eight later returns after its own serve and 55 legal returns overall. The restarted player landed 11 and 58 respectively. Carrying experience improved the score slightly on games two and five and lost ground on games three and four. These fixtures give little support for a reliable improvement from accumulating the standard updates on arm B. [Game-by-game comparison](series-b-comparison.json) · [Carried checkpoint](series-b-player.json.gz)

The carried five-axis checkpoint grew from 20 to 65 optical service corrections and reproduced all 180 native Pong action tapes, including 41 earned points. [Pong retention](series-b-pong.json.gz)

## Replication on a gantry

On the gantry, carrying practice added four points across the same five fixtures. Both policies won one game, on different fixtures. The carried player won 17–15 in game three, where restarting lost 9–11. Restarting won 11–8 in game four, where carrying practice lost 3–11. [Carried gantry games](series-gantry-carry.json.gz) · [Restarted gantry games](series-gantry-fresh.json.gz)

| Gantry game | Carried score | Restarted score | First follow-ups carried | First follow-ups restarted | Sustained points carried | Sustained points restarted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5–11 | 5–11 | 2/2 | 2/2 | 3 | 3 |
| 2 | 6–11 | 1–11 | 1/3 | 0/1 | 4 | 1 |
| 3 | 17–15 | 9–11 | 3/7 | 1/3 | 7 | 4 |
| 4 | 3–11 | 11–8 | 2/4 | 3/3 | 3 | 5 |
| 5 | 6–11 | 7–11 | 4/4 | 3/3 | 5 | 5 |
| Total | 37–59 | 33–52 | 12/20 | 9/12 | 22 | 18 |

The carried gantry player landed ten legal returns after the first follow-up and 70 legal returns overall. The restarted player landed three and 57 respectively. Across the two stations both strategies scored 74 points and sustained 34 exchanges per branch. Carrying won one of ten games and restarting won two. Fixture three favored restarting on the five-axis arm and carrying on the gantry. Four of the eight comparisons after the identical first games favored carrying by points. [Gantry game-by-game comparison](series-gantry-comparison.json) · [Carried gantry checkpoint](series-gantry-player.json.gz)

## Same physical station and same starting player

The comparison used the standard five-axis registered serve and fixed-depth receiver from the practiced B checkpoint. Both branches updated the same RGB-derived timing, response, bounce, service and score memories within games. The carried branch kept those updates across games. The restarted branch read the shared post-commission checkpoint from disk at every game boundary. Each station commissioned its optics and arm once before the branch split. Five-axis B recalled its arm map after one probe. The gantry fitted its map from eight probes. The saved background, apertures and body model were reused across all ten games at each site.

Both branches covered the same five fresh fixture seeds at each site, using identical ordered starts, body wiring, camera pose and ordinary, softened, delayed and mixed disturbances. The opponent served first and the referee rotated the service every two points until deuce and every point thereafter. Games went to at least 11 points with a two-point lead. Scores and lets could change which side served on a later shared start. A sustained point requires at least two legal player returns on rival service or a legal player follow-up on own service. Legal first follow-ups count only replies that landed legally after an own serve.

The first game on each station reproduced the complete point records across branches. All 373 point tapes across the 20 complete games were physically replayed. Reloading the boundary after B's first game into a new arena reproduced the next three live points, including requested and applied commands, RGB observations and referee outcomes. The start checkpoint retained its original digest throughout both branches. [B start](series-b-start.json.gz) · [Gantry start](series-gantry-start.json.gz) · [Boundary check](series-b-resume-check.json)

## Reproduce

Use the [hall environment](../tt6/README.md). To run the gantry replication, add `--body gantry --installation-seed 741720081` to the preparation command and use a different output directory. The final player file can seed another series on its saved installation without any new commissioning. The standard TT8 service runner can use the same portable player on a new station.

```sh
python -m scripts.experimental.tt10_series_trial /tmp/series-b-start.json.gz --prepare --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt10_series_trial /tmp/series-b-carry.json.gz \
  --seed 694216325 --resume /tmp/series-b-start.json.gz --checkpoint /tmp/series-b-player.json.gz
python -m scripts.experimental.tt10_series_trial /tmp/series-b-fresh.json.gz \
  --seed 694216325 --mode fresh --resume /tmp/series-b-start.json.gz
python -m scripts.experimental.tt10_series_trial /tmp/series-b-comparison.json \
  --compare /tmp/series-b-carry.json.gz /tmp/series-b-fresh.json.gz
python -m scripts.experimental.tt10_series_resume /tmp/series-b-resume-check.json \
  artifacts/tt10/series-b-after-one.json.gz /tmp/series-b-carry.json.gz
python -m scripts.experimental.tt8_pong_retention /tmp/series-b-player.json.gz /tmp/series-b-pong.json.gz
```
