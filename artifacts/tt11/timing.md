# Choosing actuator timing from fast first-person blade motion

## Full games

Across two fixture schedules on four installations, the RGB timing selector earned 64 points and landed 100 legal returns. Fixed ordinary timing earned 53 points on 87 returns, and fixed cursor-fit timing earned 61 on 91 returns. The selector and ordinary timing won two games each. Fixed cursor timing won one. The selector had the higher point margin in five of the eight comparisons with its unchosen model, and in three of four comparisons after summing by installation.

| Installation | Ordinary timing | Cursor-fit timing | Selected timing | Sustained points, ordinary / cursor / selected |
| --- | --- | --- | --- | ---: |
| A | 4–11, 9–11 | 6–11, 5–11 | 4–11, 9–11 | 6 / 8 / 6 |
| B | 11–9, 2–11 | 5–11, 5–11 | 5–11, 5–11 | 1 / 5 / 5 |
| C | 7–11, 11–9 | 10–12, 7–11 | 7–11, 11–9 | 8 / 6 / 8 |
| D | 1–11, 8–11 | 12–14, 11–7 | 12–14, 11–7 | 3 / 7 / 7 |
| Total | 53–84 | 61–88 | 64–85 | 18 / 26 / 26 |

On new C and D the selector went 41–41 and won two of four games. Ordinary timing went 27–42 and won one. Cursor-fit timing went 40–44 and won one. The largest gain came on D, where the selector reached deuce in the first game and won the second. On familiar B the selector sustained more exchanges, while ordinary timing won the first game and earned more points across both. [A comparison](timing-a-comparison.json.gz) · [B comparison](timing-b-comparison.json.gz) · [C comparison](timing-c-comparison.json.gz) · [D comparison](timing-d-comparison.json.gz)

### Additional fixture on C and D

The frozen selector used ordinary timing on C and cursor-fit timing on D for a third fixture with disjoint point starts. On C both fixed models won 11–8. The chosen model landed 13 legal returns and sustained five points in 19, versus nine returns and two sustained. On D the chosen model won 11–4, where ordinary timing lost 10–12. The shorter win had 11 legal returns and three sustained points in 15. The longer loss had 16 returns and five sustained in 22. [C ordinary](timing-c-inherited-replication.json.gz) · [C cursor fit](timing-c-cursor-replication.json.gz) · [D ordinary](timing-d-inherited-replication.json.gz) · [D cursor fit](timing-d-cursor-replication.json.gz)

| Cohort and measure | Ordinary timing | Cursor-fit timing | Selected timing |
| --- | ---: | ---: | ---: |
| Six new-installation games, points | 48–62 | 62–56 | 63–53 |
| Wins | 2 | 3 | 4 |
| Legal returns | 76 | 70 | 81 |
| Sustained points | 21 | 18 | 23 |
| All ten games, points | 74–104 | 83–100 | 86–97 |
| Wins | 3 | 3 | 4 |
| Legal returns | 116 | 111 | 124 |
| Sustained points | 28 | 31 | 34 |

## Prefixture measurement

The player compared the ordinary actuator response with the response fitted on 16 cursor targets at the same installation. Both models used the commissioned translation and pitch map. The ordinary response on A and B came from the TT10 starts. C and D had new command wirings and hall object appearances. Each required a new motor map. All four installations used the same five-axis mechanism design and calibrated stereo viewpoint in MuJoCo. [TT10 reverse transfer](../tt10/target-return.md) · [Common game runner](../../scripts/experimental/tt10_series_trial.py)

Each installation ran eight dry swings from varied blade heights, lateral positions and pitches, with three- and five-frame forward drives. The commands came from the commissioned local map and were shared by both timing models. The decision used the blade position and pitch reconstructed from player-eye stereo RGB and each model's response to the issued commands. It scored two-frame changes near the forward drive, weighted depth by 1.5 and converted pitch error to a displacement using an 8 cm lever arm. The timing model with lower error on the common visible pairs played the games. The run required at least four visible swings, 16 pairs and a median peak observed forward speed of 0.8 m/s. [Probe and scoring code](../../scripts/experimental/tt11_timing_select.py)

| Installation | Status | Ordinary error, mm | Cursor-fit error, mm | Median peak forward speed, m/s | Choice |
| --- | --- | ---: | ---: | ---: | --- |
| A | Familiar | 19.63 | 20.49 | 1.74 | Ordinary |
| B | Familiar | 16.91 | 16.73 | 1.73 | Cursor fit |
| C | Newly commissioned | 13.80 | 14.74 | 1.69 | Ordinary |
| D | Newly commissioned | 17.80 | 17.67 | 1.71 | Cursor fit |

Each site contributed 48 usable two-frame pairs. The aggregate error gaps ranged from 0.13 to 0.94 mm. The choice was the same within each of the two drive paces and survived individually omitting each stroke. The forward-speed error averaged across strokes changed by less than 0.25 cm/s at all four sites. On A and B the resulting choices opposed both historical TT10 game comparisons. [A dry swings](timing-a-probes.json.gz) · [B dry swings](timing-b-probes.json.gz) · [C dry swings](timing-c-probes.json.gz) · [D dry swings](timing-d-probes.json.gz)

## Protocol and cost

The selection happened before play. Each fixed model and the selector started from the same library, tracker reference, registration, latency belief and random state, with the chosen timing arrays as the only change affecting actions. All three branches shared the prefixture probes. Each site played two fresh fixture schedules in ordinary, soft, delayed and mixed conditions. Each game restarted at its common pregame checkpoint. Service alternated under the usual first-to-11 rules with a two-point lead. A sustained point required at least two legal returns on rival service or a legal return following an own serve.

The dry selection took 168 commands and 176 stereo pairs per installation, or 5.6 simulated seconds of movement. Fitting the cursor candidate used 384 commands on 16 reaching targets, or 12.8 simulated seconds, per installation. The familiar starts used 34 commissioning commands each. The new starts used 272 each, or 9.1 simulated seconds. Running the timing choice alone from a commissioned model and an available cursor fit requires only the dry-swing budget. Replaying the dry strokes for the audit used the same budget a second time. [Trial and audit runner](../../scripts/experimental/tt11_timing_trial.py)

## Checkpoint and replay

All eight independently played selector games matched their chosen fixed branch point for point. The dry replay reproduced the stereo frame hashes and reconstructed blade tapes on all 32 swings. Reloading each selected starting checkpoint reproduced the first three point records from live RGB. The game audit replayed all 439 primary point tapes and 75 additional point tapes through the physical arena, matching applied commands, referee events and scores. The additional fixtures used the archived selected branch for each site. [Combined audit](timing-family-audit.json.gz)

The postgame checkpoints preserve the selected timing, both candidate curves, the dry-stroke decision and the shared player state after the second fixture. The standard series runner can continue from any of these on its saved installation. The selected B player reproduced all 180 native Pong action tapes and 41 earned points. [A player](timing-a-selected-player.json.gz) · [B player](timing-b-selected-player.json.gz) · [C player](timing-c-selected-player.json.gz) · [D player](timing-d-selected-player.json.gz) · [Pong retention](timing-b-pong.json.gz) · [Series runner](../../scripts/experimental/tt10_series_trial.py)

## Reproduction

Run from the repository root in the [hall environment](../tt6/README.md). The four sites each write a separate receipt and portable checkpoint.

```sh
for site in a b c d; do
  python -m scripts.experimental.tt11_timing_trial all "$site" /tmp/tt11-timing --games 2
done
```

Repeat the independent C and D fixtures from the frozen starts with a different series seed. The selected comparison uses each site's already chosen fixed branch.

```sh
for site in c d; do
  for model in inherited cursor; do
    python -m scripts.experimental.tt10_series_trial \
      "/tmp/tt11-timing/timing-$site-$model-replication.json.gz" \
      --resume "/tmp/tt11-timing/timing-$site-$model-start.json.gz" \
      --seed 182395613 --games 1 --mode fresh
  done
done
```
