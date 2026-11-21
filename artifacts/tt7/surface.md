# Changing table bounce

The same RGB hall and table geometry were used throughout. The evaluator changed the table's MuJoCo contact damping from 24 to 65, 12, 90, then back to 24, with blocks of three, four, four, three, and two receiving points. The player received stereo RGB, its previous motor commands, and the public point reward. Both members of a comparison started from the same saved player and used the same installations and serve seeds.

The observer fitted separate descending and rising arcs from at least three recent stereo centers on each side of a near-table reversal. The fit estimated where the arcs met and used the ratio of their vertical speeds there. This avoids converting pixel-height bias or contact compression into a velocity error. It used the known tabletop footprint, the standard gravity prior, and a short-window drag approximation. The live receiver used a recency-weighted history to predict the next rebound and moved the strike forward when the rebound and predicted contact were both low. The read-only receiver recorded the same optical evidence while using the original flight and stroke controller.

## Optical measurements

In the first 32 read-only points, all 27 visual samples were within one camera frame of an evaluator-measured near-side bounce. The observer captured 27 of 48 approaching bounces with incoming vertical and depth speeds each above 1 m/s in magnitude and with a recorded separation. It saw only four of 15 lively bounces, where there was often little time between the rebound and racket contact.

| Table | Physical median ratio | Optical median ratio | Matched optical samples | Mean absolute ratio error |
| --- | ---: | ---: | ---: | ---: |
| Lively | .940 | .916 | 4 | .036 |
| Regular | .881 | .872 | 7 | .031 |
| Muted | .700 | .672 | 10 | .062 |
| Low | .664 | .651 | 6 | .032 |

The physical ratios were calculated from velocity just before table contact and at separation. Those events were used only for scoring. The initial live variant advanced the strike whenever its carried rebound estimate was low. It produced 24 legal returns and won 13 points, compared with 26 returns and 12 wins for the read-only player. The live gantry lost both points on the switch back to regular and earned zero returns there, compared with three read-only returns over those two points. A low carried estimate still moved the live player forward before it observed recovery. The revised reach rule gates the advance on the predicted ball height at the normal strike plane.

## Matched playing results

| Serve seed and arm | Legal returns, read-only / live | Points with a legal return, read-only / live | Wins, read-only / live |
| --- | ---: | ---: | ---: |
| 30978113 gantry | 18 / 17 | 13 / 11 | 5 / 6 |
| 30978113 offset | 8 / 12 | 8 / 10 | 7 / 7 |
| 54270819 gantry | 19 / 9 | 13 / 8 | 4 / 6 |
| Total across 48 matched points | 45 / 38 | 34 / 29 | 16 / 19 |

The revised learner earned one return on the original offset's low table, where read-only earned zero. Both players earned one low-table return on the confirmation seed. The largest loss was on the confirmation seed's switch from muted to lively. It took until the third lively point for the live player to register a bounce, and the carried ratio stayed near .67 on the first two. Live delivered one legal return in that block and read-only delivered eight. The changing surface was visible in the estimated bounce ratios, but applying the carried estimate did not yield a repeatable gain in legal returns.

## Run and replay

```bash
for condition in live read-only; do
  extra=()
  if [ "$condition" = read-only ]; then extra=(--read-only); fi
  python -m scripts.experimental.tt7_surface_trial "/tmp/surface-$condition.json.gz" \
    --seed 30978113 --course gantry offset --predictive --cached-render --reflections \
    --resume artifacts/tt6/continual-player.json.gz "${extra[@]}"
done
python -m scripts.experimental.tt6_replay artifacts/tt7/surface-read-only.json.gz
python -m scripts.experimental.tt6_replay artifacts/tt7/surface-live.json.gz
python -m scripts.experimental.tt6_replay artifacts/tt7/surface-live.json.gz \
  --visit 1 --point 11 --movie /tmp/surface-low-return.mp4
```

All five command tapes replayed with identical applied actions, events, and point results. The confirmation uses seed `54270819` and course `gantry` with the same point schedule. [Read-only receipt](surface-read-only.json.gz) · [Initial live receipt](surface-live-first.json.gz) · [Revised live receipt](surface-live.json.gz) · [Confirmation read-only receipt](surface-confirm-read-only.json.gz) · [Confirmation live receipt](surface-confirm-live.json.gz) · [Low-table return replay](surface-low-return.mp4)
