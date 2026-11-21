# Changing actuator timing during a table-tennis game

Two five-axis installations each played an adaptation fixture and a disjoint qualification fixture under regulation first-to-11 scoring with a two-point lead. Each fixture compared three branches on the same seeded toss and condition schedule. The first fixture started every branch from the same TT11 checkpoint on that installation. The two candidates were separated by 0.94 mm of dry-stroke prediction error on C and 0.13 mm on D. The next fixture started every branch from the same player saved after the adaptive game. That checkpoint includes the shared registration and rally learning, so the qualification comparison isolates the effect of applying the retained timing choices. [TT11 dry-stroke comparison](../tt11/timing.md) · [C game receipts](online-c-games.json.gz) · [D game receipts](online-d-games.json.gz)

The timing choice reads the first usable stereo sighting of the ball in frames 0–7. A ball deeper than 0.35 m on the rival side denotes receiving. A ball behind the learner's near edge, with depth below −1.10 m, denotes own service. Both coordinates are in the commissioned stereo frame. Every branch uses the TT11 dry-stroke choice until that sighting. A choice of the alternate settling curve reconstructs the motor estimator from the opening commands and the same RGB blade sightings, then uses that curve for the point. When neither optical condition holds before frame 8, the dry-stroke choice is used for the point without updating timing preferences. [Player and selection rule](../../scripts/experimental/tt12_online_timing.py)

The adaptive branch tried each timing curve twice per role, starting with the dry-stroke choice. After these trials it selected by smoothed point wins with a small exploration bonus. Wins and losses update a separate estimate for each timing curve and optical role. A let or timeout counts as an attempt and contributes no win or loss. Qualification chose the higher stored reward estimate for each role and held these choices through the game. The fixed branches used either inherited or cursor-fit timing throughout the optically recognized point. All branches had the same commissioning, zero extra practice points, a cap of 65 point attempts, and a cap of 420 commands per attempt. Longer exchanges and games used more of this shared budget. [Trial and replay audit](../../scripts/experimental/tt12_online_timing_trial.py)

## Revision during play

Across the two adaptation fixtures, the adaptive branch won 25 of 50 points and sustained eight, compared with 19 of 42 points and seven sustained with inherited timing and 14 of 36 points and two sustained with cursor-fit timing.

| Installation | Adaptive | Fixed inherited | Fixed cursor fit |
| --- | ---: | ---: | ---: |
| C | 14–12 | 9–11 | 6–11 |
| D | 11–13 | 10–12 | 8–11 |

| Adaptation performance, both installations | Adaptive | Fixed inherited | Fixed cursor fit |
| --- | ---: | ---: | ---: |
| Points won on rival service | 12/25 | 11/21 | 6/19 |
| Legal first returns on rival service | 19/25 | 14/21 | 12/19 |
| Points won on own service | 13/25 | 8/21 | 8/17 |
| Legal own serves | 16/25 | 10/21 | 9/17 |
| Legal follow-ups / replies to legal serves | 3/5 | 2/3 | 0/1 |
| Commands | 3,826 | 2,785 | 2,154 |

Both adaptive players switched their receiving timing away from the dry-stroke choice at the ninth point. C chose cursor-fit timing on eight of nine subsequent receiving points and D chose inherited timing on seven of eight. Both players kept the dry-stroke preference for their own service when the game ended. The adaptation games recognized the correct role on all 50 points. [C after adaptation](online-c-learned.json.gz) · [D after adaptation](online-d-learned.json.gz)

The adaptive game and the dry-stroke fixed branch each won 2 of 6 receiving points on the identical late starts they both reached, at each installation. C's adaptive branch won 3 of 6 matched late points on own service, versus 1 of 6 with its dry-stroke timing fixed. D's adaptive and dry-stroke branches each won 3 of 5 matched late points on own service. These matched comparisons use point attempts after the first eight and require the same toss seed, disturbance and service role in both branches.

## Holding the learned choices on new points

| Installation | Retained by role | Fixed inherited | Fixed cursor fit |
| --- | ---: | ---: | ---: |
| C | 8–11 | 6–11 | 11–9 |
| D | 16–14 | 8–11 | 8–11 |

| Held performance, both installations | Retained by role | Fixed inherited | Fixed cursor fit |
| --- | ---: | ---: | ---: |
| Points won / attempts | 24/50 | 14/37 | 19/39 |
| Points won on rival service | 12/25 | 9/19 | 12/20 |
| Legal first returns on rival service | 15/25 | 14/19 | 14/20 |
| Points won on own service | 12/25 | 5/18 | 7/19 |
| Legal own serves | 17/25 | 9/18 | 12/19 |
| Legal follow-ups / replies to legal serves | 4/8 | 4/6 | 4/7 |
| Sustained points / attempts | 7/50 | 5/37 | 7/39 |
| Commands | 3,402 | 2,655 | 2,899 |

The stored choices beat the dry-stroke fixed branch in both held scores and beat both fixed branches on D. The fixed cursor curve won the held fixture on C and had the higher overall point win rate across the two held fixtures. On exactly matched receiving starts against each site's dry-stroke branch, the two policies won 4/9 each on C and 6/10 each on D. The retained policy and dry-stroke control used the same curve on own service at each installation. Changes in the receiving exchange can affect later play through the shared in-game learning and through differences in which serves occur. The direct performance benefit of changing the receiving curve remains uncertain.

## Checkpoints and replay

The stored timing memory contains attempts, scored trials and wins for each combination of optical role and timing curve. Reloading each learned checkpoint from disk reproduced the first four point records of the next game from live RGB. All 254 point tapes replayed their applied commands, referee events and complete scores, covering 17,721 commands. The audit reconstructed every timing decision from the chronological role observations and point rewards. The optical role agreed with the referee on all 254 points across the six games per installation. [C after qualification](online-c-player.json.gz) · [D after qualification](online-d-player.json.gz) · [Full adaptive C game](online-c-adaptation.mp4)

The D checkpoint after qualification preserved all 180 native Pong action tapes against the reference schedule and again earned 41 points. [Pong retention receipt](online-d-pong.json.gz)

## Reproduction

Run in the [EGL table-tennis environment](../tt6/README.md) from the repository root.

```sh
for site in c d; do
  python -m scripts.experimental.tt12_online_timing_trial "$site" /tmp/tt12-online
  python -m scripts.experimental.tt12_online_timing_trial "$site" /tmp/tt12-online --verify
done
python -m scripts.experimental.tt8_service_replay \
  /tmp/tt12-online/online-c-games.json.gz /tmp/tt12-online/online-c-adaptation.mp4 --game 1
python -m scripts.experimental.tt8_pong_retention \
  /tmp/tt12-online/online-d-player.json.gz /tmp/tt12-online/online-d-pong.json.gz
```
