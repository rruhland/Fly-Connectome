# What later table-tennis practice can add

The 42-point receiver had a reusable advantage on unfamiliar gantry and offset arms. The next 42 varied points identified three more physical installations. The standard checkpoint also overwrote its recent stroke speed, delay posterior and bounce buffer and continued changing shot preferences. It returned fewer held starts at 84 points than at 42. The memory split tested here keeps the previously qualified portable state and appends later body fits. The receiving player only recalls a fit after its next stereo probe predicts the visually observed blade motion. Point play and the ordinary visual and public-score updates then proceed as usual. [Earlier checkpoint](../tt6/continual-player.json.gz) · [Later checkpoint](../tt6/continual-player-84.json.gz) · [Additional scored practice](../tt6/additional-practice.json.gz)

## What did protected motor recall retain?

On three newly seeded installations, the four-map and seven-map checkpoints both used 24 probe episodes, fitted new maps, and executed identical command tapes on all 21 points. Each landed 12 opening returns and 17 total returns, sustained five points, and won nine. The consolidated checkpoint also preserves the exact native Pong numeric and role state already tested after 42 hall points. [Earlier player on fresh installations](stability-fresh-base.json.gz) · [Seven-map player on fresh installations](stability-fresh-protected.json.gz) · [Native Pong retention](../tt6/pong-retention.json.gz)

On the three installations introduced after point 42, every test serve used a new seed. The extended library recalled the gantry map after one probe and an offset map after two. It refitted the first offset installation. Total calibration fell from 816 to 374 commands, a 54% reduction. Both players won six points. The larger library had one fewer opening, five fewer total legal returns and three fewer sustained rallies on these returning fixtures. [Earlier player on returning installations](stability-recall-base.json.gz) · [Seven-map player on returning installations](stability-recall-protected.json.gz) · [Portable seven-map checkpoint](stability-protected-player.json.gz)

| Matched course | Learned maps at entry | Probe episodes | Openings | Total returns | Sustained points | Wins |
| --- | ---:| ---: | ---: | ---: | ---: | ---: |
| Fresh installations, seed 62812719 | 4 | 24 | 12/21 | 17 | 5 | 9 |
| Fresh installations, seed 62812719 | 7 | 24 | 12/21 | 17 | 5 | 9 |
| Returning installations, seed 62723917 | 4 | 24 | 17/21 | 27 | 7 | 6 |
| Returning installations, seed 62723917 | 7 | 11 | 16/21 | 22 | 4 | 6 |

The returning experiment uses the earlier installation and camera seeds only inside the evaluator. The player receives the same command interface and RGB as on the fresh course. These checks measure retained portable behavior and further body identification. They leave score improvement from general stroke practice open.

## Which changes caused the earlier regression?

Replacing state components in the archived 84-point checkpoint and playing the first published forward course again showed a trade-off. Restoring only the 42-point speed reduced both openings and sustained points. Restoring the visual latency belief increased sustained points while reducing wins. Restoring all three recent quantities together raised openings by one and reduced wins by five. Each run used the same 21 live serves, motor calibrations and disturbances.

| State at entry, seed 54782309 | Openings | Total legal returns | Points with a second return | Wins |
| --- | ---: | ---: | ---: | ---: |
| Full 42-point state | 18 | 27 | 6 | 8 |
| Full 84-point state | 16 | 22 | 5 | 9 |
| 84-point state with the 42-point shot counts and scores | 15 | 23 | 4 | 6 |
| 84-point state with the 42-point stroke speed | 14 | 18 | 3 | 7 |
| 84-point state with the 42-point latency belief | 16 | 25 | 7 | 6 |
| 84-point state with the 42-point visual bounce buffer | 15 | 20 | 5 | 6 |
| 84-point state with all three recent quantities restored | 17 | 25 | 6 | 4 |

The rows are interventions on a single archived course. Effects on one start can alter later state and actions within that course. [Full 42-point state](../tt6/forward-experienced.json.gz) · [Full 84-point state](../tt6/forward-extended.json.gz) · [Shot swap](stability-swap-shot.json.gz) · [Speed swap](stability-swap-speed.json.gz) · [Latency swap](stability-swap-delay.json.gz) · [Bounce swap](stability-swap-bounce.json.gz) · [Recent-state swap](stability-swap-recent.json.gz)

## Can a public-reward stroke baseline keep improving?

In a separate learner, each stroke with a visual opportunity added its used speed and public point reward to a Gaussian-weighted grid. The resulting baseline used eight regularizing observations of weak negative reward and a broad preference centered on the practiced checkpoint’s speed. Every point used this baseline plus a local optically informed correction. The temporary correction decayed between points and restarted at zero on a new installation. A win reduced the influence of a single optical update. Previous practice receipts contributed the player's own speed, the presence of a visual opportunity and public reward. Speed and conditions co-varied during source play, so the archive estimates an association between speed and reward.

Starting from 42 points, the stroke baseline returned 25 of the original 42 held openings across two courses and won nine points. The standard receiver managed 31 openings and 13 wins. It landed 36 legal returns and sustained eight points versus 50 and 14 for the standard receiver. [Held course A](stability-stroke-42-a.json.gz) · [Held course B](stability-stroke-42-b.json.gz)

On the matched 42-point continuation itself, the stroke baseline won 19 points while the standard receiver won nine. It gave up three openings and two legal returns. Both paid for 28 stereo probe episodes. [Score-smoothed practice](stability-stroke-practice.json.gz) · [Matched standard practice](../tt6/additional-practice.json.gz) · [Score-smoothed 84-point checkpoint](stability-stroke-player-84.json.gz)

| Live continuation, seed 18204759 | Openings | Total legal returns | Points with a second return | Wins–losses–capped |
| --- | ---: | ---: | ---: | ---: |
| Standard receiver | 32 | 42 | 8 | 9–29–4 |
| Score-smoothed stroke | 29 | 40 | 9 | 19–17–6 |

The score-smoothed 84-point checkpoint was also evaluated on the new course at seed 62812719. Both score-smoothed players entered with the same camera fits, shot state, optical response, timing and RNG. One kept the accumulated speed and reward observations. The other kept the same starting preference and erased those observations, then continued updating through the same 21 held serves. The erased version landed one more opening and four more legal returns, with the same nine wins. This comparison gives no evidence that the accumulated stroke observations transferred to fresh play. [Learned stroke outcomes](stability-stroke-84-fresh.json.gz) · [Erased outcomes with live learning](stability-stroke-84-erased.json.gz)

Another learner corrected the displacement between the visually tracked blade and the planned strike position around the first interception. It tracked lateral and vertical corrections by the number of public action channels. Each accepted update required two nearby visible frames that agreed on at least one coordinate. One matched 21-point forward course yielded 16 openings, 21 legal returns, three sustained points and five wins. The reference yielded 18, 27, six and eight. On the 42-point continuation it improved wins from nine to ten and fell from 32 to 27 openings. [Optical position compensation](stability-residual-a.json.gz) · [Matched standard receiver](../tt6/forward-experienced.json.gz) · [Position-compensated practice](stability-residual-practice.json.gz)

## Reproduce the trials

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt7_stability_consolidate /tmp/protected-player.json.gz \
  artifacts/tt6/continual-player.json.gz artifacts/tt6/continual-player-84.json.gz
python -m scripts.experimental.tt6_continuous_trial /tmp/protected-fresh.json.gz \
  --seed 62812719 --course gantry offset offset-b --predictive --cached-render --reflections \
  --resume /tmp/protected-player.json.gz
python -m scripts.experimental.tt6_continuous_trial /tmp/protected-recall.json.gz \
  --seed 62723917 --course offset gantry offset-b --predictive --cached-render --reflections \
  --resume /tmp/protected-player.json.gz --stations-from artifacts/tt6/additional-practice.json.gz
python -m scripts.experimental.tt7_stability_trial /tmp/stroke-practice.json.gz \
  --seed 18204759 --course offset gantry offset-b offset gantry offset-b \
  --resume artifacts/tt6/continual-player.json.gz --practice-receipts artifacts/tt6/combined-carry.json.gz \
  --checkpoints /tmp/stroke-checkpoints
python -m scripts.experimental.tt7_stability_trial /tmp/stroke-erased.json.gz \
  --seed 62812719 --course gantry offset offset-b --stroke-mode erased \
  --resume /tmp/stroke-checkpoints/visit-6.json.gz
```

The state-swap diagnostic copies selected keys from the 42-point checkpoint into the 84-point checkpoint before a standard predictive run. Use `numeric`, `speed`, `delay_belief` or `bounce` for the single-component comparisons. The recent-state comparison copies `speed`, `delay_belief` and `bounce`. Run with seed `54782309` and course `offset gantry offset-b`.
