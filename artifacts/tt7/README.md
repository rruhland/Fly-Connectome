# Contact timing, optical flight and skill retention

The [six-joint timing experiment](contact-timing.md) includes the player, training receipts, two matched games and a [five-return first-person rally](contact-soft-five-returns.mp4). The [motor recall experiment](stability.md) compares later practice with the 42-point donor on returning and unfamiliar stations. The [catching experiment](catch-transfer.md) tests forward transfer into a separate RGB game. [Probe caching](probe-cache.md) gives equivalent calibration tapes more cheaply.

## Optical control of the outgoing shot

The previous receiver often reached the ball and hit it short. Four coherent outgoing stereo sightings provide a flight estimate. On the untouched first course, the player obtained 30 estimates in 40 physical racket separations on the gantry and 13 in 17 on the offset arm. All 43 optical estimates matched a separation within six camera frames. Median absolute error in forward and vertical speed was 0.10 and 0.10 m/s on the gantry, and 0.07 and 0.13 m/s on the offset arm. The referee and its events were used only after play for this comparison.

Each entry below is a scored receiving game to 11, win by two. Each arm gets a newly fitted visual motor map as needed. A 42-point donor supplies the same initial portable preferences. Racket response, delay and noise change during the games.

| Course | Arm | Original | Camera flight correction | Fixed flight correction |
| --- | --- | ---: | ---: | ---: |
| 53785301 | Gantry | 9–11 | 11–7 | 11–2 |
| 53785301 | Five-axis offset | 4–11 | 4–11 | 7–11 |
| 79460213 | Five-axis offset | 2–11 | 3–11 | — |
| 79460213 | Gantry | 10–12 | 11–7 | — |
| Fresh 178435031 | Five-axis offset | 7–11 | 7–11 | 11–9 |
| Fresh 178435031 | Gantry | 6–11 | 8–11 | 8–11 |

On the fresh course the adaptive player started with a 16-sample optical buffer acquired during 21 additional points, then continued updating. Only this buffer transferred from those points. The original and fixed versions started from the same 42-point player. The first two courses started with an empty optical buffer. Instrumenting the original left all 35 command tapes on course 53785301 identical to the archived TT6 run. The fixed correction subtracts 1.2 m/s from predicted forward speed and 0.54 m/s from predicted rise. This deliberately simple control won the only fresh five-axis game. The camera predictor measured real impacts, but using that data to select a better shot across bodies remains open.

The two-axis impulse fit took one long gantry game 16–14 with 47 legal returns, then had no wins on either tested offset installation. Choosing the stroke against a distribution of optical errors likewise lost 0–11 on the first offset installation. A separate controller that penalized errors in blade velocity at the strike reached 13/18 opening returns on that installation, with a 7–11 point score. Combining velocity control with the median optical correction lost 3–11. On the short 21-point forward course, the median learner fell to 16 openings and 21 total legal returns from the original 18 and 27. The saved 42-point reference remains the general receiver.

[Unchanged player](flight-game-observe.json.gz) · [Adaptive flight](flight-game-adapt.json.gz) · [Replication](flight-game-confirm.json.gz) · [Fixed control](flight-game-loft.json.gz) · [Fresh control](flight-fresh-observe.json.gz) · [Fresh fixed](flight-fresh-fixed.json.gz) · [Fresh adaptive](flight-fresh-adapt.json.gz) · [Optical practice](flight-learn.json.gz) · [Portable optical state](flight-learned.json.gz) · [Impulse model](flight-impulse-confirm.json.gz)

## Running the fresh optical comparison

Use the [TT6 environment instructions](../tt6/README.md). This example resumes the 42-point player and the donor optical state. Use the same command with feedback set to fixed or observe and omit flight-in for the two controls.

```bash
python -m scripts.experimental.tt7_flight_trial /tmp/flight-adapt.json.gz \
  --seed 178435031 --course offset gantry --predictive --cached-render --reflections \
  --probe-cache /tmp/tt7-calibration --resume artifacts/tt6/continual-player.json.gz \
  --race-to 11 --max-steps 420 --feedback adapt --flight-in artifacts/tt7/flight-learned.json.gz
python -m scripts.experimental.tt7_flight_audit artifacts/tt7/flight-game-observe.json.gz
python -m scripts.experimental.tt6_replay /tmp/flight-adapt.json.gz
```

Learning a stroke-to-flight mapping from sensor data is also used in physical robot table tennis. Here the evidence for useful control comes from played games and fixed-policy comparisons. [Data-efficient Online Learning of Ball Placement in Robot Table Tennis](https://arxiv.org/abs/2308.14562)
