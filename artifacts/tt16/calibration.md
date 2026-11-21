# Identifying a new motor wiring from a short movement

The four-axis gantry and five-axis arm each began with a table-tennis policy and one previously learned motor model for its original wiring. It predicts visible blade position and pitch under commanded movements. The new installation may change which public port drives which actuator and reverse any sign. The learner knows this family of changes. It must identify the particular wiring from public commands and stereo images.

## Clean calibration and scored games

Each port follows a different sequence of positive and negative pulses during a single 12-command movement. A candidate sequence must stay within three cumulative command units per port. The controller chooses the sequence that gives the largest minimum singular value after filtering by the old arm's average settling response. This makes it harder for two ports to produce indistinguishable observations. The criterion is related to excitation design in system identification. The specific tape, loss and test here use the simulated arm's own previously fitted camera model. [Excitation design background](https://underactuated.csail.mit.edu/sysid.html) · [Implementation](../../scripts/experimental/tt16_coded_probe.py)

The controller tries each signed port assignment on the recorded command sequence, including the source model's nonlinear terms and per-axis lag. It removes a constant camera registration error and scores observed position and visible pitch in units scaled by the old model's sensitivity. Using the earlier stroke-planning weight for pitch had confused two axes on both five-axis TT15 development fixtures. We changed the scaling on those earlier fixtures, then kept it fixed on TT16. The new wiring, exact joints, referee events, executed noisy commands and time-delay labels are unavailable to the fit.

On all four new camera and wiring fixtures, this direct test recovered the whole assignment. It used 13 image pairs and 12 control intervals (0.40 simulated seconds), plus a shared 68-interval camera setup on the original wiring. The two quiet terminal intervals let the movement settle. The active port commands used 28 normalized absolute rate units on the gantry and 35 on the five-axis arm. The single-port pitch check used only 2.1 units in the same time. Absolute command magnitude measures requested movement, not physical energy. [Held-out scores, elapsed time, and full preparation costs](transfer.md)

| Fixture | Game score | Legal returns | Points with at least two arm returns | All frames to first return |
| --- | ---: | ---: | ---: | ---: |
| Gantry 0 | 12–10 | 27 | 7 | 126 |
| Gantry 1 | 11–5 | 26 | 5 | 125 |
| Five-axis 0 | 11–7 | 18 | 4 | 128 |
| Five-axis 1 | 11–7 | 12 | 2 | 129 |

The original table-only controller updated its map from scored points and won one game with 54 returns. Direct coded calibration won four and made 83 returns, with the first legal return earlier on every fixture. The corrected wiring is fixed in the coded games. The existing rally and service updates continue. Its five-axis checkpoint reproduced all 180 native Pong action tapes and the original 41 points. [Complete game from the left camera](five-axis-full-game.mp4) · [Five-axis checkpoint](transfer-five-0-coded-player.json.gz) · [Pong replay](coded-pong.json.gz)

On this cloud CPU, after one warmup, selecting the command sequence took a median 4.2 ms on gantry and 4.3 ms on five-axis over ten calls with one BLAS thread. Decoding the saved optical observations took 2.3 and 17.5 ms respectively. Rendering, camera tracking and physical movement were outside this timing. Enumerating signed assignments considers 384 candidates at four ports and 3,840 at five, so these timings do not establish scalability to high-dimensional arms.

## Unknown transport during the calibration movement

Twelve installations received the same 12-tick tape in each of three transport conditions. Four repeated the scored fixtures and eight used new wirings. The tool tracker saw only stereo images and the commands it requested. The alternative fit compared possible clocks for the observed movement, allowing a constant delay from zero to three ticks or one mid-tape switch. It used the same images and no extra movement. [Timing fit and evaluator](../../scripts/experimental/tt16_transport_calibration.py)

| Transport during probe | Exact wiring under immediate response | Exact wiring allowing timing uncertainty |
| --- | ---: | ---: |
| Ordinary | 11/12 | 11/12 |
| Constant two-frame delay plus noise | 0/12 | 9/12 |
| Delay switched between zero and three frames, with changing noise | 3/12 | 9/12 |

On the four scored installations, using the changing-transport calibration raised wins from two to three and returns from 53 to 56. Both five-axis arms went from 0–11 to 11–7. On gantry 0 the extra timing freedom confused two ports, turning a 12–10 win with 27 returns into a 0–11 loss. We counted an existing game only where the setup, donor and entire frozen motor map exactly matched, and verified its points against physics. Different maps played full games. [Paired game verifier](../../scripts/experimental/tt16_transport_games.py) · [Gantry reversal](coded-gantry-0-transport-games.json.gz) · [Five-axis comparison](coded-five-0-transport-games.json.gz)

## A second movement on 24 fresh wirings

We next kept the arm and its transport running for a second set of 12 commands if the first movement produced few sightings, a weak winning margin, or a high residual. The second sequence stays within the same excursion limit over the entire 24-tick movement. Each half can have its own fitted timing. We evaluated eight ordinary, eight delayed and eight switching installations, with command noise changing within the latter trajectories. All 24 fixed-wiring counterfactuals were captured through the second movement. The adaptive learner used its first 12 observations to decide whether to consume that continuation. [Controller and evaluation](../../scripts/experimental/tt16_continuous_probe.py)

| Independent installation group | Exact after 12 commands | Exact after optional continuation |
| --- | ---: | ---: |
| Ordinary | 5/8 | 7/8 |
| Delayed | 2/8 | 6/8 |
| Delay changed twice | 4/8 | 7/8 |
| Total | 11/24 | 20/24 |

The rule requested the second movement in 22/24 cases, using 23 commands per installation on average. Always using 24 commands produced the same 20 exact results. Most of the gain came from the extra independent movement. The selection rule saved little time. The held-out camera mounts still use public calibration and the optical tool tracker. Arbitrary changes to body geometry are outside this test.

The first shifting-transport installation in each of the four fixture groups also played a complete game. With only the first movement they lost all four and made 20 returns overall, with three points containing two or more returns. The optional continuation won one game and made 44 returns, with nine multiple-return points. On gantry 0 it changed a 2–11 loss into a 12–10 win. On five-axis 0 it reduced the score from 4–11 to 1–11. These are new wiring and game seeds on the four established camera setups. Reacquiring the 64 short movements across the three cohorts reproduced their optical readings and fitted decisions. [Fresh gantry game](coded-gantry-0-continuous-games.json.gz) · [Five-axis counterexample](coded-five-0-continuous-games.json.gz) · [Independent optical audit](../../scripts/experimental/tt16_coded_audit.py)

## Continuing from storage

We restored the five-axis player that had won 11–7 on the first clean calibration fixture. Recomputing that first game through stereo vision reproduced all 18 point tapes. Its first four new point tapes matched a player that had stayed in memory across the boundary. The restored player then won the new game 15–13 with 13 legal returns and three points with at least two returns. It used no further camera or motor setup. The resumed checkpoint also reproduced all 180 native Pong action tapes. [Post-reload Pong audit](resumed-pong.json.gz) [Resumed game](coded-five-resumed-game.json.gz) · [Second-game checkpoint](coded-five-resumed-player.json.gz) · [Resume procedure](../../scripts/experimental/tt16_coded_resume.py)

## Reproduction

Use the [TT6 environment](../tt6/README.md) and the frozen primary setups from [the transfer protocol](transfer.md). For each of the four primary fixture names run the following commands.

```sh
python -m scripts.experimental.tt16_coded_trial artifacts/tt16 "$fixture"
python -m scripts.experimental.tt16_coded_trial artifacts/tt16 "$fixture" --game
python -m scripts.experimental.tt16_transport_calibration artifacts/tt16 "$fixture"
python -m scripts.experimental.tt16_transport_games artifacts/tt16 "$fixture"
python -m scripts.experimental.tt16_continuous_probe artifacts/tt16 "$fixture"
python -m scripts.experimental.tt16_continuous_games artifacts/tt16 "$fixture"
python -m scripts.experimental.tt16_coded_audit artifacts/tt16 "$fixture"
```
