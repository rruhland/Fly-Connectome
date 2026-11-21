# Learning a new motor mapping during scored games

A simulated four-axis gantry and a five-axis offset arm each played with two previously unseen motor wirings. On all four, the player recovered the exact port permutation and signs after the first point played with the new controls. Its ordinary table-tennis actions supplied the evidence. Across six complete games it won three, scored 60 points and made 90 legal returns. Keeping its original motor map won zero games, scored 24 points and made 36 returns. The two arms started from their own earlier calibrated players. [Gantry starting player](../tt12/motor-gantry-recall-player.json.gz) · [Five-axis starting player](../tt11/carried-d-player.json.gz)

| Arm and game | Online remapping | Fixed motor map | Local refit | Legal returns, online / fixed | Sustained points, online / fixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gantry, new wiring A | 8–11 | 1–11 | 0–11 | 19 / 0 | 4 / 0 |
| Gantry, restore original | **11–9** | 9–11 | 0–11 | 17 / 19 | 5 / 5 |
| Five-axis, new wiring A | 7–11 | 2–11 | 2–11 | 9 / 2 | 2 / 1 |
| Five-axis, restore original | **14–12** | 7–11 | 0–11 | 11 / 7 | 3 / 2 |
| Gantry, new wiring B | **11–7** | 2–11 | — | 19 / 3 | 5 / 1 |
| Five-axis, new wiring B | 9–11 | 3–11 | — | 15 / 5 | 6 / 2 |

Each A fixture started with four ordinary point starts, then switched the ports. Each restoration fixture loaded the previous game's checkpoint, started with four points on the changed ports, then restored the originals. The B fixture on each arm started independently from its earlier calibrated player and used fresh ports, serves and disturbances. The freeze comparison holds only the motor map fixed. All arms retained the usual optically driven stroke, rally, timing and registration updates. The local refit learns linear geometry from exactly the same kinds of play observation, with the original home pose, averaged settling curve and past local fits available for reuse. It frequently replaces its fit, including on familiar ports. Its poor scores are specific to this estimator. [Gantry A](live-gantry-remap.json.gz) · [Gantry B](second-gantry-remap.json.gz) · [Five-axis A](live-five-remap.json.gz) · [Five-axis B](second-five-remap.json.gz)

**What happened after the switch.** Counting only time on changed ports across the six games, remapping scored 30 points and landed 54 legal returns in 69 point starts. The fixed controller scored zero and landed zero in 44 starts. The differing denominators arise from playing each game to completion. A sustained point means the arm legally answered a return to its own serve, or made at least two legal returns on rival serve. There were 25 such points with remapping and 11 with a fixed map over all six games. The physical command tapes of fixed and remapped players were identical through the opening four starts of each independent fixture. After a score difference the roles and point histories can diverge. [Gantry fixed](live-gantry-fixed.json.gz) · [Five-axis fixed](live-five-fixed.json.gz) · [Independent gantry fixed](second-gantry-fixed.json.gz) · [Independent five-axis fixed](second-five-fixed.json.gz)

**What information the learner had.** The sensor path uses the existing calibrated first-person stereo tracker. At the end of a point, the adapter compares predicted blade motion against the player's recorded RGB blade position and pitch at several lags. If the retained map misses, the adapter tests signed relabelings of that map, selecting with alternating four-frame blocks and validating against the intervening blocks from the same point. Model fitting and switching use requested commands and visual measurements. The physical wiring, seed, scheduled switch, applied commands and referee events enter the simulator and audit. The player receives ordinary game rewards through its inherited game policy. No backpropagation, extra probe points or unscored calibration swings were used for these switches. The test supplies a family of possible motor transformations, a calibrated visual system and an existing stroke planner. [Learner](../../scripts/experimental/tt13_live_ports.py) · [Full game protocol](../../scripts/experimental/tt13_live_trial.py) · [Evaluator-only exact port audit](../../scripts/experimental/tt13_live_audit.py)

**Stability.** Both arms reselected their exact original map after one point when the original controls returned. Three isolated false selections occurred across the six games, each corrected after one further point. On a separate five-axis fixture with unchanged controls the learner made zero switches. Both branches won 11–7 with identical command tapes on all 18 starts. Closely spaced frames within a rally share noise and motion, so the within-point validation here checks consistency within the point. Independent fixtures provide the check on game performance. [Unchanged controls](stable-five-remap.json.gz) · [Fixed comparison](stable-five-fixed.json.gz) · [Gantry checkpoint after restoration](live-gantry-remap-player.json.gz) · [Five-axis checkpoint after restoration](live-five-remap-player.json.gz)

**Update time.** Replaying the 16 saved optical histories of opponent replies to own service reproduced all recorded map choices. The twelve updates that skipped a search took about 0.4 ms median CPU time on each arm. Three gantry searches took 5.1 ms median, and the single five-axis search took 29.7 ms. These are per-point update costs on one CPU, excluding the stereo tracker, rendering and physics. [Timing samples](adaptation-cost.json.gz) · [Timing script](../../scripts/experimental/tt13_live_cost.py)

The five-axis player after the 14–12 game reproduced every action tape on 180 native Pong starts, earning the same 41 points. This checks preservation of native Pong behavior. [Native retention](live-five-pong.json.gz) · [14–12 first-person replay](five-revisit.mp4)

## What transferred to a different task

We extracted the motor map in use at the end of game A, before restoration, and used its local geometry as the starting map for following a moving 3D target. Each player faced the same eight fresh target paths twice per arm, once on familiar display axes and once on randomly rotated axes. Delay changed during the trial and motor noise began at frame 48. All four controllers continued to estimate their local motor response from visual feedback. The target-only player started with zero geometry and spent its first movements probing. The objective and colored marker observations are supplied. This tests immediate reuse of a learned motor relationship in a different task on the same body. [Tracking implementation](../../scripts/experimental/tt13_live_tracking.py)

| Frames within target tolerance (1,536 per row) | Remapping from table tennis | Local refit from table tennis | Original table-tennis map | Target-only |
| --- | ---: | ---: | ---: | ---: |
| Gantry, familiar axes | 890 | 847 | 777 | 649 |
| Gantry, rotated axes | 772 | 727 | 791 | 611 |
| Five-axis, familiar axes | 868 | 704 | 485 | 565 |
| Five-axis, rotated axes | 704 | 640 | 493 | 606 |

The remapped player spent 3,234/6,144 frames on target, versus 2,431 for the target-only learner and 2,546 for the original map. On rotated gantry views the original map finished 19 frames ahead of remapping. Every scored motion tape was independently replayed with the same simulated hardware. [Gantry target evidence](tracking-gantry.json.gz) · [Five-axis target evidence](tracking-five.json.gz)

## Reproduction

Use the [simulation environment](../tt6/README.md) from the repository root with the experimental scripts on PYTHONPATH. Write results to an existing output directory. The first example plays and reloads the two-game five-axis series. The other commands audit and render it.

```sh
python -m scripts.experimental.tt13_live_trial artifacts/tt11/carried-d-player.json.gz /tmp/live-five-remap.json.gz --checkpoint /tmp/live-five-remap-player.json.gz --mode remap --seed 14379123 --ports 215648317
python -m scripts.experimental.tt13_live_audit /tmp/live-five-remap.json.gz
python -m scripts.experimental.tt8_service_replay /tmp/live-five-remap.json.gz /tmp/five-revisit.mp4 --game 2
```

For the gantry A series use the linked gantry starting player, seed 67254031 and ports 95203617. The independent B cases use one game (`--games 1`), with gantry seed 97581321 and ports 22531667, and five-axis seed 7265583 and ports 16498251. Stable five-axis uses seed 36194235, ports 215648317 and `--games 1 --stable`. The control modes are `fixed` and `refit`. To reproduce pursuit, run `scripts.experimental.tt13_live_tracking` on a directory containing the A series and their checkpoints, with site `gantry --seed 17546921` or `five --seed 81637509`.

Predicting sensory consequences of an action to select among motor controllers has a precedent in Wolpert and Kawato. Here the discrete candidate family makes a signed port change identifiable; a useful next test is a coupling or change in motor gain outside that family. [Multiple paired forward and inverse models for motor control](https://pubmed.ncbi.nlm.nih.gov/12662752/)
