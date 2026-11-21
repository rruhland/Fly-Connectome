# Which changed, the arm or the goal?

A gantry and a five-axis arm each completed ten uninterrupted hidden-source searches. Every search included four unannounced conditions in random order. In the 80 scored intervals, a player with two table-tennis motor maps available identified all 40 changes in motor wiring and 37 of 40 source relocations. It made zero cross-channel false alarms: a motor-only change never triggered a reward reset, a reward-only change never triggered a motor switch, and the 20 stable intervals triggered neither. The detector was more reliable at identifying what changed than at reaching the new source. [Results by arm and condition](context-summary.json.gz)

| Actual change | Intervals | Motor switches | Reward resets | Sources acquired | Frames on target, selective / reset both |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neither | 20 | 0 | 0 | 17 | 1,899 / 2,059 |
| Motor only | 20 | 20 | 0 | 16 | 1,338 / 886 |
| Goal only | 20 | 0 | 18 | 17 | 782 / 904 |
| Both | 20 | 20 | 19 | 14 | 547 / 568 |

The selective learner spent 4,566/10,926 frames on target, against 4,417 when either detector reset both models, and 2,432 when only the reward detector was available. The respective acquisition counts were 64, 66 and 32 of 80. The clearest practical benefit of preserving the unaffected model occurred on motor-only intervals. The selective learner then already had a useful estimate of where the source was. The difference in total time on target includes the consequences of earlier decisions in the same continuous run.

When both changed, motor identification usually occurred while the arm explored the relocated source: median 20 frames on gantry and 19 on five-axis. On motor-only changes the corresponding medians were 55 and 29.5. A stationary tool gives little evidence about which motor port drives it. A separate variant added small reversible motor probes every 59 frames. Under that protocol selective resets spent 4,821 frames on target versus 4,166 with paired resets, with 63 versus 66 acquisitions. Every scored case includes those probe frames in the game clock. [Gantry trajectories](context-gantry.json.gz) · [Five-axis trajectories](context-five.json.gz)

## Inputs and controls

The physical arm ran continuously for five blocks. The first established the source and native controls. The four scored blocks were 120–152 frames each, with stable, motor-only, goal-only and simultaneous changes shuffled by seed. The two possible motor assignments came from stored TT13 table-tennis experience on that body. In the motor-only and simultaneous blocks the environment toggled which wiring connected the actuator requests. Source changes moved the target by 14–40 cm. Transport delay and noise also varied within a run. The learner had a calibrated two-view RGB tool marker and a noisy scalar strength reading from an anisotropic quadratic or logarithmically compressed field. Only the evaluator saw the source, schedule and wiring. An acquisition is four consecutive physical frames within 4.5 cm. The task supplies the marker reader and field-maximization objective.

The motor detector used the local effect of earlier commands to predict recent tool displacement at four plausible lags, and switched after two supported comparisons. The reward detector compared strength to its fitted value at the *observed* tool position. Switching motor maps re-estimated used travel in the new coordinates from that position. The task continued running while the learner updated its estimates. The paired-reset control used the same observations, detectors, motor library and exploratory opportunities. It discarded both fits when either detector fired. All requested action tapes reproduced their signal readings and physical outcomes. The distinction between sensory and reward prediction error has also been studied in human motor adaptation. [Learner](../../scripts/experimental/tt15_context.py) · [Evaluator](../../scripts/experimental/tt15_context_trial.py) · [Experimental background](https://pmc.ncbi.nlm.nih.gov/articles/PMC10699606/)

## Unfamiliar controls with no second motor map supplied

A separate cohort used eight fresh runs on each body and a new second port assignment on each run. The starting arms and field protocol were the same. This player could fit a new linear motor map from the earlier part of a moving observation window, validate it on later motion and retain it for future selection. It could start with the original table-tennis geometry, zero geometry, or the original geometry with its port columns reversed. The fourth control started with the original geometry and continuously updated one motor servo. All four players had the same commanded opening probes. Each retained the same reward-change detector.

| Arm and metric over 32 scored intervals | Carried geometry + new maps | Cold + new maps | Reversed columns + new maps | Carried geometry + one motor servo |
| --- | ---: | ---: | ---: | ---: |
| Gantry, frames on target | 1,863 | 1,092 | 1,770 | 890 |
| Gantry, acquired | 24 | 17 | 25 | 11 |
| Five-axis, frames on target | 1,688 | 1,293 | 1,157 | 1,029 |
| Five-axis, acquired | 20 | 19 | 19 | 15 |

The carried learner spent 3,551 frames on target and acquired 44/64 intervals, against 2,385 and 36/64 from scratch. On gantry the reversed-column control also did well; the gain over cold cannot all be attributed to knowing the original port labels. The fitted banks grew to 2–9 maps in the carried runs although there were only two physical contexts per run. That redundancy limits claims about discovery of distinct contexts. [Gantry new wiring](discovery-gantry.json.gz) · [Five-axis new wiring](discovery-five.json.gz) · [Motor discovery](../../scripts/experimental/tt15_discovery.py)

## Reproduction

Use the [simulation environment](../tt6/README.md) from the repository root. Select `five --seed 6423187` or `gantry --seed 26182591` for the known-map suite, and seeds 72148637 and 52843419 respectively for discovery. The discovery trial writes the first run's terminal motor bank beside its output. The transfer experiment separately uses experience up to the first block on the unfamiliar wiring.

```sh
python -m scripts.experimental.tt15_context_trial five /tmp/context-five.json.gz --seed 6423187 --count 10
python -m scripts.experimental.tt15_discovery_trial five /tmp/discovery-five.json.gz --seed 72148637 --count 8
python -m scripts.experimental.tt15_context_report artifacts/tt15/context-five.json.gz artifacts/tt15/context-gantry.json.gz
```
