# Identifying a changed motor interface before receiving

The serial diagnostic identified all 12 hidden interfaces across two fresh six-setting courses. The first confirmed estimate was correct in every setting. It took 26 actuator ticks with no delay and 27 ticks with a one-tick delay. On 96 held feeds, both active learners landed all 16 returns achieved by the known-interface expert. The matched passive learner landed nine. [Paired courses and 328 command replays](motor-active-serial.json.gz)

| Receiver | Exact interfaces out of 12 | Legal returns out of 96 | Paddle contacts |
| --- | ---: | ---: | ---: |
| Active probes, carried routing prior | 12 | 16 | 68 |
| Active probes, reset routing prior | 12 | 16 | 68 |
| Passive ordinary command history | 7 | 9 | 41 |
| Passive with the active learner's previous map | 6 | 8 | 38 |
| Frozen previous map | 0 | 0 | 6 |
| Known-interface expert | 12 | 16 | 68 |

The active learners gained seven paired returns and lost none versus passive. They each landed eight legal returns per stream, compared with four and five for passive. Retaining the prior routing gave no advantage once both learners actively probed each channel. [Per-setting outcomes](motor-active-serial.json.gz)

Five short single-port diagnostic motions provide enough information to identify which joint each command drives, whether its sign is flipped, and whether the interface delays commands by a tick. Each packet sends a positive and negative request of magnitude 0.32, then waits four ticks. The arena starts each packet at its normal home state. This costs 30 actuator ticks at 30 Hz and ten nonzero requests per setting. The simulator supplies five home starts, so the physical time needed to rehome a robot remains unmeasured. [Diagnostic commands and encoder traces](motor-active-serial.json.gz)

The greatest measured joint excursion was 0.076 rad, and the maximum joint deviation at the end of a packet was 0.0124 rad. [Physical encoder traces](motor-active-serial.json.gz)

The source recorded 20 nominal encoder templates in 120 ticks. These cover five isolated native joint responses, both signs and both delays. The model retains the physical angle and velocity response instead of summing individual effects. At a hidden switch it scores all 7,680 possible signed permutations and delays against the five observed prefixes. A candidate needs a prediction margin on two consecutive readings before the system accepts it. The reset learner uses the same source model, observations and decision rule with a uniform routing prior. The carried learner gives extra prior weight to its previously accepted permutation. [Source templates and implementation](motor-active-serial.json.gz) · [Estimator](../../scripts/experimental/tt27_motor_active_serial.py)

The test changes the interface six times in each of two independently seeded streams. Three distinct permutations appear in each stream, and signs and delays vary between settings. The held incoming feeds use fresh recorded-launch perturbations whose lateral position and speed also change. The passive learner receives the same number of home starts and begins five ordinary receiving attempts, with six measured ticks in each. It uses the earlier bounce-derived encoder predictor with its match-clock correction. Another passive branch starts from the active learner's previous accepted estimate. The known-interface expert uses the same camera receiver and actual delayed plant with the correct inverse. All scored commands are replayed through the actual hidden channel against the physical referee. [Paired feeds and replays](motor-active-serial.json.gz)

## Receiving without a reset after the diagnostic

A separate 40-feed test ran all five diagnostic packets and then released the ball on the same simulated plant. The receiver identified all eight hidden settings and landed ten legal returns, matching the known-interface expert and the equally active reset. The stale-map and factory controllers each landed zero. Six of the ten returns occurred after the first setting in each stream. [Continuous preparations and 110 verified command branches](motor-active-continuous.json.gz)

The receiver waited eight ticks at its fixed servo targets, then issued five ten-tick packets. Each packet used two opposed requests and eight zero-request settling ticks. The first accepted estimate was correct after 50 or 51 total ticks in all eight settings. The feed began at tick 58, after 1.93 simulated seconds and zero intermediate arm resets. Across the five packets and eight settings, peak joint excursion was 0.074 rad and the largest end-of-packet deviation from the packet start was 0.00122 rad. The simulated holder retained the ball during the diagnostic. Releasing the feed preserved the physical arm position, velocity, servo targets and clock. The paired branches shared an identical release-state checksum. [Continuous implementation](../../scripts/experimental/tt27_motor_active_continuous.py) · [Encoder and release-state receipts](motor-active-continuous.json.gz)

An earlier simultaneous packet also used balanced requests. Its linear superposition model had roughly 15–40 milliradians of position error on nominal validation motions. The encoder evidence failed the confidence rule on every hidden setting, leaving scores at zero, while the known-interface expert landed 17 of 96. [Simultaneous-pulse trial](motor-active.json.gz)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_active_serial
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_active_continuous
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt27_motor_active
```
