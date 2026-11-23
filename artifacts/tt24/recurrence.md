# Transferring the effects of local motor interventions

A readout trained on paired native Pong trials improved wrist-camera bounce performance relative to otherwise identical reward-only and shuffled-source updates. The independent held replication averaged 3.50 rebounds with Pong credit, 2.54 with reward-only calibration, and 2.25 with shuffled credit. The prior embodied controller averaged 3.29 without the additional calibration. The experiments establish a benefit over the matched updates in the stock-body replication. The gain over retaining the original embodied readout remains uncertain. [Held replication](recurrence-arm-replication.json.gz)

## Native training

Native Pong runs shared their seeds, randomized camera presentation, initial RGB and optical motor identification. After both component tracks were available, a visible approach estimate scheduled a short motor pulse. The trial branched into each signed pulse and a neutral command. The pulse lasted eight camera ticks. A short RGB window around the estimated encounter supplied independent mover positions and motion, measured relative to its visible approach and the controlled object's apparent half-span. A ridge fit on paired feature differences predicted which branch would have a longer point. Contact tags and simulator ball coordinates were outside this fit. The shuffled control flipped the elapsed-duration difference on a randomly assigned set of whole launches.

The native practice roster contained 192 launches and yielded 127 visible non-tied comparison pairs across 59 launches. The independent 126-launch course yielded 86 such pairs across 36 launches. Pair ordering accuracy was 88.4% with trained Pong labels, 66.3% with shuffled labels and 50.0% with a flat score. The bootstrap interval for the gain over shuffle was 8.8 to 36.1 percentage points using whole launch resampling. [Native evidence](recurrence-native.json.gz) · [Frozen weights](recurrence-weights.json)

## Matched arm calibration

All stock-body results in the first two rows below used the same warm wrist controller, which had ten earlier motor probes and 256 scored target trials. The new calibration ran ten pairs on matched drops, with each branch changing the framewise actuator readout along one signed random direction for nine control ticks. Both branches returned to the common readout when the pulse ended. The 20 calibration branches used 27.23 simulated seconds in total. Each branch used the same fixed-camera approach estimate to time the witness. The portable source compared observed mover motion between branches. The local fit combined that comparison with each branch's public rebound reward and elapsed time. The fitted effects changed continuous motor readout weights. Held play used only wrist RGB and encoders.

| Paired held course | Drops | Pong | Reward only | Shuffled | Unadapted anchor |
|---|---:|---:|---:|---:|---:|
| First stock-body course | 12 | 35 | 30 | 26 | 31 |
| Exact-readout fresh replication | 24 | 84 | 61 | 54 | 79 |
| Fresh calibration on alternate arm and changed physics | 12 | 30 | 29 | 27 | 29 |
| Fresh stock calibration from zero motor readout | 12 | 35 | 31 | 25 | 34 |

Values are total verified rebounds. The 24-start replication froze all four readouts before testing. The paired mean increase was 0.96 rebounds versus reward-only updates, with a 95% bootstrap interval of 0.33 to 1.63. The increase versus shuffled was 1.25, with an interval of 0.58 to 2.04. The increase versus the warm anchor was 0.21, with an interval of -0.25 to 0.71. Best consecutive counts were nine with Pong credit and ten with the warm anchor. None of the alternate-arm intervals separated the treatments. From a zero readout the Pong advantage over shuffle was 0.83 per drop, with an interval of 0.42 to 1.25. Intervals for reward-only and stationary comparisons straddled zero. Every held tape in this table was replayed through the physical scorer. [First course and exact policy arrays](recurrence-arm.json.gz) · [Alternate arm](recurrence-arm-changed.json.gz) · [Zero-readout course](recurrence-arm-untrained.json.gz)

The alternate-arm and zero-readout courses used new pulse directions and new practice and held starts. The alternate-arm course changed arm geometry, wrist camera placement, actuation and ball/contact dynamics together. The zero-readout course used stock physics with zeroed motor weights and fresh motor-effect estimates. Its unadapted anchor used encoder stabilization and zero optical drive.

The confirmed stock-body result uses supplied optical encounter timing in native practice and target calibration. It also uses two available cameras during target calibration and a previously experienced body. These are part of the tested route. The result measures the value of the learned local effect score under that scaffold.

## Revisit after new practice

After the alternate-arm and zero-readout calibrations, the stock warm and Pong readouts reissued identical motor commands on twelve original starts each and reproduced all rebound and termination fields. The native task reproduced all 378 held pulse-branch command digests, RGB-track digests, and durations on its 126-start course. This tests recovery from separate saved readouts. [Stock-arm revisit](recurrence-arm-retention.json.gz) · [Native revisit](recurrence-native-retention.json.gz)

## Removing the approach timer

In a separate native experiment, each signed pulse started after a random delay from the first visible actor binding. A learned gate used differences in the two branches' recent RGB velocity and position. Its training target was future separation of the independent mover's two tracks. The 180 held launches contained only 12 positive windows among 3,601 measured windows. The influence detector's AUC was 0.999, and the continuation comparison had only one eligible held pair. It failed the native qualification for arm use. [Random-time course](recurrence-surprise-native.json.gz)

An independent native course used twelve-frame pulses and a 105-frame witness cap. Its 150 held launches provided 210 positive windows out of 4,691. The learned gate's AUC was 0.9999. The later-duration readout ordered 11 of the 13 eligible launch pairs correctly. The flat score gave 50.0% and the shuffle 23.1%. Eligibility remained sparse. [Extended native course](recurrence-surprise-extended-native.json.gz) · [Extended weights](recurrence-surprise-extended-weights.json)

The corresponding arm calibration began each pulse after a random delay from a visible fixed-camera track prefix. It recorded up to 34 additional frames without a supplied encounter estimate. The frozen source gate chose the first paired RGB window with enough evidence of directed mover divergence. All ten matched practice pairs produced a supported witness. Across the first twelve held drops, the Pong, reward-only, shuffled and unadapted readouts made 35, 32, 28 and 39 verified rebounds. Neither matched Pong advantage separated from zero in that course. [Random-time arm course](recurrence-surprise-arm.json.gz)

The exact same readouts on 24 more held starts made 69, 65, 56 and 73 rebounds. The Pong increase over shuffled credit was 0.54 rebounds per drop, with a 95% paired bootstrap interval of 0.08 to 0.96. The difference from reward-only updates was 0.17, with an interval of -0.04 to 0.38. The difference from the unadapted readout was -0.17, with an interval of -0.63 to 0.33. All 96 held command tapes replayed successfully. The random-time result separates from the shuffle while leaving the reward-only comparison unresolved. [Random-time replication](recurrence-surprise-arm-replication.json.gz)

## Reproduction

Use the repository's table-tennis environment from its root directory. Set `PYTHONPATH=src`, `MUJOCO_GL=egl`, `OPENBLAS_NUM_THREADS=1`, and `OMP_NUM_THREADS=1`.

```sh
python -m scripts.experimental.tt24_recurrence
python -m scripts.experimental.tt24_recurrence_arm
python -m scripts.experimental.tt24_recurrence_arm --replicate artifacts/tt24/recurrence-arm.json.gz
python -m scripts.experimental.tt24_recurrence_arm --arm alternate --regime changed \
  --schedule 105352043 --output artifacts/tt24/recurrence-arm-changed.json.gz
python -m scripts.experimental.tt24_recurrence_arm --untrained \
  --schedule 962546997 --output artifacts/tt24/recurrence-arm-untrained.json.gz
python -m scripts.experimental.tt24_recurrence_arm --revisit \
  artifacts/tt24/recurrence-arm-changed.json.gz artifacts/tt24/recurrence-arm-untrained.json.gz
python -m scripts.experimental.tt24_recurrence --revisit \
  artifacts/tt24/recurrence-arm-replication.json.gz artifacts/tt24/recurrence-arm-changed.json.gz \
  artifacts/tt24/recurrence-arm-untrained.json.gz
python -m scripts.experimental.tt24_recurrence_surprise
python -m scripts.experimental.tt24_recurrence_surprise --extended --train 192 --held 150
python -m scripts.experimental.tt24_recurrence_arm --source artifacts/tt24/recurrence-surprise-extended-weights.json \
  --schedule 603472194 --output artifacts/tt24/recurrence-surprise-arm.json.gz
python -m scripts.experimental.tt24_recurrence_arm --replicate artifacts/tt24/recurrence-surprise-arm.json.gz \
  --schedule 646979735 --output artifacts/tt24/recurrence-surprise-arm-replication.json.gz
```
