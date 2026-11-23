# SO-101 wrist-camera motor learning

The five-axis wrist controller learned to sustain more physical paddle rebounds than a stationary paddle after extended practice. On 14 fresh starts it scored 47 rebounds against the stationary paddle's 35. Removing the live wrist input from the same learned controller gave 34. A separate 16-start block reproduced the benefit, with 55 live rebounds against 42 stationary and 46 with vision removed. The learned peaks were five and seven rebounds, respectively. [Primary runs](motor-rate-continued.json.gz) · [Second evaluation block](motor-changed-drive.json.gz)

| Standard SO-101 evaluation | Drops | Live wrist | Vision removed | Stationary paddle | Constant visual terms only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Extended practice, first held block | 14 | 3.36 | 2.43 | 2.50 | 2.43 |
| Same weights, second held block | 16 | 3.44 | 2.88 | 2.63 | 3.00 |
| Target reward only, first held block | 14 | 2.71 | 2.50 | 2.50 | 2.57 |

Values are mean verified physical rebounds per drop. The extended-practice controller used 10 short motor probes, 72 rebound-reward trials, and 144 more trials with the capped post-apex bonus. The target-reward-only comparator used 10 short motor probes and 256 scored trials. The two extended training runs began from separately developed readouts, so the difference between them is an observational comparison. [First-stage wrist training](motor-rate-practice.json.gz) · [Target-reward-only outcome](motor-rate-target-only.json.gz)

On held start `1687620118`, an earlier position-readout checkpoint made six rebounds with live wrist images and one in each paired vision-removed and stationary condition. All five joint commands varied in the sighted episode. The recorded commands reproduced each contact and apex event in an independent simulator run. The mean gain in that ten-start development block was 0.10 rebounds over the stationary paddle. [Video](motor-position-witness.mp4) · [Paired episode](motor-position-witness.json.gz) · [Replay receipt](motor-replay.json)

## Learning signals and controls

The motor survey drives balanced short pulses through every joint and measures changes in wrist-camera luminance and encoders. The measured retinal response biases exploration toward movements that disturb the image less. The position or rate readout uses pixel centroids, apparent size, optical changes, and their traces. The implemented color front end uses orange and red components with connected-component isolation. A replaceable detector can provide the same optical observations. Encoder feedback stabilizes the current joint posture, and recursive least squares estimates how action changes joint velocity. [Motor learner](../../scripts/experimental/tt24_motor.py) · [Motor survey](../../scripts/experimental/tt24_motor_modes.py)

Each proposed readout is a reward-gated perturbation of weights between these sensory features and all five actuator outputs. A proposal stays in place through its associated causal trial, with selection based on the arena's scalar task reward, a small duration term, and optional native Pong credit from valid RGB history. Held scoring uses fixed weights. The vision-removed condition keeps encoders and the same weights. The stationary condition issues zero rates. The constant-visual-term control retains the bias and visible-state readout contribution while removing the retinal measurements. Runtime commands use observed pixels, own encoders, and the learned readout. [Paired trial runner](../../scripts/experimental/tt24_bounce_trial.py)

## Short practice and changed body

| Experiment | Training for the measured stage | Pong credit | Cold credit | Shuffled credit | Stationary |
| --- | --- | ---: | ---: | ---: | ---: |
| First position-readout comparison | 24 fresh stock-body trials | 2.00 | 2.00 | 2.00 | 2.90 |
| Quiet-mode rate readout | 10 probes and 24 fresh stock-body trials | 2.67 | 2.58 | 2.92 | 2.75 |
| Same quiet-mode trial with anonymous vision | 10 probes and 24 fresh stock-body trials | 2.67 | 2.58 | 2.67 | 2.75 |
| Source-routed short probes, cohort 1 | 10 motor probes, 9 optical prefixes, 4 scored trials | 2.60 | 2.60 | 2.60 | 2.60 |
| Source-routed short probes, cohort 2 | 10 motor probes, 9 optical prefixes, 4 scored trials | 2.80 | 2.80 | 2.80 | 2.70 |
| Same trained rate checkpoint on changed arm and ball | 12 further full trials per arm | 2.50 | 2.50 | 2.50 | 2.00 |

On the last block, the frozen carry averaged 2.67 rebounds. The three source-credit treatments selected the same readout. On a separate 16-start changed-arm block, updating only encoder-estimated channel gains after 12 practice drops raised the frozen carry from 3.06 to 3.19, compared with 2.06 for a stationary paddle. Updating both gains and the optical motor readout scored 2.88. The short experiments leave the physical score benefit of correctly trained native source credit inconclusive. [Quiet-mode comparison](motor-short-pong.json.gz) · [Anonymous comparison](motor-anonymous-pong.json.gz) · [Short optical probe cohorts](motor-prefix-431202.json.gz) · [Second probe cohort](motor-prefix-13311576.json.gz) · [Changed body and drives](motor-changed-drive.json.gz)

The recurrent readout added a fixed random recurrent state driven by retinal changes, proprioception, and the previous action. On a matched 144-trial development run plus 10 short motor probes, both Pong-credit and cold variants selected the same policy. Each averaged 2.64 over 14 paired starts, against 2.50 for both the stationary and visually ablated controls. [Recurrent Pong](motor-recurrent-pong.json.gz) · [Recurrent cold](motor-recurrent-cold.json.gz)
