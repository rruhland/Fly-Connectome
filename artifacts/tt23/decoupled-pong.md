# Independent Pong release height broadened source experience

The ordinary launch coupled height to vertical speed with correlation +.9999. Drawing height from a separate deterministic stream reduced it to −.0136 across 16,384 checked seeds. The independent-height game produced 54 mixed position-and-velocity optical observations in 144 training starts and 16 in 48 held starts. The coupled game produced zero in either sample. The new native fit improved held duration relative to its shuffled labels. On previously measured independent-release arm starts it produced fewer legal returns than the same learner trained on ordinary Pong. [Design](decoupled-pong-design.md) · [Frozen native results](decoupled-pong-native.json.gz)

## What changed in native Pong

The opt-in subclass keeps the ordinary simulator reset and replaces only release height. Both curricula use the same .2–.8 height range and the same ±.6 radian angle range. The 5th, 50th and 95th height percentiles were .231, .499 and .770 in ordinary Pong and .231, .505 and .771 in independent-height Pong. Launch velocities were bitwise equal at each matched seed, including after selective resets. The ordinary harness reproduced 12 archived controls, first-return events and optical contexts exactly. The Pong, arcade-wall and return-learning tests passed, 19 in total. [Experimental subclass](../../scripts/experimental/tt23_decoupled_pong.py) · [Native harness](../../scripts/experimental/tt3_pong.py)

The same optical servo and fixed left, center and right motor commands ran each native option. Each native curriculum used 144 training and 48 held launches, with three actions executed at every launch. The independent-height cohort supplied 576 native episodes and 218,732 physics frames, 30.4 minutes at 120 Hz. A shared model setting of unit bandwidth and zero uncertainty penalty won the six leave-one-cohort-out folds by mean elapsed duration. Every fit and duration shuffle then used the same setting. The arm outcomes entered after freezing both native models. [Native outcomes](decoupled-pong-practice.json.gz) · [Fit and model settings](decoupled-pong-native.json.gz)

## Native held behavior

| Source policy | Ordinary held mean duration / first returns | Independent-height held mean duration / first returns |
|---|---:|---:|
| Coupled source | 3.309 s / 15 | 3.124 s / 12 |
| Coupled source labels shuffled | 2.695 s / 8 | 3.027 s / 11 |
| Decoupled source | 3.348 s / 15 | 3.262 s / 14 |
| Decoupled source labels shuffled | 3.015 s / 8 | 2.479 s / 6 |
| Center | 2.695 s / 8 | 3.027 s / 11 |

On the held independent-height starts, the decoupled fit gained 0.783 seconds relative to its shuffled fit, with a paired 95% interval of +0.229 to +1.423 seconds. Its gain against the coupled fit was +0.138 seconds, with interval −0.174 to +0.543. On the 16 optically mixed starts it averaged 3.838 seconds, compared with 3.156 for the coupled fit and 3.644 for center. Those 16 starts yielded six, four and six first returns respectively. The decoupled model chose a jog on all 48 held starts. The coupled model held center 11 times and held on four of the seven starts where center physically lasted longer than either jog. [Native held durations and decisions](decoupled-pong-native.json.gz)

## Descriptive transfer to the same physical arm options

| Physical bank | Coupled source legal openings | Decoupled source legal openings | Decoupled shuffled | Center |
|---|---:|---:|---:|---:|
| Original, 64 | 21 | 21 | 0 | 11 |
| Independent release, 64 | 26 | 22 | 3 | 36 |
| Peer independent release, 64 | 23 | 19 | 2 | 25 |
| Visible mixed approaches across the independent banks, 56 | 20 | 12 | 1 | 36 |

The two true fits chose exactly the same arm actions on the original bank. On the 56 conflicting-motion arm approaches, the decoupled fit made 36 physical blade contacts versus 31 for the coupled fit, while legal returns fell from 20 to 12. There it chose no holds. The coupled fit held on 31 approaches. Broader native optical experience thus left a domain difference in the value of holding and converting contacts. The new mixed-course gate stayed closed. [Physical selections, contacts and strict paired outcomes](decoupled-pong-arm.json.gz)

Recreate the native comparison with `python -m scripts.experimental.tt23_decoupled_pong --workers 3`. The command retains the independent native tapes and scores the three existing arm banks after freezing the source models. Add `--arm-only` to reuse the frozen source artifacts.
