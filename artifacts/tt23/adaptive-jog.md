# Continuing native Pong duration learning on the SO-101

Native pretraining improved early held performance relative to both equally exploratory selectors. It averaged about three extra legal returns out of 64 over flat initialization on each physical bank. Holding the arm centered still produced more legal returns than the Pong-pretrained selector over the early budgets on both banks. The continuous-point follow-up gate remained closed. [Design](adaptive-jog-design.md) · [Paired choices and receipts](adaptive-jog.json.gz)

## Held legal returns by completed arm points

The primary initialization pooled four independent native Pong datasets. The original and replication columns identify the held arm bank. Each learner practiced on the other bank in its recorded order.

| Held bank | Initialization | 0 points | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | Pong duration | 0.0 | 5.8 | 10.4 | 11.1 | 14.5 | 11.2 | 16.1 |
| | Flat | 4.4 | 4.9 | 6.1 | 7.2 | 10.5 | 7.6 | 12.0 |
| | Shuffled Pong duration | 11.0 | 8.0 | 4.2 | 2.8 | 9.0 | 7.7 | 10.1 |
| | Center | 11.0 | 11.0 | 11.0 | 11.0 | 11.0 | 11.0 | 11.0 |
| Replication | Pong duration | 0.0 | 5.8 | 14.4 | 14.7 | 17.5 | 13.1 | 16.2 |
| | Flat | 6.0 | 7.6 | 7.0 | 10.7 | 11.7 | 10.7 | 11.8 |
| | Shuffled Pong duration | 18.0 | 7.8 | 6.6 | 11.5 | 9.4 | 11.0 | 11.4 |
| | Center | 18.0 | 18.0 | 18.0 | 18.0 | 18.0 | 18.0 | 18.0 |

The mean difference at 4, 8 and 16 completed points was +3.03 legal returns versus flat in the original bank and +3.18 in the replication bank. Paired 95% intervals were +1.35 to +4.84 and +1.63 to +4.97. The differences versus shuffled experience were +4.14 and +2.98, with intervals +0.92 to +7.54 and +0.86 to +5.16. Center exceeded Pong by 1.89 and 6.38 at the same budgets. The 256 paired exploration draws each reused the same chronological physical practice course. The intervals resampled those paired draws and the distinct held launches within each release width. [Complete budget curves](adaptive-jog.json.gz)

## Elapsed time and the first legal practice point

| Held bank | Initialization | 15 s | 30 s | 60 s | 90 s | Median elapsed through the first legal practice point |
|---|---|---:|---:|---:|---:|---:|
| Original | Pong duration | 9.3 | 12.2 | 10.8 | 16.4 | 14.6 s |
| | Flat | 7.0 | 6.7 | 7.0 | 12.4 | 10.3 s |
| | Shuffled Pong duration | 5.3 | 6.3 | 5.9 | 11.0 | 7.8 s |
| | Center | 11.0 | 11.0 | 11.0 | 11.0 | 5.4 s |
| Replication | Pong duration | 11.0 | 16.4 | 14.3 | 16.5 | 17.4 s |
| | Flat | 5.8 | 9.1 | 11.1 | 11.8 | 19.5 s |
| | Shuffled Pong duration | 3.0 | 8.8 | 10.9 | 12.2 | 21.0 s |
| | Center | 18.0 | 18.0 | 18.0 | 18.0 | 5.1 s |

Checkpoint time is the sum of actual selected arm frames at 30 Hz. Each decision precedes the receipt of its elapsed duration. Between completed points the policy is still the most recent completed policy. The 64-trial Pong streams used a mean of 92.3 and 90.6 seconds of simulated practice. Center used 101.6 and 97.8 seconds on those same courses. The four pooled source sets supplied 1,344 native episodes on 192 distinct launch seeds spanning 450,304 Pong frames, 62.5 simulated minutes at 120 Hz. [Training traces and clocks](adaptive-jog.json.gz)

## What the duration learner acquired

The pooled native model initially requested a 15.3 cm left or right jog on every detected target approach. It made 17 contacts and zero legal returns on each bank. The elapsed-time feedback carried useful target information. Across all physically measured options, the longest-running choice landed a legal opening on every launch where some choice could do so, 32/64 in the original bank and 38/64 in the replication. The learner's own history contained just one duration per practice launch. Its improvement after that exposure mostly recovered from an unreliable initial movement size. After 64 practice trials it surpassed center on the original bank and remained below center on the replication bank. [Original physical action bank](anonymous-action-arm.json.gz) · [Replication physical action bank](anonymous-replication-arm.json.gz)

The 72-start native initialization showed the same early ordering. It added 1.57 and 2.52 returns over flat, and 2.47 and 3.99 over its fixed duration-label shuffle. It remained below center by 3.49 and 7.97. Its source training comprised 504 episodes and 162,491 frames, about 22.6 simulated minutes. [Native training histories](anonymous-action-source.json.gz) · [Independent native sets](anonymous-replication-pong.json.gz)

Each adaptive learner used one optical contextual kernel and three effective prior observations per action. We restored the native context average to the already fitted centered log durations. This preserved every initial choice in the original published source policy. The flat prior assigned that same context average to each action. Only the true source or shuffled prior varied by action. The same transform of target elapsed duration updated the selected action and its reflection with half a sample each. The physical banks supplied legality solely to evaluation. Replacing every unchosen practice duration left all six audited histories unchanged, covering the three adaptive modes on both banks. [Replay and fitting code](../../scripts/experimental/tt23_adaptive_jog.py)

Run `python -m scripts.experimental.tt23_adaptive_jog` to reproduce the analysis from the two existing physical banks. It uses the project Python environment with one OpenBLAS and OpenMP thread.
