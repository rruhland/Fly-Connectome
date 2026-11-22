# Learning a smooth motor-displacement score from Pong duration

The source-only fit improved average withheld native Pong duration by 0.707 seconds over holding position, with a gain in all five source batches. The previous kernel gained 0.661 seconds. On 64 fresh physical arm launches, the smooth fit and its shuffled control each made one legal return. Holding position made 13. [Grouped native results and frozen model](smooth-jog-source.json.gz) · [Arm assessment](smooth-jog-assessment.json.gz)

Each of 264 archived native starts has physical outcomes for all seven displacements. Each fold excludes an entire collection batch. The fit centers log duration within each start and imposes reflection symmetry by pairing odd action powers with odd lateral context features and even action powers with even features. Training uses a closed-form ridge solve. Model selection uses the raw elapsed time of the selected physical actions in held batches. The held and shuffled policies use those same native outcomes for scoring. The 64 shuffle controls permute duration labels within a source start and preserve each permutation across every fold that uses that start. Each control gets the best result across the same nine candidate fits. None matched the observed improvement, giving a one-sided permutation estimate of 1/65.

| Mean seconds by withheld source | Original 72 | Replicate 1 | Replicate 2 | Replicate 3 | Replicate 4 | First returns across 264 starts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hold position | 2.703 | 2.587 | 2.748 | 2.684 | 3.003 | 56 |
| Earlier kernel scorer | 3.246 | 3.390 | 3.107 | 3.680 | 3.607 | 91 |
| Selected smooth scorer | 3.104 | 3.467 | 3.135 | 3.766 | 3.791 | 86 |

The physical transfer bank uses independent schedule `846093157`. Thirty-two launches come from each of the moderate and wide release families. Each launch contributes eight initial camera readings and seven physical arm trials using the existing swing primitive and displacement conversion. All source fitting and model selection precede the arm assessment. The same bank can be scored by another frozen source policy. [Physical arm bank](smooth-jog-arm.json.gz)

| Frozen arm policy | Contacts | Legal returns | Moderate legal | Wide legal |
| --- | ---: | ---: | ---: | ---: |
| Smooth source fit | 2/64 | 1/64 | 0/32 | 1/32 |
| Source labels shuffled within start | 2/64 | 1/64 | 0/32 | 1/32 |
| Hold position | 25/64 | 13/64 | 8/32 | 5/32 |

The source model chose every available displacement magnitude in grouped native validation. In the arm bank, both learned and shuffled models chose the largest magnitude on all 61 starts with identifiable optical motion. All policies held position on the other three starts. Smooth action scores preserved the same extreme-magnitude failure on the transferred observations. [Per-launch choices and outcomes](smooth-jog-assessment.json.gz)

Recreate the source fit with `python -m scripts.experimental.tt23_smooth_jog`. Collect the physical bank and score the frozen smooth fit with `python -m scripts.experimental.tt23_smooth_jog_arm`.
