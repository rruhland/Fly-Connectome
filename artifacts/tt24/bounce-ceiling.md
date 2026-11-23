# Privileged paddle feasibility

The evaluator reads MuJoCo ball and paddle state and the robot's joint Jacobian. It forecasts the next descending contact, steers the paddle normal against lateral drift, and times a smooth paddle rise to restore flight energy. The action channel, rigid-body dynamics, and referee are the benchmark defaults. The evaluator limits its commands to 70% of the native normalized joint-rate cap. Its results are separate from unaided learning.

| Arm | Development consecutive rebounds | Fresh consecutive rebounds | Fresh at least 10 | Fresh reached 5 s | Fresh median rise |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 14.69 mean, 14–16 range | 14.88 mean, 14–15 range | 16/16 | 16/16 | 12.3 cm |
| Alternate | 13.75 mean, 2–15 range | 13.19 mean, 2–15 range | 14/16 | 14/16 | 12.6 cm |

The fresh held seeds were 243001–243016 on each arm. The two alternate-arm losses had runs of four and two. Both ended at the table. At the closest sampled downward face-plane crossing, their radial offsets were 8.36 and 14.39 cm against a 7.6 cm blade radius. At least one servo was at its force cap in both samples. The stock and alternate seed 243001 command tapes each replayed all 300 steps exactly and scored 15 consecutive rebounds. [Outcome data](bounce-ceiling.json.gz) · [Privileged command and state tapes](bounce-ceiling-trajectories.json.gz)

The separate reset screen paired stationary and privileged control on 192 held drops per arm. Stationary control scored zero on eight stock and seven alternate drops. The ball had at least five matching orange pixels around its true camera projection on all 15 release images and all 190 sampled pre-impact frames. The approximate projected disc fit fully in 13 release images. The privileged controller qualified the first impact on all 15 drops. [Reset screen](bounce-ceiling.json.gz)

## Shorter arcs

A separate profile targeted a 65 mm rise and used stronger joint regularization. Its 16 fresh held seeds per arm were 246001–246016. All 32 episodes reached five seconds. Stock reached a maximum of 20 consecutive rebounds, with a median of 18.5 and 15 of 16 reaching at least 10. Alternate reached a maximum of 19, with a median of 17 and 14 of 16 reaching at least 10. Median verified rises were 7.3 and 7.7 cm, respectively. Runs on some seeds included unqualified impacts, so the ranges were 7–20 on stock and 5–19 on alternate. [Short-arc outcomes](bounce-ceiling-short.json.gz) · [Privileged short-arc tapes](bounce-ceiling-short-trajectories.json.gz)

Run `python -m scripts.experimental.tt24_bounce_ceiling` to regenerate the standard profile and add `--short-arc` for the second profile.
