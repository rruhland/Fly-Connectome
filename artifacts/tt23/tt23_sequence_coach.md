# Fixed Pong openers with a learned physical practice partner

The test repeats all 24 receiving starts from banks A and B. It keeps the original anonymous stereo opening, narrow motor intervention, tick-25 handoff, and frozen inverse-flight continuation with seven optical examples. At the far arm, the flight-and-timing coach ranks the stock menu of legal candidate replies with its fitted estimate of whether this near arm can return them. The body, release, far target profile and strict referee are shared with the archived run. The selected Pong-duration and neutral opening policies both play each start. The first two starts also check the native-score-permuted opener. [Opening benchmark](tt23_sequence.md) · [Coach fit](coach-practice.json.gz) · [Coach protocol](coach.md)

When a candidate in the stock tape passed the approximate legal-return filter, both arms play live from the initial release. For the other branches, the evaluator replays the archived near commands while the installed far opponent computes its commands live. A fail-fast hook checks that the learned ranking function stays inactive. The audit requires every far command, the full stereo digest, every referee event, terminal outcome and point duration to match. Both methods retain all starts, including failures to track or return. The live comparisons verified the shared physical prefix through the first changed far command. [Comparison runner](../../scripts/experimental/tt23_sequence_coach.py)

The main endpoints are points reaching ten and twenty alternating legal returns, and active frames to the first ten-return point. The comparison uses the original course order and tracks threshold counts for both far policies with matched openings. A fresh learned-coach confirmation on the existing C/D launch generators is reserved if duration produces at least two more ten-return points than neutral, or at least one more twenty-return point with at least as many ten-return points. That confirmation uses the frozen near and coach checkpoints. [Original C/D generators](tt23_sequence_replication.json.gz)

## Results

| Outcome across 24 starts | Duration, geometry | Duration, learned coach | Neutral, geometry | Neutral, learned coach |
| --- | ---: | ---: | ---: | ---: |
| Opening contacts | 23 | 23 | 11 | 11 |
| Legal opening returns | 9 | 9 | 8 | 8 |
| All SO-101 legal returns | 31 | 39 | 28 | 64 |
| All far partner legal returns | 30 | 38 | 27 | 63 |
| Points won | 1 | 1 | 1 | 1 |
| Points reaching 10 alternating returns | 2 | 5 | 2 | 4 |
| Points reaching 20 alternating returns | 0 | 0 | 0 | 3 |
| Longest alternating exchange | 12 | 14 | 16 | 46 |
| First point reaching 10 | 13 | 1 | 13 | 13 |
| Cumulative control frames to first 10 | 1,187 | 195 | 884 | 879 |
| Simulated seconds to first 10 | 39.57 | 6.50 | 29.47 | 29.30 |

The learned coach preserved all four ten-return points from the geometry comparison and added three for duration and two for neutral. On overall start 1, seed `26147261`, duration reached ten with the learned coach and eight with geometry. Neutral reached zero with either partner on that start. The first far command changed at frame 131. [Matched scores and commands](tt23_sequence_coach.json.gz)

Neutral reached 22, 46 and 30 alternating returns on overall starts 17, 18 and 23. The 46-return point on seed `777728055` ran for 827 frames, with 23 legal returns from each arm. The far command first changed at frame 28. The near arm ended the point by landing on its own half. The same neutral opening with the geometry partner reached six alternating returns. The broadcast replay shows both arms, the moving ball and the live referee counts over the complete point. [Matched scores and commands](tt23_sequence_coach.json.gz) · [46-return replay](tt23_sequence_coach_46.mp4) · [Video verification](tt23_sequence_coach_46.json)

Nineteen points were played live, including both permuted opening checks, which each had zero returns. Thirty-one inactive branches matched their archived physical and RGB traces. All 50 two-arm command tapes reproduced point duration, strict referee events, terminal outcomes and alternating counts on replay. The duration policy led by one point at ten. Neutral led by three points at twenty. The C/D confirmation gate stayed closed. [Replay checks](tt23_sequence_coach.json.gz)

## Reproduce

Use the MuJoCo environment and native component library from the original opening study. An existing receipt resumes its missing branches. [Environment setup](tt23_sequence.md#reproduce)

```sh
python -m scripts.experimental.tt23_sequence_coach
python -m scripts.experimental.tt23_sequence_coach --verify
python -m scripts.experimental.tt23_sequence_coach_movie
```
