# Native Pong opening preferences in a continuous SO-101 point

The same target-trained clock stroke opens every condition. From eight live fixed-stereo RGB frames under zero commands, the existing anonymous-motion estimator locates an approaching track. The Pong-duration policy uses signed relative displacement and motion gains selected by native Pong episode time. Neutral uses zero gains. The native-score-permuted control uses the corresponding frozen selection from the Pong checkpoint's shuffled-score comparison. Each condition applies its gains only to the lateral placement of the opening stroke. [Opening transfer](../tt22/source-transfer.md) · [Native Pong qualification](../tt22/duration-pong.json.gz) · [Clock stroke](../tt22/openloop-practice.json)

The point starts with the native 0.45-spread receiving release, with balanced randomized lateral lanes. The physical five-axis partner stays active from the first frame through the referee's point outcome. At control tick 25, after the clock stroke's full taper, the identical stereo inverse-flight controller takes over using seven frozen optical observations from four earlier points. Its command-driven arm model has advanced from the first control frame. The handoff uses the clock, and the learner tracks live stereo throughout. [Integrated runner](../../scripts/experimental/tt23_sequence.py) · [Optical checkpoint](../tt22/rally-learning-checkpoint.json.gz) · [Far partner](../tt22/tt22_opponent-learned.json.gz)

The schedule contains 24 fresh receiving starts in two independent 12-start banks. Each bank has six moderate and six wide launches and balances lane sign within both strata. The three policies face each start. The launch order interleaves banks and strata and rotates policy order. Outcome counts include tracking failures and missed strikes. Opening return credit follows the physical contact made on or before the handoff, even when its legal landing occurs later. The first ten-return time accumulates simulated control frames up to the tenth legal landing, including earlier missed points.

On the first two archived starts per stratum, the original MotorArena and the integrated five-axis arena produced identical ball position, velocity and spin, near arm joints and blade position at tick eight. Their anonymous stereo mover estimates also matched exactly. All six opening command tapes on the first two fresh starts matched the original opening implementation through handoff. The integrated trials use their own live RGB prefixes and require exact within-start agreement across conditions. [Archived opening prefixes](../tt22/duration-transfer.json.gz)

## Results

| Outcome across 24 starts | Pong duration | Neutral | Native score permuted |
| --- | ---: | ---: | ---: |
| Opening paddle contacts | 23 | 11 | 0 |
| Legal opening returns | 9 | 8 | 0 |
| All SO-101 legal returns | 31 | 28 | 0 |
| All far partner legal returns | 30 | 27 | 0 |
| Points won | 1 | 1 | 0 |
| Points reaching 10 alternating returns | 2 | 2 | 0 |
| Points reaching 20 alternating returns | 0 | 0 | 0 |
| Longest alternating exchange | 12 | 16 | 0 |
| First point reaching 10 | 13 | 13 | — |
| Cumulative control frames to first 10 | 1,187 | 884 | — |
| Simulated seconds to first 10 | 39.57 | 29.47 | — |

All 11 neutral opening contacts also occurred with Pong duration. Pong duration added contacts on 12 other starts. Legal opening returns favored Pong duration alone on six starts and neutral alone on five. Three starts had a legal opening return in both conditions. The anonymous tracker recovered an opening mover on 23 of the 24 matched starts. [Complete matched receipt](tt23_sequence.json.gz)

| Overall start | Bank and stratum | Seed | Pong duration length | Neutral length | Permuted length |
| ---: | --- | ---: | ---: | ---: | ---: |
| 13 | A moderate | 179271280 | 12 | 16 | 0 |
| 18 | B moderate | 777728055 | 10 | 6 | 0 |
| 23 | A wide | 749679286 | 0 | 10 | 0 |

These are all starts with at least one ten-return point. Bank A contributed one such point for Pong duration and two for neutral. Bank B contributed one for Pong duration. Both policies first crossed the threshold on the same start, with the tenth landing at frame 218 for Pong duration and frame 193 for neutral. The 72 two-arm command tapes reproduced the exact point duration, strict referee events, terminal outcomes and alternating counts. [Per-point controls, results and replays](tt23_sequence.json.gz)

## Replication schedule

Two additional banks were frozen while the original course was running. Banks C and D use generator seeds `1750364263` and `194462034`, with six moderate and six wide launches per bank and balanced signs in each stratum. The manifest contains all 24 fresh starts, the same three policies, and the same macro, handoff and checkpoint hashes. Run this course once if the duration policy reaches ten on at least one of the original 24 starts and exceeds both controls on the count of ten-return points. Report all replication starts using the same endpoint counts and clock. [Frozen replication manifest](tt23_sequence_replication.json.gz)

The duration and neutral policies tied at two ten-return points each. The replication manifest contains zero evaluated starts.

## Reproduce

```sh
tt23_native=$(mktemp -d)
c++ -O3 -std=c++17 -shared -fPIC scripts/experimental/tt3_components.cpp -o "$tt23_native/components.so"
export TT3_COMPONENT_LIBRARY="$tt23_native/components.so"
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt23_sequence
python -m scripts.experimental.tt23_sequence --verify
```

An existing receipt resumes missing branches on the frozen schedule. Use `--output` to supply a custom path.
