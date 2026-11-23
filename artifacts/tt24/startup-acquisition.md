# Wrist acquisition and the second rebound

Most archived startup losses happened after a real first rebound. On independently randomized joint starts, the most visible failure was an angled first return that sent the ball past the paddle before the next descent.

## Archived startup

| Original wrist run | Drops | First qualifying rebound | Second qualifying rebound | Median first-contact tilt | Median second plane crossing relative to the rim |
|---|---:|---:|---:|---:|---:|
| Gravity-switch, survived 30 s | 7 | 7 | 7 | 1.2° | 48 mm inside |
| Gravity-switch, ended before 4 s | 5 | 4 | 3 | 3.6° | 15 mm inside on the four observed crossings |
| Ordinary posture | 12 | 11 | 11 | 2.4° | 46 mm inside on the 11 observed crossings |
| Expanded posture | 24 | 23 | 0 | 11.7° | 210 mm outside on the 23 observed crossings |

The ordinary and expanded starts had virtually the same median first-12-frame optical position error, 9.9 and 9.7 mm. The expanded paddle sent the ball sideways at a median 0.724 m/s after its first qualifying hit, versus 0.098 m/s at ordinary posture. Every measured second crossing in the expanded group passed 45–622 mm outside the paddle rim.

The short gravity-switch group included a handle strike immediately after an accurate first face observation and initial silhouette clipping on a different drop. On seed 997103512, the upper edge clipped the orange sphere. At tick one the observer accepted the fragment and put the estimated ball more than 200 mm from its true position. It then recovered the ball and made one rebound, but the next descent crossed 10 mm beyond the rim. After other first rebounds the late losses included poor pickup as the ball moved away, a table contact and a wrist camera housing strike. [Wrist frames and optical error](startup-acquisition-vision.png)

The existing privileged controller, at the learned actor's 0.65 command cap, produced 11–15 consecutive rebounds on each of the five archived launches that originally ended before a gravity change. On the 24 expanded archived starts it made a second rebound in 19 and at least ten in 9. This establishes physical recovery on those launches under the same simulator and referee.

## Paired causal tests

| Cohort and sensor | First rebound | Second rebound | Longest run at least 10 | Mean longest run | Reached cap |
|---|---:|---:|---:|---:|---:|
| Fresh standard, original wrist | 11/12 | 11/12 | 10/12 | 20.83 | 10/12 |
| Fresh standard, public startup gate | 11/12 | 11/12 | 10/12 | 20.50 | 10/12 |
| Fresh standard, true state for first 12 ticks | 11/12 | 11/12 | 11/12 | 20.33 | 11/12 |
| Fresh standard, full state | 11/12 | 11/12 | 11/12 | 22.67 | 11/12 |
| Fresh expanded, original wrist | 11/12 | 0/12 | 0/12 | 0.92 | 0/12 |
| Fresh expanded, public startup gate | 11/12 | 0/12 | 0/12 | 0.92 | 0/12 |
| Fresh expanded, true state for first 12 ticks | 11/12 | 1/12 | 0/12 | 1.00 | 0/12 |
| Fresh expanded, full state | 11/12 | 1/12 | 0/12 | 1.00 | 0/12 |
| Five archived failures, original wrist | 4/5 | 3/5 | 0/5 | 2.20 | 0/5 |
| Five archived failures, public startup gate | 4/5 | 4/5 | 1/5 | 5.80 | 1/5 |
| Five archived failures, true state for first 12 ticks | 4/5 | 4/5 | 3/5 | 13.80 | 3/5 |
| Five archived failures, full state | 4/5 | 4/5 | 3/5 | 15.20 | 3/5 |

The public startup gate held the joints still when the first circle touched an image border or was missing. It resumed the original actor after two supported detections or 12 ticks. It used only the wrist image and own encoders. On the clipped archived seed 997103512 it waited eight ticks and lifted the run from one rebound to twenty. True-state input for the first 12 ticks or throughout produced 26 on that seed. On the fresh cohorts the gate triggered on two standard and two expanded starts and left the second-rebound and survival counts unchanged.

The first-12 diagnostic kept the wrist observer running on its own real images throughout. On fresh seed 227447384 it lifted the run from 2 to 21 and reached the eight-second cap despite continuous initial visibility. The second descent crossed 54 mm inside the rim under the brief state substitution, versus 22 mm under the original wrist. A separate fresh handle strike remained a zero in all four conditions. With expanded posture, perfect ball and paddle state still left the frozen actor at zero or one qualifying rebound on 11 of 12 fresh drops.

The standard paired runs used the original gravity schedule through the first change at four seconds and a cap of eight seconds. The expanded paired runs used nominal gravity and a cap of five seconds. The full-state diagnostics also received the scheduled gravity factor. The public gate and brief state-substitution conditions retained the original causal visual gravity estimator. All conditions used the same frozen donor actor. No policy fitting used the archived or held observations.

All 48 archived action tapes reproduced their counts and terminal frames, and the 12 gravity tapes also reproduced individual qualifying contact times. All 116 paired tapes and 29 privileged physical feasibility tapes reproduced their full physical event records on exact headless replay.

Reproduce from the repository root with `MUJOCO_GL=egl python -m scripts.experimental.tt24_startup_audit`. Run again with `--held`, `--reachability` and `--figure` to generate the separate receipts and figure.
