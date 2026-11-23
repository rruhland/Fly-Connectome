# Online rally practice on the solid compact court

The near SO-101 played from the start of every point on the TT25 compact court with tabletop contact enabled for its links. The far arm used the same five-axis live planner and learned reply ranking as TT25. The simulator generated the opening far serve. Every near command came from the stereo camera pair, measured near joints and previously issued commands. The pupil could read camera capture timestamps. It received the legal near-return count when each point ended. [TT25 court and opponent](../tt25/recorded-benchmarks.md) · [Experiment](../../scripts/experimental/tt27_compact.py)

The online learner selected among the existing combined, retina and optical feedback fits before each point. All three branches started with the same preference for the combined fit. The retained branch kept discounted scores across the complete course. The reset branch restarted the scores at each change. The frozen branch ran the combined fit throughout. The learner received no condition label. The three branch choices were paired on each fresh release, and identical choices reused the same verified deterministic tape. No branch replayed a tutor command during assessment. [TT24 training data and fits](../tt24/rally-improve.md)

Across consecutive blocks we changed lateral serve placement by adding 0.18 to its lane parameter, changed the table contact damping from −24 to −25.35 in MuJoCo's direct `solref` format, then delivered both camera images one 30 Hz frame late. Each block retained the previous changes. The final block restored the original configuration. In the delayed block every fit extrapolated its visible ball position using camera capture age and recent measured motion.

## Held points

Two fresh changed-rebound points reached **14 alternating legal returns**, seven per arm. The illustrated point lasted 295 frames, or 9.83 seconds of play. The other lasted 275 frames, or 9.17 seconds. All three branches chose the combined fit on both points. The previous TT25 stock policy reached 12 alternating returns on a separate 24-start course. [Watch the compact-court point at normal speed](compact-live.mp4) · [Verified point and video timing](compact-live.json) · [TT25 stock course](../tt25/recorded-benchmarks.md)

| Block, four fresh releases each | Retained near / far | Reset near / far | Frozen near / far |
| --- | ---: | ---: | ---: |
| Stock | 2 / 2 | 2 / 2 | 4 / 4 |
| Changed placement | 7 / 7 | 4 / 4 | 7 / 7 |
| Changed rebound | 15 / 15 | 15 / 15 | 15 / 15 |
| One-frame camera delay | 2 / 1 | 0 / 0 | 2 / 1 |
| Restored stock | 5 / 5 | 8 / 8 | 5 / 5 |
| **Total** | **31 / 30** | **29 / 29** | **33 / 32** |

The retained, reset and frozen branches reached a second near return on 7, 6 and 8 of 20 releases. Each reached ten alternating returns on three releases. Retention matched the frozen fit in each changed block and reached a second near return on the second late-camera point, after one release with zero near returns. Reset made zero near returns during that block. Reset reached a second return on the second restored-stock point, while retention and frozen both took four points. The frozen fit had the most first and second returns overall on this course. [All 20 paired starts and 31 verified physical tapes](compact-assessment.json.gz)

The three branches averaged 5.85, 5.42 and 5.53 abrupt adjacent reversal events per second, respectively. The largest observed joint speeds were 5.51, 6.25 and 6.10 rad/s. The illustrated 14-return point had 3.56 abrupt reversals per second and peaked at 5.43 rad/s. Motor speeds were sampled at 30 Hz. [Recorded command and velocity metrics](compact-assessment.json.gz)

## Comparing the feedback choices

The separate eight-release pilot ran five fixed feedback options and replay-verified all 40 branch tapes. The stock and delayed blocks each had four starts. The combined fit made nine near returns and reached a second near return on three starts. The retina fit also made nine near returns and reached a second on two. The retina fit reached eight alternating returns at its longest and handled two of the four late-camera starts with one legal return apiece. The same late starts yielded zero returns with the combined fit. Raising the extrapolation factor from 1.0 to 1.5 yielded zero late-camera returns. Applying 14% carryover from the preceding motor command reduced total near returns to three, with a single return on each successful start. [Pilot tapes and scores](compact-development.json.gz)

Across the pilot, the retina fit had 4.52 abrupt adjacent reversal events per second compared with 8.06 for the combined fit. A reversal counts a change directly from above +0.1 to below −0.1 or the reverse, on any of the five active normalized command channels. Peak measured joint speeds were 5.09 and 5.69 rad/s, respectively. These are simulated control-tick samples for different realized rallies. The retina fit was the smoother fallback under this test, and the combined fit had the better second-return rate in the pilot.

## Reproduce

```sh
PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
  python -m scripts.experimental.tt27_compact --count 4 --streams 1
PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
  python -m scripts.experimental.tt27_compact --movie
```
