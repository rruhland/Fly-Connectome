# What a missed shot taught the direct joint controller

A simulated SO-101 used one scoreless point to recognize 100 ms of stereo camera delay. On 16 held releases it then made five legal opening returns. With the same motor policy and uncompensated images it made one. On a different eight-shot course, timing continued to update through four unannounced delay conditions. It raised total legal arm returns from 14 to 19, including two extra returns after the opening. [Matched points and first-delivery checks](zero-reward-results.json) · [Watch one point with both controllers](timing-6323.mp4)

## What learned during play

The baseline maps the observed ball position, velocity and joint angles to slew-limited joint commands at 60 Hz. Its weights came from 30 rewarded teacher strikes on five releases. It has a configured ready pose, a stereo ball detector, nominal optical flight dynamics and an incoming-motion gate. It issues commands without choosing a stroke or computing a future interception point. This run kept those weights frozen. The new learning changed the state estimate supplied to them.

1. Twenty previously played, scoreless points on four practice releases supplied 1,063 simultaneous encoder and visible paddle measurements, plus 375 short sequences of ball position and velocity measured through RGB. Local regressions learned what the paddle looks like as the joints move and how an observed ball state usually changes over the next few camera frames. This stage uses synchronized cameras and learns without a stroke, contact or outcome label. The experimenter selected recordings with zero legal arm returns. [Source receipt](lag-bank.json) · [Fitted observation bank](lag-bank.npz)
2. In the changed game the camera stream arrived late. The learner compared visible paddle motion to its own recent encoder motion over candidate ages from zero to five frames. On the first calibration point, release 10898, it identified three frames after 62 motor ticks (1.03 simulated seconds) and finished with zero legal arm returns. Its inputs did not include capture time or the simulator's camera delay. [Calibration trace](lag-fit.json.gz)
3. The learned age selects how far to advance a stale observation to estimate the ball **now**. A fixed-age test carried the estimate to 16 unseen releases. A separate test started with age zero and updated from imagery during play. It retained the age between points and cleared the short motion history at ordinary point boundaries. All results are from physical games against the moving practice opponent.

## Actual play

| Camera delay | Points | Fixed timing | Learned timing | What changed |
|---|---:|---:|---:|---|
| 100 ms, new roster | 16 | 1 | 5 | Four additional opening returns. No second returns. |
| 100 ms, earlier reference roster, estimating online | 8 | 3 | 3 | One gain and one loss. |
| 0 ms, switching course, first block | 8 | 5 | 5 | Each point retained its score. |
| 100 ms, switching course | 8 | 1 | 3 | Two additional opening returns. |
| 33 ms, switching course | 8 | 3 | 6 | One additional opening and two later returns. |
| 0 ms, switching course, final block | 8 | 5 | 5 | Each point regained its original score. |

Counts are the SO-101's own legal returns. The switching course repeats the same eight openings and opponent seeds across the four blocks. One of its release IDs also appears on the 16-shot roster with a different opponent seed. On the switching course the adapted controller improved five point scores, tied 27, and shortened none. The film shows release 6323 during the 33 ms block. Both panels replay the recorded joint commands at simulation speed. The left panel holds its final frame when its point ends.

With the cameras restored on the separate earlier eight-point roster, online re-estimation recovered the baseline score on all eight. Three command tapes matched exactly. An independent replay of all 48 adapted games from the 16-shot and switching tests reproduced the referee and placement stream, with zero CAD penetrations in 56,138 physics steps. [New-roster replay](lag-novel-carry-audit.json) · [Switching replay](switch-online-audit.json)

A control using the same learned camera age with nominal flight equations landed three of the 16 openings. On 430 time-matched observations from the adapted games, stale RGB had median position error of 41.0 cm relative to timely RGB. The learned reconstruction reduced this to 7.0 cm. The nominal equations did better on this metric at 1.1 cm, despite making fewer returns. These are optical comparisons to the timely camera estimate. The game scores are the independent physical outcome. [Sensor audit](sensor-audit.json)

## What transferred to the motor skill

With changed actuator response and timely cameras, reward-free body calibration recovered the baseline's six legal returns from five, including one additional second reply. With both actuator and camera changes present, the frozen arm landed three, correcting the camera alone landed four, and composing the independently learned camera and body corrections landed five. All five were opening returns. [Actuator measurements](body.md) · [Combined physical test](mixed.md)

Three approaches that tried to improve the nominal striking policy directly gave mixed or negative results.

| Observation or training change | Earlier 8 | New 16 |
|---|---:|---:|
| Original raw policy | 3 | 6 |
| Visual directions that change slowly over ordered frames from misses | 5 | 5 |
| Same calculation with shuffled frame pairs | 2 | 3 |
| Translate the ball, joint positions and targets together using a learned local body response | 4 | 5 |
| Translate only the observed ball in the training examples | 6 | 6 |
| Encode current ball position relative to the body's estimated movement | 2 | 3 |
| Same relative coordinates with shuffled encoder-to-paddle training pairs | 2 | 8 |

The temporal readout follows the idea of learning visual directions that vary slowly over successive observations. Here the learner solves a small eigenvalue problem and refits a direct joint output by regularized least squares. The shuffled comparison uses the same observations. Ordered frames helped relative to shuffling on both courses, but the advantage over the original motor policy failed to repeat. The body-relative observation experiment also failed its shuffle control. None of these became the default motor policy. On a separate direct controller, assigning penalties to unsuccessful exploratory joint offsets issued the same held commands as learning from the successful offsets alone. Both scored five versus six for the fixed policy across 16 points. [Slow Feature Analysis](https://pubmed.ncbi.nlm.nih.gov/11936959/) · [Joint-offset experiment](visual.md) · [Relative observation experiment](../../scripts/experimental/tt35_relative.py)

## Reproduce

Use the environment and measured launch file from the [direct joint reproduction](../tt33/joint.md). From the repository root, set the following paths and run the comparison. The `capture` command can optionally regenerate the archived synchronized observation bank.

```sh
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
out=artifacts/tt34
feeds=/tmp/tt25-rallies.json
mod=scripts.experimental.tt34_latency
python -m "$mod" capture --bank /tmp/tt34-bank.npz
python -m "$mod" online --course artifacts/tt31/full-plan.json.gz --feeds "$feeds" --limit 1 --output /tmp/age.json.gz
python -m "$mod" carry --course "$out/body-novel-plan.json.gz" --feeds "$feeds" --state /tmp/age.json.gz --output /tmp/carried.json.gz
python -m "$mod" frozen --course "$out/body-novel-plan.json.gz" --feeds "$feeds" --output /tmp/frozen.json.gz
for mode in online frozen; do
  python -m "$mod" "$mode" --course "$out/switch-plan.json.gz" --feeds "$feeds" --blocks 0 3 1 0 --output "/tmp/switch-$mode.json.gz"
done
python -m scripts.experimental.tt33_joint verify /tmp/carried.json.gz /tmp/carried-audit.json --cad
python -m scripts.experimental.tt34_sensor_audit /tmp/carried.json.gz /tmp/current-state-audit.json
python -m scripts.experimental.tt34_results
python -m scripts.experimental.tt34_replay_pair
```

The nominal motor experiments use the baseline's 1,080 training samples from `tt33_joint capture --raw`. Fit `tt34_temporal ordered` or `shuffled` and evaluate with its `play` command. For the spatial experiment fit `tt34_translate` and use `tt33_joint evaluate`. The body-relative experiment uses `tt35_relative learned` or `shuffled` then `tt35_relative play`. The two course files are `artifacts/tt32/fresh-plan.json.gz` and `artifacts/tt34/body-novel-plan.json.gz`.
