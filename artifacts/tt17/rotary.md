# Giving the rotary arm a deadline at second contact

The camera forecast was usually available and accurate before the second collision. Weighting the collision deadline more heavily lost two of four distinct second-return points on 40 identical held serves. It gained one rival-service second return and one point across three complete games. Reusing camera-measured first-stroke blade speed also failed to add distinct second-return points on a separate cohort. [Matched comparisons](rotary-second-comparison.json.gz)

## Where the second reply was lost

Seventeen second-return opportunities from the TT14 and TT15 rotary archives were replayed from their recorded commands. The audit located the opponent's second incoming trajectory and reconstructed its crossing of the planned contact plane. It compared that crossing with the player's earlier stereo forecasts and compared the physical blade with the ball at arrival. The physical state and referee labels stayed in the post-play audit. [Historical point measurements](rotary-second-diagnostics.json.gz)

Sixteen approaches had an available forecast at least eight control frames before arrival. Their median absolute time error was 0.23 frames, and their median absolute position errors were 4 mm laterally and 27 mm vertically. The timing error fell to 0.07 frames at four frames of lead. Four approaches missed despite an eight-frame forecast within two frames and 80 mm in the lateral and vertical plane. At eight frames their raw stereo blade heights already differed from the commanded heights by 29–217 mm. The physical difference was 35–215 mm. At arrival their vertical ball-to-blade offsets ranged from 83 to 350 mm. One other miss lacked a usable forecast until the ball had reached the plane. [Per-frame audit](rotary-second-diagnostics.json.gz)

Twelve balls contacted the blade. Four landed and eight failed. Six failures bounced on the learner's half. After separation the median forward ball speed was 1.27 m/s for the eight failures and 4.45 m/s for the four legal returns. The results point to blade arrival and the achieved collision as constraints even with a timely ball forecast. [Recorded outcomes and departures](rotary-second-diagnostics.json.gz)

## Shifting the predictive objective before the deadline

The alternative used the same camera tracker, ball forecast, learned motor response, blind recentering, opening stroke and service controller as the baseline. After an earlier receiving attempt, it assigned more weight to the predicted blade pose near the later collision deadline and less to poses beyond it. The rule used the optical strike forecast and estimated actuator latency, activating inside the predictive horizon while a new command could still reach contact. [Controller](../../scripts/experimental/tt17_rotary_player.py)

Both branches began from the [TT16 inherited checkpoint after game two](../tt16/rotary-first-base-two-player.json.gz). The held cohort reset before each of 40 matched rival serves. Three independent full games kept ordinary adaptation running and used separate seeded streams for own and rival service. Ordinary, delayed, soft and mixed points were balanced in blocks of four. [Runner](../../scripts/experimental/tt17_rotary_trial.py)

| Policy | Held first returns | Held second returns and opportunities | Held points won | Full-game scores | Second returns on rival service in games | Second returns after own service in games |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Inherited | 15/40 | 4/10 | 5 | 3–11, 5–11, 1–11 | 0/5 | 2/3 |
| Collision deadline | 15/40 | 2/10 | 5 | 4–11, 5–11, 1–11 | 1/5 | 2/3 |

All 40 held opening action sequences agreed through first contact, and all ten second trajectories matched. The deadline controller was active on 84 held frames across ten points. It retained two of the inherited player's four second-return points. Across the 13 matching second trajectories in held and complete play, the median absolute vertical blade-to-ball gap increased from 94 to 139 mm. The median forecast time error eight frames before arrival was 0.30 frames for both players. [Inherited held tapes](rotary-second-base-held.json.gz) · [Deadline held tapes](rotary-second-deadline-held.json.gz) · [Matched second-contact audit](rotary-second-comparison.json.gz)

The one added full-game point came on a delayed rival serve. Both players touched the second reply at frame 76. Under deadline weighting it returned legally and won. The two branches already had different action histories early in that point, and the second incoming bounce differed. [Inherited game](rotary-second-base-one.json.gz) · [Deadline game](rotary-second-deadline-one.json.gz)

## An alternative based on realized forward blade speed

For the eight archived second contacts that failed, the median measured first-stroke blade pace was 1.58 m/s. The replayed blade's median forward speed near those second contacts was 0.68 m/s. The next controller estimated the actual first pace from at least three coherent stereo sightings and carried it into the later approach. If a plan clipped to the existing motor bounds predicted a shortfall of at least 0.006 m per frame near contact, and a new command could still arrive before impact, it added a forward-speed objective. The deadline and inherited controllers ran on the same 32 new rival serves. [Historical measurements](rotary-second-diagnostics.json.gz) · [Camera pace model](../../scripts/experimental/tt17_rotary_player.py)

| Policy | Repeat first returns | Repeat second returns and opportunities | Repeat points won |
| --- | ---: | ---: | ---: |
| Inherited | 10/32 | 1/6 | 5 |
| Collision deadline | 10/32 | 1/6 | 5 |
| Measured forward speed | 10/32 | 1/6 | 4 |

The measured-speed controller found a first-stroke reference on 27 starts and added its objective on 33 frames across five later approaches. On those five matched second trajectories, reconstructed blade speed near contact changed by a median of +0.005 m/s relative to the inherited controller, with individual changes from −0.042 to +0.060 m/s. All three branches sustained the same one point. The inherited controller made four returns on it. Each alternative made two. [Independent inherited tapes](rotary-second-base-repeat.json.gz) · [Independent deadline tapes](rotary-second-deadline-repeat.json.gz) · [Independent measured-speed tapes](rotary-second-speed-repeat.json.gz) · [Matched speed measurements](rotary-second-comparison.json.gz)

## Checkpoint verification

All 261 new point tapes replayed their applied actions and referee results. All 126 paired rival openings matched through the legal service bounce, and all 20 paired own-service command histories matched through the strike. The deadline checkpoint after the first full game reproduced all 15 point action tapes using RGB. After loading it, four new point tapes matched uninterrupted play. [Evaluation checkpoint](rotary-second-deadline-one-player.json.gz) · [Exact resume audit](rotary-second-deadline-one-audit.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) from the repository root.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
output=/tmp/tt17-rotary
mkdir -p "$output"
"$python" -m scripts.experimental.tt17_rotary_diagnostics "$output/rotary-second-diagnostics.json.gz"
for mode in base deadline; do
  for fixture in held one two three; do
    "$python" -m scripts.experimental.tt17_rotary_trial "$output" play "$fixture" "$mode"
  done
done
for mode in base deadline speed; do
  "$python" -m scripts.experimental.tt17_rotary_trial "$output" play repeat "$mode"
done
"$python" -m scripts.experimental.tt17_rotary_report "$output" --repeat
"$python" -m scripts.experimental.tt17_rotary_trial "$output" audit deadline one
```
