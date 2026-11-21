# Flight correction and between-stroke recovery on the six-joint arm

Two corrections learned from optical departures improved the score in both complete games. Each landed zero second returns. Recentering the requested motor positions between strokes reduced the error at the next interception. It produced a second return on rival service in one complete game and a second return after own service in a separate full game. It also landed six returns in one independent held rally. The arm lost every fixture. [Side-by-side first-person replay](rotary-six-returns.mp4) · [Matched comparison](rotary-comparison.json.gz) · [Independent repeat](rotary-comparison-replication.json.gz)

## Learning from short departures

The acquisition used the 24 [TT13 optical traces](../tt13/rotary-recalled-held.json.gz) and their camera-predicted intercept times. Four coherent outgoing stereo sightings gave nine usable estimates. A drag and gravity model predicted that five of the nine arcs would fail to clear the net. Post-play referee comparison showed that these five were among the eight contacts that landed on the learner’s half. The other three own-half contacts lacked enough optical observations for this fit. [Learned parameters and sighting indices](rotary-flight-fit.json.gz) · [Post-play labels](rotary-diagnostics.json.gz)

For each short arc the fit searched for the smallest increase in forward or upward departure speed that would clear the net and land between 0.22 and 1.15 m into the opposite half. It mapped the typical feasible increments onto a forward blade-velocity objective or a pitch increment using the existing collision model. The learned forward objective was 0.0715 m per control frame. The pitch increment reached its 0.15-radian cap. Both policies kept these learned values fixed in qualification. [Player implementation](../../scripts/experimental/tt14_rotary_player.py)

All branches began with the [TT13 signed-recall checkpoint after game two](../tt13/rotary-recalled-game-2-player.json.gz). They used the same motor maps, camera tracker, serving controller, toss profile and ordinary carried updates. The extra objectives became active when the receiving servo took over visually. In the 24 held paired starts, the forward objective changed median camera-measured blade speed by −0.0014 m per frame, about −0.04 m/s. The loft branch raised measured blade pitch by 0.108 rad on the same 24 starts. On the four matched arcs with usable departure measurements in the loft branch and baseline, estimated upward launch speed rose by 0.27 m/s. [Paired optical measurements](rotary-diagnostics.json.gz)

## Held starts and complete games

Every held start reset the player's carried state. Each complete game carried ordinary updates through its independently seeded service and receiving streams. The full games alternated service to 11 points with a two-point lead. Identical model, initial state, service motion and incoming starts let the comparison isolate receiving changes. [Exact schedules and paired checks](rotary-comparison.json.gz)

| Receiving policy | First returns on 24 held serves | Later held returns | Full-game scores | First returns on rival service in games | Later returns on rival service in games | First returns after own serve in games |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Carried baseline | 7 | 1/4 | 0–11, 1–11 | 3/12 | 0/3 | 3/5 |
| Forward objective | 6 | 0/2 | 2–11, 4–11 | 4/14 | 0/3 | 4/5 |
| Loft | 7 | 0/3 | 2–11, 3–11 | 3/13 | 0/2 | 3/5 |
| Recentering | 7 | 2/6 | 1–11, 1–11 | 2/12 | 1/2 | 3/5 |

All four policies won four of the 24 held points. The baseline made a second held return on one point. Recentering made both a second and a third held return on a different point. It made the only second return on rival service in the eight initial full games. Later denominators count the rival's legal replies after the first receiving opportunity, separately by service role. [Baseline held tapes](rotary-baseline-held.json.gz) · [Recentering held tapes](rotary-recenter-held.json.gz) · [Game with a second return](rotary-recenter-game-1.json.gz)

## The receiving arm between strokes

The baseline formed an intercept plan on all ten second or later opportunities across the held cohort and the two games. It had a median 16.6 frames of planning lead and 4.5 paired stereo ball and blade sightings around predicted contact. The cameras recorded a median 164 mm depth displacement from the ready pose at the rival strike. Near the attempted interception, the median blade error from the requested position was 40 mm laterally, 96 mm in depth and 178 mm vertically. The median requested lateral reach was 190 mm and the median number of near-impact frames with a saturated motor request was eight. These observations point to recovery and available motor travel as constraints alongside occlusion and departure speed. [Per-attempt optical audit](rotary-diagnostics.json.gz)

The alternative uses the visually planned intercept time and its own integrated public commands. Starting five frames after the planned strike, and while there is no active intercept, it sends rate commands to bring the integrated requests back to neutral. Each reset command is bounded at 0.72 of the full rate. In the initial comparison the median depth displacement when the rival struck was 7 mm, median vertical error near the next planned contact was 55 mm and saturated commands fell to a median of four frames. Recentering landed three of eleven later opportunities compared with one of ten for the baseline. Those totals include own-service rallies and can differ as earlier returns change the opponent's replies. [Recovery implementation](../../scripts/experimental/tt14_rotary_player.py) · [Later-return diagnostics](rotary-diagnostics.json.gz)

On the six later replies whose incoming bounce matched across the first comparison, recentering moved the median depth displacement down from 101 to 7 mm and median near-contact height error from 159 to 112 mm. Both policies had about 15 frames of lead. Two of the six returns landed with recentering and one with the baseline. [Matched later replies](rotary-diagnostics.json.gz)

## Independent repeat

On fresh opponent serves, recentering and the baseline both landed 6/24 first returns. Recentering landed 5/8 later replies and won three points. The baseline landed 0/4 and won two points. **All five later returns came from one rally.** On that start, both players made the same first return and contacted the same second ball. The baseline's second contact landed short. The recentered player landed a total of six returns and won. In two fresh complete games, recentering scored 5–11 and 6–11 compared with 5–11 twice. A second return after own service earned the extra point. Both policies landed zero second returns on rival service in the full-game repeat. [First-person replay of the matched point](rotary-six-returns.mp4) · [Independent held point tapes](rotary-recenter-held-replication.json.gz) · [Full-game point with a second return](rotary-recenter-game-2-replication.json.gz)

An additional forward policy added the learned 0.0225 m/frame increment to the carried stroke's speed, beginning at 0.1095 m/frame and bounded at 0.112. Its median outgoing speed increased by 1.08 m/s on the five independent held starts with usable departure fits for both players. It landed 5/24 first returns and zero second returns, and scored 3–11 and 5–11. The observed speed increase covered five paired contacts. [Repeat comparison](rotary-comparison-replication.json.gz) · [Repeat optics](rotary-diagnostics-replication.json.gz)

## Verification and continuation

All 365 qualification point tapes reproduced their applied actions and events in physics. Across the comparisons, 188 shared opponent-service trajectories matched, and the requested and applied commands through the strike matched on 63 shared own tosses. The referee service events matched exactly on 62 tosses and to numerical tolerance on the remaining toss. The pace and recenter checkpoints reproduced 15 and 12 original point action tapes respectively from RGB, then each matched four additional point tapes against uninterrupted play. The recenter checkpoint retained all 180 native Pong reference tapes, 41 earned points and 195/206 later returns. [Pace resume audit](rotary-pace-resume-audit.json.gz) · [Recentering resume audit](rotary-recenter-resume-audit.json.gz) · [Recentering checkpoint](rotary-recenter-game-1-player.json.gz) · [Native Pong replay](rotary-recenter-pong.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) from the repository root. Fixture 0 is the held cohort. Each completed game writes a portable checkpoint and replays its physical action tape.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
output=/tmp/tt14-rotary
"$python" -m scripts.experimental.tt14_rotary_trial "$output" train
for mode in baseline pace loft recenter; do
  for fixture in 0 1 2; do
    "$python" -m scripts.experimental.tt14_rotary_trial "$output" play "$mode" "$fixture"
  done
done
"$python" -m scripts.experimental.tt14_rotary_trial "$output" compare
"$python" -m scripts.experimental.tt14_rotary_diagnostics "$output"
"$python" -m scripts.experimental.tt14_rotary_trial "$output" audit-resume pace 2
"$python" -m scripts.experimental.tt14_rotary_trial "$output" audit-resume recenter 1
for mode in baseline recenter pace-relative; do
  "$python" -m scripts.experimental.tt14_rotary_trial "$output" play "$mode" 0 --seed 338174957 --label replication
  "$python" -m scripts.experimental.tt14_rotary_trial "$output" play "$mode" 1 --seed 858874689 --label replication
  "$python" -m scripts.experimental.tt14_rotary_trial "$output" play "$mode" 2 --seed 1282358628 --label replication
done
"$python" -m scripts.experimental.tt14_rotary_trial "$output" compare baseline recenter pace-relative --label replication
"$python" -m scripts.experimental.tt14_rotary_diagnostics "$output" --modes baseline recenter pace-relative --label replication
"$python" -m scripts.experimental.tt14_rotary_replay "$output" "$output/rotary-six-returns.mp4"
"$python" -m scripts.experimental.tt8_pong_retention \
  "$output/rotary-recenter-game-1-player.json.gz" "$output/rotary-recenter-pong.json.gz"
```
