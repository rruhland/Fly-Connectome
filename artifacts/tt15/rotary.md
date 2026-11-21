# Optical recovery for a second return

Optical readiness reduced the work spent returning to neutral and maintained both second returns on a fresh 40-start cohort. They came on the same two points as blind recentering. It made zero second returns in three complete games versus one with blind recentering. Two separate changes to the later stroke produced a point win and an extra return on the one game point that blind recentering already sustained. The arm lost all 12 games. [Paired results](rotary-comparison.json.gz)

## Recovery based on the blade and the ball

Every branch started from the [TT14 recentered post-game player](../tt14/rotary-recenter-game-1-player.json.gz) and used the same six-joint arm, stereo tracker, motor models, own-service controller, and ordinary adaptation. The blind control used the existing fixed reset. After a receiving attempt, between five and 24 frames past its visually predicted impact, it returns the integrated public motor requests towards zero whenever no strike is planned. It caps each rate at 0.72.

The optical branch compares the raw stereo blade position with its calibrated ready pose. It lowers the reset rate near readiness, increases it up to 0.94 with larger visible displacement, and holds the requested posture when the blade is within 35 mm of ready, moves at most 17 mm per frame and has little pending travel. It retains the original cap during occlusion. The current optical ball estimate or intercept plan gives an arrival time. The player reserves the inherited planning horizon, estimated transport delay and the remaining motor-reset time for the next stroke. [Player implementation](../../scripts/experimental/tt15_rotary_player.py)

## Held and full-game comparison

The 40 held starts reset both players to the same checkpoint before each point. The three full games started independently from the same checkpoint and carried ordinary updates within each fixture. A separate stream matched the nth opponent serve and the nth own toss across branches. Ordinary, soft, delayed and mixed conditions were balanced in blocks of four within each stream. [Runnable experiment](../../scripts/experimental/tt15_rotary_trial.py)

| Receiver | First returns on 40 held serves | Second returns on held serves | Held points won | Full-game scores | First returns on rival service in games | Second returns on rival service in games | Second returns after own service in games |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Blind reset | 13 | 2/8 | 5 | 1–11, 2–11, 2–11 | 5/19 | 1/4 | 0/1 |
| Optical reset | 13 | 2/8 | 5 | 1–11, 5–11, 2–11 | 6/21 | 0/3 | 0/1 |
| Bounded later strokes | 13 | 1/8 | 5 | 1–11, 2–11, 3–11 | 5/20 | 1/4 | 0/1 |
| Timed feedback on later strokes | 13 | 2/8 | 5 | 1–11, 1–11, 2–11 | 3/19 | 1/3 | 0/1 |

The two held second-return seeds were 1249243906 and 1151491349 for both players. Each rally contained two legal returns and ended in a point loss. Each branch made 15 legal returns across the 40 starts. The optical method reduced nonzero recovery frames from 247 to 216 and the sum of requested rates from 426.6 to 418.1 across those same starts. [Blind held tapes](rotary-blind-held.json.gz) · [Optical held tapes](rotary-optical-held.json.gz)

The extra three game points came in the own-service-first fixture. The optical player won three of eight opponent-served points and two of eight own-served points. The blind control won one point in each service role. Opponent serves can remain comparable if one game ends earlier because the streams are indexed separately by server. [Blind game](rotary-blind-game-2.json.gz) · [Optical game](rotary-optical-game-2.json.gz)

The optical reset had an incoming estimate during two recovery-window frames in the held cohort and none in the complete games. On ten later replies with matching physical bounces, both branches contacted eight and landed two. Each began planning about 16.5 frames ahead. The observed median depth deviation from the ready pose was 14 mm for both, while the median height error at the next intercept was 68 mm with blind recentering and 81 mm with optical recentering. The change in recovery activity came chiefly from raw blade feedback. [Matched optical evidence](rotary-comparison.json.gz)

## Separate approaches on later replies

The bounded servo kept blind recovery and applied the existing joint rate and travel constraints to all three planned moves within the predictive horizon of a later strike. It changed the first commanded move on 62 of 88 eligible held frames. It lost one of the two held second returns. In game three it won the same two-return point that blind recentering lost. Across 11 later replies with matching bounces it landed two compared with three for blind recentering. [Solver implementation](../../scripts/experimental/tt15_rotary_player.py) · [Two-return game point](rotary-bounded-game-3.json.gz)

The timed feedback servo also kept blind recovery. During a later approach it used two spatially consistent stereo blade sightings to supplement the modeled pose with a bounded residual. It weighted that correction by the frames left after estimated actuator delay, and applied it on 30 held frames and 16 game frames. It reproduced the two held second returns on the same seeds as blind recentering. On 12 matched later replies both modes contacted nine and landed three. [Timed feedback implementation](../../scripts/experimental/tt15_rotary_player.py) · [Matched comparisons](rotary-comparison.json.gz)

In the third full game, the second incoming bounce on seed 2044287039 was identical for blind recentering, the bounded solver and timed feedback. Their requested commands first diverged at command 77 for the bounded solver and 81 for feedback, after that bounce at frame 77. The bounded solver won with its second return. Feedback landed a third return and lost after the opponent answered. The third incoming ball had different measured physics after the changed second stroke. The changes in rally length and point outcome remained concentrated on that one full-game start. [Blind game](rotary-blind-game-3.json.gz) · [Bounded game](rotary-bounded-game-3.json.gz) · [Timed feedback game](rotary-feedback-game-3.json.gz)

## Checkpoint verification

The optical player after game two reconstructed all 16 point action tapes from RGB and matched four new continuation tapes against uninterrupted play. It preserved all 180 native Pong reference action tapes, 41 earned points and 195/206 later returns. [Optical checkpoint](rotary-optical-game-2-player.json.gz) · [Resume audit](rotary-optical-game-2-resume-audit.json.gz) · [Pong audit](rotary-optical-pong.json.gz)

All 315 recorded points replayed their applied actions and results through the physics. All 177 paired rival-serve trajectories matched, and all 56 paired own-service command prefixes were exactly alike. Every held branch matched the blind control's requested commands through each of the 27 first contacts. The bounded and timed feedback game-three checkpoints replayed all 14 and 13 original RGB action tapes and matched four new continuations apiece. [Bounded checkpoint](rotary-bounded-game-3-player.json.gz) · [Bounded resume audit](rotary-bounded-game-3-resume-audit.json.gz) · [Timed feedback checkpoint](rotary-feedback-game-3-player.json.gz) · [Timed feedback resume audit](rotary-feedback-game-3-resume-audit.json.gz)

## Reproduction

Use the [hall environment](../tt6/README.md) from the repository root.

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TT3_COMPONENT_LIBRARY=/tmp/tt3-components-final.so
python=/workspace/Fly-Connectome/.venv/bin/python
output=/tmp/tt15-rotary
for mode in blind optical bounded feedback; do
  for fixture in 0 1 2 3; do
    "$python" -m scripts.experimental.tt15_rotary_trial "$output" play "$mode" "$fixture"
  done
done
"$python" -m scripts.experimental.tt15_rotary_report "$output" --modes blind optical bounded feedback
"$python" -m scripts.experimental.tt15_rotary_trial "$output" audit optical 2
"$python" -m scripts.experimental.tt15_rotary_trial "$output" audit bounded 3
"$python" -m scripts.experimental.tt15_rotary_trial "$output" audit feedback 3
"$python" -m scripts.experimental.tt8_pong_retention \
  "$output/rotary-optical-game-2-player.json.gz" "$output/rotary-optical-pong.json.gz"
```
