# Learning actuator delay during ordinary points

The strict moved-clamp SO-101 played complete games against the standard physical opponent using either one outside camera or one moving wrist camera. Each controller received only the selected 480 × 360 RGB stream. The evaluator inserted a four-frame command FIFO after four ordinary points, delaying the actual six motor commands by 133 ms. Both versions began at zero lead, used the same stock serve and receive policies, and drew point starts from matched game seeds. The delay estimate carried forward across points. Each game began with a fresh estimate.

The learner stored its issued commands and a scalar RGB motion trace during each point. For the outside eye, the trace measures pixel changes in the red blade silhouette. For the wrist eye, it measures motion in three image regions above or alongside the held blade. At point end, a separate nominal MuJoCo arm replayed the issued commands with the ball parked and zero transport delay. The reference uses the declared nominal clamp, lens mount and hall and renders the range around the issued strokes. A discrete alignment over 0–6 frames compared nominal and observed motion. Recent losses carried forward, and the estimate updated only when visible motion and the alignment margin met fixed thresholds. The learner received no change cue, score, applied motor values or live simulator pose. [Learner](../../scripts/experimental/tt21_game_delay.py)

The adaptive controller used the accepted lead in the existing opening serve clock and in predictions of the incoming ball at the time a physical command would begin. The frozen controller held the lead at zero. Recognition used regular points and charged no extra real control frames. Nominal rendering and fitting time are recorded separately. [Trial](../../scripts/experimental/tt21_game_delay_trial.py)

## Recognition and play

All four learners held zero through the four nominal points and accepted four frames at the end of the first delayed point, after 34–44 ordinary frames (1.13–1.47 simulated seconds). The frozen and adaptive games shared identical actions, RGB digests and scoring events through that point. The learned estimates then stayed at four.

Post-change results below count starts with the same ball seed and service role that both game branches reached. The full scores include the additional points each branch played before the game ended.

| Eye and game seed | First estimate, frames after change | Full score frozen → adaptive | Matched point wins | Matched first returns on receive | Matched legal own serves |
|---|---:|---:|---:|---:|---:|
| Outside, 1920024739 | 44 | 3–11 → 5–11 | 0/10 → 2/10 | 0/6 → 1/6 | 0/4 → 2/4 |
| Wrist, 1920024739 | 34 | 1–11 → 5–11 | 0/8 → 3/8 | 0/4 → 1/4 | 0/4 → 4/4 |
| Outside, 384469573 | 41 | 0–11 → 0–11 | 0/7 → 0/7 | 0/4 → 1/4 | 0/3 → 3/3 |
| Wrist, 384469573 | 41 | 1–11 → 4–11 | 0/8 → 2/8 | 0/4 → 3/4 | 0/4 → 4/4 |
| Total | | | 0/33 → 7/33 | 0/18 → 6/18 | 0/15 → 13/15 |

The opponent won all eight games, and no point had multiple legal learner returns. The adaptive games took 310–602 nominal renders and 12.1–20.8 seconds of cumulative inference wall time per game. All 107 native points replayed exactly from the issued commands. [Four-frame trial receipt](game-delay.json.gz)

The wrist video shows the first receiving point played after recognition on seed 1920024739. The two controllers see the same launch and issue their commands into the same four-frame delay. The adaptive return lands legally and wins the point. [Wrist comparison](game-delay-wrist.mp4) · [Video replay receipt](game-delay-wrist.json)

## Outside-camera switchback

The additional match on seed 1692022069 kept the observer and thresholds fixed. The evaluator changed the FIFO from zero to two frames after point four, then restored zero after point eight. The lead estimate carried through both changes. The game and point seeds were disjoint from the four-frame schedules and the earlier archived runs.

| Shared points | Evaluator delay | First correct estimate after switch | Matched point wins frozen → adaptive | Matched first returns | Matched legal own serves |
|---|---:|---:|---:|---:|---:|
| 1–4 | 0 frames | Started at zero | 0/4 → 0/4 | 0/2 → 0/2 | 1/2 → 1/2 |
| 5–8 | 2 frames | End of point 5, 40 frames / 1.33 s | 1/4 → 0/4 | 1/2 → 1/2 | 0/2 → 1/2 |
| 9–11 | 0 frames | End of point 9, 35 frames / 1.17 s | 0/3 → 0/3 | 0/2 → 0/2 | 1/1 → 1/1 |

The estimate was correct at the end of all eleven adaptive points. The frozen game finished 1–11 and the adaptive game finished 0–11. The learner used 315 nominal renders, 14.2 seconds of cumulative inference wall time and zero dedicated target frames. All 23 points replayed exactly. [Switchback receipt](game-delay-switchback.json.gz)

## Reproduction

With the repository’s MuJoCo environment active, run from the repository root.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt21_game_delay_trial
```

The receipt saves actions, applied motor commands, chronological RGB ball estimates, both motion traces, shift losses, accepted estimates, point events and score. `--resume` preserves complete matched game pairs. `--verify` checks physical transport, scoring and events from the saved requested commands. [Machine receipt](game-delay.json.gz)

Add `--switchback` to run or verify the separate outside-camera switchback schedule.
