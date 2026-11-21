# Removing optical scaffolding on the clamped SO-101

The pilot compares 8 marked and 8 unmarked practice points with 16 unmarked practice points from the start. A frozen factory controller provides a zero-practice reference. All held play uses unmarked native wrist and fixed RGB on the 1.600 × 0.848 m mini court under the strict scorer. The rival and prechosen launch sequence are common. The learner sees RGB, known lens optics and court dimensions, public service role, and point reward during practice. Simulator truth only measures errors and verifies replays.

The only scene intervention is the visible hall survey posts. Ball and paddle material, CAD, clamp, wrist optics, court, illumination, and gameplay stay fixed. The 320 × 240 tracker shares the ball and blade code across conditions. It uses the surveyed geometry when enough posts are visible and the ordinary court lines elsewhere. The wrist lens is 84° in both. The sparse opening cadence is every other control tick. Receiving uses RGB on every 30 Hz tick. The optical placement merit works from the observed ball path and the final point reward.

The same four serve candidates and 2,048 return tapes come from an isolated model of the publicly declared clamp and factory arm. The four candidates each served legally on 6/8 independent simulated tosses in the shortlist evaluation. Their order is fixed before target interaction. The two live practice schedules match across 16 distinct tosses and four balanced rounds. Practice ranks serve candidates, with the receiver idle, using a small common tie-break toward the top-ranked simulation candidate. The marked-first arm initializes the natural tracker afresh after removal. Both trained arms start the held game with fresh camera fits. This isolates transfer through learned serve selection. A separate identical-tape replay evaluates retaining camera fits. The shared match receiver retains older gantry flight and impact parameters. The nominal-world reference changes those two parameters while retaining the factory arm model and receiver algorithm.

## Practice

| Training | Target practice points | Control frames | Legal serves, first 8 | Legal serves, last 8 | Points won | Candidate after 4 / 8 / 12 / 16 attempts |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Visible posts, then removed | 16 | 946 | 7 | 4 | 3 | 0 / 0 / 2 / 0 |
| Unmarked from the start | 16 | 931 | 7 | 4 | 3 | 0 / 0 / 2 / 0 |
| Frozen factory | 0 | 0 | — | — | — | 0 throughout |

The posts reduced median ball error in the first eight practice points from 9.4 mm to 5.1 mm. Median error four control ticks ahead dropped from 18.7 mm to 14.2 mm under the common nominal flight extrapolation. Meanwhile the marked color scene yielded zero valid 3D blade estimates on 404 in-court frames, versus 391 of 391 without posts. In a paired resting start, each fixed camera's red detection landed nearer the red survey post with it present. Removing the post moved the detection within 0.36 pixel of the projected blade in both fixed cameras. The gameplay policy used the ball track, so the lost blade channel did not reduce the early legal-serve count.

The first and second halves contain different tosses, so their serve counts measure different sets of starts. Reported practice control frames reflect actual point durations and the same fixed number of starts in each treatment. This tests a short candidate-selection curriculum with a shared world model on one simulated body.

[Marked-first practice receipt](curriculum-fade-practice.json.gz) · [Unmarked-only practice receipt](curriculum-natural-practice.json.gz) · [Factory design receipt](curriculum-factory.json.gz)

## Held unmarked games

| Condition | SO-101–rival | Learner legal serves | Learner legal returns | Points with multiple learner returns | Exact referee replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visible posts during first half of practice | 8–11 | 4/9 | 11 | 2 | 19/19 |
| Unmarked throughout practice | 8–11 | 4/9 | 11 | 2 | 19/19 |
| Frozen factory, carried flight and impact | 8–11 | 4/9 | 11 | 2 | 19/19 |
| Frozen factory, nominal flight and impact | 6–11 | 4/8 | 9 | 0 | 17/17 |

Both training paths selected the original factory serve. The first three separately played games reproduced the same 1,098 actions, RGB hashes, and referee events. The natural tracker estimated the ball on 910 of 922 in-court captures at 8.9 mm median error and the blade on 922 of 922 at 4.6 mm. Dropping the carried flight and impact together cost two points and two legal returns on the same launch sequence. This single game per distinct policy supports a policy effect from those shared priors and provides no evidence of a lasting marked-first advantage.

[Marked-first held game](curriculum-fade-2082473549.json.gz) · [Unmarked-only held game](curriculum-natural-2082473549.json.gz) · [Frozen held game](curriculum-frozen-2082473549.json.gz) · [Nominal-world held game](curriculum-frozen-2082473549-nominal.json.gz)

The longest rally was point 9, with two legal returns per side. The multi-camera clip shows the three actual RGB inputs and a spectator view as the score moves from 3–5 to 3–6. All 101 native capture ticks reproduced the original RGB hash. The 4.77-second MP4 holds the result after the 3.37-second point. [Rally clip](so101-longest-rally.mp4) · [Replay receipt and events](so101-longest-rally-replay.json)

## Same-action camera replay

The first four held tapes span both service roles and 166 native RGB captures. The three replay conditions reproduced the same frame-by-frame ball trajectory and referee events. The unmarked replay also reproduced all four original RGB hashes. The carried condition starts with both fixed poses fitted during marked practice. The moving wrist starts fresh.

| Camera condition | Ball tracks | Median ball error | Ball error p90 | Blade tracks | Median blade error | Median four-tick forecast error | Forecast error p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Posts in view | 154/155 | 4.7 mm | 16.1 mm | 0/155 | — | 14.5 mm | 92.8 mm |
| Unmarked, fresh fits | 154/155 | 9.7 mm | 26.0 mm | 155/155 | 4.5 mm | 17.1 mm | 76.5 mm |
| Unmarked, carried fixed fits | 154/155 | 9.7 mm | 26.0 mm | 155/155 | 4.5 mm | 17.1 mm | 76.5 mm |

The posts supplied both fixed fits on all 166 frames, and supplied the moving wrist fit on 63 frames. Registration from ordinary lines gave 116 valid wrist fits without posts and 69 additional fits in marked frames with insufficient visible survey points. The survey improved median ball localization on these tapes while four-tick forecasts had a larger upper tail. Once the posts were removed, the fixed cameras recovered the same poses and tracks with or without carried state, including a ball estimate at the first capture on each point.

[Paired perception replay receipt](curriculum-paired-perception.json.gz)

## Reproduction

Run from the repository root with native MuJoCo EGL and the repository's Python environment. Use a fresh `--folder` for a new course. The last command takes explicit paths when using a separate output folder.

```bash
python -m scripts.experimental.tt20_curriculum fade --operation model
python -m scripts.experimental.tt20_curriculum fade
python -m scripts.experimental.tt20_curriculum natural
python -m scripts.experimental.tt20_curriculum frozen --operation game
python -m scripts.experimental.tt20_curriculum frozen --operation game --nominal-world
python -m scripts.experimental.tt20_curriculum_replay
```

[Training and scoring code](../../scripts/experimental/tt20_curriculum.py) · [Paired replay code](../../scripts/experimental/tt20_curriculum_replay.py)
