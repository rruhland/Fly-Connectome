# Reward-selected optical motion gain before fixed-launcher bounce

The learned gate reduced drive while the launcher held the ball and brought mean gain back near one during free play. It produced shorter stock streaks and tied the unchanged carried control on the alternate body. A small immediate-release control favored the gate. [Held trajectories and gains](tt24_motion_ready.json.gz)

The fixed-launcher benchmark holds a visible ball for a random 0.23–0.63 seconds while the wrist starts with independently varied joint angles. The established carried policy acts from the first camera frame, using the same familiar donor equilibrium in its sensory and motor coordinates. This experiment learns how much of that policy's visual joint target to use while observed ball motion is weak. [Established benchmark and donor posture origin](ready.md)

Each camera frame supplies the existing observer's accepted ball position in a frame stabilized by the arm's own encoders. A causal filter uses the last three accepted positions. It multiplies travel speed by the ratio of net displacement to the observed path length, so turning-back noise contributes less. A leaky maximum retains recent evidence across short slow periods and missing sightings. The gate smoothly increases from a selected floor to unit visual gain as this evidence reaches a selected speed scale. The fixed servo still drives toward the familiar joint equilibrium and retains its original limits. Unit gain executes the exact original visual motor rule. The gate receives the observation stream alone. Its input has no launcher flag, contact label, release countdown or prescribed joint trajectory. [Modulation and unchanged benchmark integration](../../scripts/experimental/tt24_motion_ready.py)

The source screen compares the four combinations of speed scale 0.14 or 0.34 m/s and visual floor 0 or 0.28. Three physically different donor bodies each supply two fixed-launcher wide starts to every candidate. Source perception uses actual wrist RGB and encoder-derived camera and blade transforms from supplied donor CAD, including the nominal paddle weld. The matching reference conditions use the unmodified carried gain of 1.0 and a constant 0.65. Only original physical rebounds select the learned setting. The objective weights the mean fraction reaching eight, the worst body fraction and the mean clipped streak. The source screen uses a 3.5-second cap after release. [Locked roster and objective](tt24_motion_ready-plan.json) · [Source rewards and optical gains](tt24_motion_ready-source.json.gz)

The selected setting is frozen before target evaluation. Each of the three conditions then plays eight stock and eight changed-body wide releases with the frozen public stock observer. It also plays four independent immediate moving-ball stock releases. These runs contain no target motor practice. Every held and donor trajectory, including the supported-ball preparation, must reproduce the original physical judge record on action replay. [Frozen source selection](tt24_motion_ready-lock.json) · [Complete held data](tt24_motion_ready.json.gz)

## Learned setting and held results

The donor fit selected a 0.34 m/s motion scale and a floor gain of 0.28. Its source reliability score was 0.417, compared with 0.454 for the unchanged carried controller and 0.321 for the fixed 0.65 gain. The fit selects among motion-dependent candidates using the source score. The two constant settings are reference conditions.

| Held course | Learned mean longest | Carried mean longest | Fixed 0.65 mean longest | Runs reaching ten, learned / carried / fixed |
|---|---:|---:|---:|---:|
| Wide stock with held ball, 8 starts | 6.75 | 8.63 | 4.50 | 3 / 4 / 1 |
| Wide alternate with held ball, 8 starts | 1.75 | 1.75 | 3.50 | 0 / 0 / 1 |
| Immediate-release wide stock, 4 starts | 7.50 | 5.25 | 5.25 | 2 / 1 / 1 |

The selected gate averaged 0.427 before release and 0.975 afterward on stock. On the alternate it averaged 0.568 before release and 0.978 afterward. The unmodified stock controller had a 15-rebound run on the start where the gate scored two, accounting for 13 of the 15 lost consecutive rebounds. The immediate-release gain was similarly concentrated. One start went from two to 13, and two other starts each lost one rebound. The registered held-ball courses give no improvement in long streak reliability from this modulation.

The study used 36 scored donor episodes and 60 held evaluations over 26 disjoint release seeds. All 96 action tapes reproduced their physical outcomes. An independent comparison reproduced all 64 original motor responses exactly at unit gain. [Source and held action evidence](tt24_motion_ready-source.json.gz) · [Held phase gains](tt24_motion_ready.json.gz)

## Reproduction

Use the repository's table-tennis Python environment, from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_motion_ready
```
