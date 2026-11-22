# Anonymous stereo and Pong-selected lateral contact against a changing far arm

Four fresh paired starts used native unmarked stereo on the clamped SO-101. The shared opening moved the near arm through its ordinary neutral ramp. The first eight stereo frames initialized a compact anonymous motion track, and the tracker learned that region's appearance from those images. The bootstrap identifies its tracked region by coherent motion. The near arm's reference position came from factory forward kinematics driven by its own commands in an independent arm model. [Anonymous tracker](../../scripts/experimental/tt22_motion_object.py) · [Trial](../../scripts/experimental/tt22_spatial_anonymous.py)

The far arm used its extra physical wrist axis and an existing left, center and right target cycle. Each lane received the same depth, height and pace shift that a previous optical-outcome experiment learned for the far arm's static target. The lane ordering was supplied. The opening generator reduced native spread to 0.55 and pace to 0.90, following that earlier qualification. Every return still required physical racket contact and a legal table bounce. The receipt includes actual near and far rate tapes, the far arm's requested lanes, and where legal replies reached the near strike window. [Learned far profile](tt22_opponent-learned.json.gz) · [Paired physical tapes](spatial-anonymous-varied.json.gz)

The near controller used the same fixed flight law, known arm geometry and limits, analytic yaw, stroke catalogue search, and centrally weighted legal-landing cost in all conditions. The only policy intervention was the lateral contact coordinate and how its two portable gains were chosen. The direct rule sends its proposed coordinate to the existing inverse-kinematics and motor bounds without an additional displacement cap. The source's signed 2D bias is unused in the 3D camera frame. [Spatial law](../../scripts/experimental/tt22_spatial.py) · [Native source qualification](../tt3/discovered-source-refinement.json.gz)

The comparisons include the fixed native Pong winner, the same scored finalist family learning from official target outcomes, the finalist family with its native scores permuted, a source-naive signed search, and fixed unit geometry. The three learning arms persist separate outcome counts over the same four starts. Ties and one shared exploration choice use a fixed independent random seed. An exact same-coefficient condition on the same start shares its physical tape, with separate score accounting and state updates. [Choices and event traces](spatial-anonymous-varied.json.gz)

## Matched four-start comparison

| Preference | First returns | Second returns | Total legal returns | Point wins | Longest alternating rally |
|---|---:|---:|---:|---:|---:|
| Fixed Pong score winner | 4/4 | 3/4 | 7 | 2/4 | 4 |
| Pong finalist family | 4/4 | 3/4 | 10 | 0/4 | 8 |
| Signed source-naive | 4/4 | 3/4 | 9 | 0/4 | 8 |
| Score-permuted family | 4/4 | 2/4 | 9 | 1/4 | 8 |
| Fixed unit geometry | 4/4 | 2/4 | 7 | 1/4 | 6 |
| Fixed Pong duration winner | 4/4 | 4/4 | 8 | 1/4 | 4 |

The eight-return point used the same randomized signed exploration choice in all three adaptive branches, including scratch. The native-finalist family made ten legal returns in total and won no points. The fixed duration winner landed a second legal return on all four starts and shared the exact first legal serve event with the other five branches. [Paired source-score branches](spatial-anonymous-varied.json.gz) · [Fixed duration branch](spatial-anonymous-duration.json.gz)

The rival changed physical arrival lanes within a single point. On seed 1957082491 the fixed-geometry player received three replies at lateral x = +4.8, −3.9 and −16.9 cm in the strike window. The anonymous tracker bootstrapped on all four starts. It located the mover within 15 mm on 461 of 501 eligible frames in the adaptive finalist family and 345 of 383 in the duration branch. All 17 distinct two-arm score-test tapes and four duration tapes replayed their strict events, scores and durations. The longest witness from each group also reproduced its original native RGB sequence. [Two-arm comparisons](spatial-anonymous-varied.json.gz) · [Duration tapes](spatial-anonymous-duration.json.gz)

## Common-prefix late-contact diagnostic

The eight-return point provided a real losing continuation for a controlled replay. We held both arm command tapes and optical observations identical through the far strike at tick 156. The same legal near-side bounce at tick 171 occurred on each fork. The original signed choice and unit geometry both produced a near-arm contact followed by an own-half fault. Changing only the lateral preference to the duration winner did not produce a contact and the point ended on the floor. All three finishes still contained eight legal alternating returns. [Shared early trajectory and real late forks](spatial-anonymous-late.json.gz)

## Fresh six-start duration comparison

| Fixed preference | First returns | Second returns | Total legal returns | Point wins | Longest alternating rally |
|---|---:|---:|---:|---:|---:|
| Pong duration winner | 6/6 | 2/6 | 8 | 1/6 | 4 |
| Pong score winner | 5/6 | 2/6 | 8 | 1/6 | 6 |
| Unit geometry | 5/6 | 2/6 | 7 | 0/6 | 4 |

All openings were legal. The duration winner gained one paired first return against each comparator. Its four-of-four second returns from the initial comparison became two of six, the same as both controls. The duration branch physically contacted later incoming replies from −15.3 to +9.5 cm laterally, and the anonymous tracker was within 15 mm on 430 of 474 eligible frames. The 18 complete two-arm command tapes reproduced their scores and all events. The longest held witness also matched its original RGB digest. [Fresh paired holds](spatial-anonymous-duration-held.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1
python -m scripts.experimental.tt22_spatial_anonymous
python -m scripts.experimental.tt22_spatial_anonymous --verify
python -m scripts.experimental.tt22_spatial_anonymous --duration
python -m scripts.experimental.tt22_spatial_anonymous --late-contact
python -m scripts.experimental.tt22_spatial_anonymous --duration-held
python -m scripts.experimental.tt22_spatial_anonymous --duration-held --verify
```
