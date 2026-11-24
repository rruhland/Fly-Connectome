# Wrist-camera warm-up on the table-edge clamp

Twenty-four practice drops increased qualified rebounds from 6 to 25 on eight paired held drops. The longest held streak increased from 1 to 8, and the median was 1 at both checkpoints. Five paired streaks increased and three tied. The held deck varied release position, velocity, spin and servo behavior. The learned policy ran from wrist RGB and its own encoders at 60 Hz. [Solo course and command streams](tt29_bounce_course.json.gz) · [Held video](tt29_bounce_held.mp4)

| Completed practice drops | Qualified rebounds across 8 held releases | Longest streak | Held releases with at least 5 rebounds |
| ---: | ---: | ---: | ---: |
| 0 | 6 | 1 | 0 |
| 4 | 33 | 13 | 2 |
| 12 | 42 | 15 | 3 |
| 24 | 25 | 8 | 3 |

The carried motor weights stayed frozen. Practice estimated the new base registration and blade pose from stereo silhouettes, checked paddle attitude against the first camera-visible ball reversal in 23 of 24 drops, and fit the new wrist projection from 588 simultaneous wrist and stereo sightings. A separate tolerance learner consumed scalar rebound rewards as each drop ended. The final fit had a 0.62 px median weighted reprojection error at the 192 × 144 training resolution. The solo course used 2,377 control intervals, about 40 simulated seconds. The 12-drop checkpoint scored higher than the 24-drop checkpoint on these held releases. [Training and replay code](../../scripts/experimental/tt29_bounce.py) · [24-drop checkpoint](tt29_bounce_checkpoint.json)

## Checkpoint promotion

The controller saved an immutable incumbent and compared each newer candidate against it on six fresh validation drops with a three-second cap. Promotion required more paired streak wins than losses, at least two additional qualified rebounds, and a median streak at least as high as the incumbent's. The three validation groups were mutually disjoint and excluded the chronological course and its original held deck. A real robot would have to play both sets of six drops at each gate. The simulation matches release seeds and verifies the resulting serial command streams. [Promotion implementation](../../scripts/experimental/tt29_bounce_promotion.py) · [Validation traces](tt29_bounce_promotion.json.gz)

| Challenger | Paired streak wins / losses | Change in qualified rebounds | Decision |
| --- | ---: | ---: | --- |
| 4 drops | 3 / 0 | +10 | Promoted |
| 12 drops | 1 / 3 | −8 | Retained 4-drop checkpoint |
| 24 drops | 1 / 1 | 0 | Retained 4-drop checkpoint |

On another eight releases withheld from the gate, the retained four-drop controller made 10 qualified rebounds and reached a five-rebound streak. The cold policy made four qualified rebounds and reached a one-rebound streak. Three paired streaks increased and five tied. Finite validation provides evidence for each promotion. Performance can still vary on future drops. [Retained deployment checkpoint](tt29_bounce_retained.json)

## Conditional regulation receiving

A coach first demonstrated a single legal physical return of measured opening 4230. Each camera condition received the same 32 practice perturbations and 24 separately seeded held perturbations of that opening. Each policy used the first six synchronized wrist exposures and encoders to authorize the recorded motor pattern. The bounce-calibrated learner also selected among five small wrist-roll rate adjustments using only the legal outcomes of its actual practice returns. The cold condition carried the earlier wrist projection and the same pattern, lens intrinsics and camera acceptance rule. This assay measures warm-up transfer to one opening family. [Coach tape and launch](tt29_bounce_receive_demo.json.gz) · [Paired table traces](tt29_bounce_receive.json.gz)

| Receiver | First legal practice return | Legal practice returns | Legal held returns |
| --- | ---: | ---: | ---: |
| Archived mount calibration | No return in 32 trials | 0 / 32 | 0 / 24 |
| Bounce calibration | 4 | 21 / 32 | 15 / 24 |
| Bounce calibration and table reward updates | 5 | 21 / 32 | 20 / 24 |

The cold arm physically contacted the ball in all 56 trials while holding ready. Its archived camera transform pointed the reconstructed bearings away from the court, so the gate never selected the stroke. The warmed branches accepted all 56 launches. The updated wrist adjustment was selected after 32 trials and made 20 legal held returns. Its median incoming speed at successful blade contact was 4.06 m/s. The table course replayed all 168 command streams against the physical referee. After the table updates, the verified warm-up checkpoint hash was identical and the eight repeated held drops reproduced all original commands and referee outcomes. [Receiving and retention code](../../scripts/experimental/tt29_bounce_receive.py)

Running the same receiving assay from the retained four-drop checkpoint reproduced all 168 table action streams and referee outcomes. It also reproduced all eight of its independent held bounce trajectories after table updates. [Retained checkpoint transfer and retention log](tt29_bounce_retained_receive.json.gz)

An independent full-CAD shadow checked the chronological course, its held drops, the post-table held drops and the first table run. It found zero checked penetrations beyond 3 mm across 17,068 sampled 60 Hz control states from 233 command streams including the coach. [Clearance and referee replay log](tt29_bounce_clearance.json.gz)

```bash
export MUJOCO_GL=egl PYTHONPATH=src OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt29_bounce
python -m scripts.experimental.tt29_bounce_promotion
python -m scripts.experimental.tt29_bounce_receive
python -m scripts.experimental.tt29_bounce_audit
python -m scripts.experimental.tt29_bounce_movie
python -m scripts.experimental.tt29_bounce_receive --output /tmp/tt29-retained \
  --checkpoint artifacts/tt29/tt29_bounce_retained.json \
  --course artifacts/tt29/tt29_bounce_promotion.json.gz
```
