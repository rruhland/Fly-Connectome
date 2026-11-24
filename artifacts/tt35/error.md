# Learning a joint correction from camera-observed misses

Ten matched releases supplied fifty real games in which none of the five compared control settings earned a legal SO-101 return. Each release had been played by the raw RGB-and-encoder pupil with its original command law, or with a small positive or negative shoulder pan or lift offset. The existing recordings retained the physical opponent and near-side bounce. We replayed the commands and measured the ball and blade centers from their orange and red silhouettes in synchronized stereo. [Source training experiment](../tt34/visual.md) · [Optical practice receipt](error-practice.json.gz)

For each usable miss we found two observed frames in which the approaching ball passed the blade's depth. We interpolated between those frames, after play, to measure the signed lateral and vertical error. The timing check rejected a finite difference when either perturbation shifted that passage by more than nine control ticks. Six releases had enough observations to estimate how at least one joint changed the miss. For example, on release 10898 the measured gap was 210 mm with no offset, 192 mm with a +0.12 rad lift offset and 222 mm with −0.12 rad. All three scored zero. [Training contexts, optical errors and fitted joint responses](error-checkpoint.json)

We used damped least squares on the observed action effects to choose a shoulder correction of at most 0.16 rad per joint. On new play the controller mixed those learned corrections using only the current ball position, velocity and encoders. It applied the correction in the same local visual neighborhood as before. The post-play passage time and error were training measurements. Neither an interpolation of future samples nor a target interception plane entered the playing controller. All sixteen held release IDs were disjoint from the optical training releases. The sign-reversed control used the same corrections and magnitudes with opposite signs. [Implementation](../../scripts/experimental/tt35_error.py)

## Held games

All sixteen initial opponent shots matched across controllers through far contact, near bounce, and the measured receiving line. The fixed actor reproduced the original sixteen TT34 games down to its commands and referee events. The learned correction preserved its scores on all points. Reversing the correction lost one legal return on 7906. [Game scores and delivery checks](error-summary.json) · [Fixed games](error-fixed.json.gz) · [Learned games](error-learned.json.gz) · [Sign-reversed games](error-reversed.json.gz)

| Held cohort | Fixed arm returns | Learned arm returns | Sign-reversed arm returns |
| --- | ---: | ---: | ---: |
| Earlier fresh cohort, eight points | 2 | 2 | 2 |
| New-release cohort, eight points | 4 | 4 | 3 |

On the four earlier fresh points with an optical passage measurement in all three conditions, the mean gap was 175.2 mm fixed, 170.4 mm learned, and 177.2 mm sign-reversed. On the three common measurements in the new-release cohort it was 236.4, 236.2, and 239.4 mm. These paired gaps quantify visible misses. The return scores include all sixteen games. [Paired optical passages](error-summary.json)

An evaluator-only native-pose replay found the same direction of change. On seven common physical passages, the mean center-to-center gap was 195.1 mm fixed, 192.1 mm learned and 196.7 mm reversed. Most of the reduction came from release 13131, where the native gap went from 290.2 to 270.9 mm. The camera's median absolute lateral and vertical errors were 1.9 and 2.8 mm on the fixed games with both silhouettes visible. Frames with incomplete visibility did not supply a gap label. [Native geometry and camera audit](error-optical-audit.json)

The learned 7906 return replayed with every referee event intact. Full CAD replay checked 1,575 states with zero penetrations and a minimum clearance of 18.6 mm. The learned model reduced measured misses and preserved the original legal returns on this deck. It added zero legal returns. [Physical return audit](error-7906-audit.json)

## Reproduction

Run from the repository root with the pinned measured feed and the TT34 exploration and held archives. [Regulation dependencies](../tt29/endurance.md)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt35_error /tmp/tt25-rallies.json practice
python -m scripts.experimental.tt35_error /tmp/tt25-rallies.json fit
for mode in fixed learned reversed; do
  python -m scripts.experimental.tt35_error /tmp/tt25-rallies.json "$mode"
done
python -m scripts.experimental.tt35_error /tmp/tt25-rallies.json summary
python -m scripts.experimental.tt35_error /tmp/tt25-rallies.json audit
```
