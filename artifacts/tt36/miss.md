# Outgoing camera feedback from misses

This test used the TT33 raw joint policy unchanged as its starting controller. We replayed all nine earlier matched release-and-seed triples for which the original joint search had recorded one legal incoming offer but zero legal arm returns under every tested setting. Those 27 games were played by the same raw pupil with three known joint target offsets. The fitting inputs included none of the held releases or four planned continuation releases. [Frozen raw pupil](../tt33/joint-raw.json.gz) · [Physical exploration source](../tt35/search-practice.json.gz) · [All zero-score optical labels](miss-practice.json.gz)

The evaluator reconstructed the ball and blade centers from orange and red stereo silhouettes. It measured their nearest visible three-dimensional separation around the first approach. It also looked for sustained visible movement back toward the opponent after that encounter, and gave more credit to a ball seen to travel farther forward within the court width. Every geometric feature came from recorded images. A contact, long shot or net miss could change the optical grade even when the legal score stayed zero. The label used no predicted interception or projected landing. [Camera measurement and fit](../../scripts/experimental/tt36_miss.py)

Of the 27 trials, 26 supplied a visible ball-to-blade gap and eight also supplied an outgoing arc. On release 10313, one perturbed attempt produced no visible forward arc while the other sent the ball to within 3.3 cm of the net. Their legal scores were both zero. [Practice observations](miss-practice.json.gz)

We used the difference in optical grades between the two opposite joint perturbations to obtain a signed corrective direction per release. During play the controller mixed these directions by similarity to the current optically estimated ball position, velocity and measured joint angles. It added the resulting bounded angle offset to the existing joint target and preserved the existing motor limits. Only the observation at the beginning of each visible approach enters this correction. The checkpoint stores the nine comparison observations and their signed directions. The sign control multiplies the same correction by −1. [Composable checkpoint](miss-checkpoint.json)

## Held games

| Release-disjoint cohort | Frozen | Learned | Sign reversed |
| --- | ---: | ---: | ---: |
| Eight ready-pose fresh points | 3 | 2 | 4 |
| Sixteen novel points | 6 | 5 | 5 |
| All 24 points | 9 | 7 | 9 |

Scores count legal SO-101 returns. All first deliveries matched across the three controllers at far contact, near bounce and the receiving line. The frozen policy also reproduced all 24 earlier raw-control games exactly. The learned direction lost the first return on 14993 and the second return on 3025. The reverse sign regained two on 8239, lost 9532 and made compensating changes on the novel cohort. The camera grade distinguished scoreless attempts, but the direction learned from these labels failed to improve the raw pupil's legal returns on either held cohort. [Paired scores, CPU time and six tapes](miss-summary.json)

Replay of all six changed-score games confirms that a smaller gap could accompany a worse shot. On 14993, the learned change narrowed the nearest visible gap from 173 to 163 mm, yet the observed outward flight stopped 20 mm short of the net. The frozen control landed its return and was visible 1.069 m onto the far half. Full CAD replay of the three versions of all six points reproduced their referee events with zero penetrations over 23,730 solver states. Minimum clearance was 17.8 mm. [Optical and physical replay evidence](miss-audit.json)

## Reproduction

Use the pinned measured feeds and regulation dependencies. [Base pupil reproduction](../tt33/joint.md)

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_miss /tmp/tt25-rallies.json collect
python -m scripts.experimental.tt36_miss /tmp/tt25-rallies.json fit
for mode in fixed learned reversed; do
  python -m scripts.experimental.tt36_miss /tmp/tt25-rallies.json "$mode"
done
python -m scripts.experimental.tt36_miss /tmp/tt25-rallies.json summary
python -m scripts.experimental.tt36_miss /tmp/tt25-rallies.json audit
```
