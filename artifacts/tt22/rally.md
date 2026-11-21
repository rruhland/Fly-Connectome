# Sustained physical SO-101 exchanges

The pupil moved the stock clamped SO-101 and its physical paddle through the unmarked compact court. Gentle opening releases used the standard legal-service trajectory with 82% of the original launch speed and 30% of the original random spread. The physical rival then used the selected return policy. The referee counted consecutive alternating legal returns after the serve. [Experiment](../../scripts/experimental/tt22_rally_trial.py)

The pupil registered the two fixed cameras from visible table lines and reconstructed ball flight from native images. It prepared an approximate ready position and proposed joint-rate paths at several reachable depths, blade pitches, lead times and forward speeds. Candidate forecasts used a separate factory model initialized from the pupil’s prior commands, with the latest optical ball estimate and bounded uncertainty. The physical opponent’s exact-state search stayed separate. [Pupil](../../scripts/experimental/tt22_rally.py) · [Anchor opponent](../../scripts/experimental/tt22_opponent.py)

On six development points, this native receiver and stock-anchor rival achieved 8, 8, 6, 6, 2 and 14 alternating legal returns. The same learner with the earlier coach reached 1, 2, 3 and 3 on the first four matched starts. The 14-return point contained seven from each arm. It lasted 303 control frames and ended after the rival sent the ball toward x = 0.384 m at the pupil’s receiving depth, outside the pupil’s planned workspace. All six stock-anchor command sequences reproduced the original referee events. [Anchor development](rally-anchor-612423769.json.gz) · [Earlier coach](rally-coach-612423769.json.gz) · [14-return replay](rally-14.mp4)

The learning comparison retained the same arm, target generator, factory model and opponent. The online policy inferred the net crossing and first court-height bounce from its own observed outward flight. A bounded running mean of prediction errors corrected subsequent landing and clearance estimates, changing which blade pose and velocity won the physical search. The learner updated on four practice points and as later outcomes became visible. Each new assessment seed was shared with the original frozen planner, alternating evaluation order. A third policy used the learned policy’s edge and net penalties with a fixed correction of zero. This controlled for the added scoring penalties on the same six points. The practice checkpoint contains seven observed strokes. [Learning points](rally-learning-anchor.json.gz) · [Frozen checkpoint](rally-learning-checkpoint.json.gz) · [Zero-correction control](rally-anchor-1505337903-zero-guard.json.gz)

## Matched live learning

| Pupil policy | Consecutive legal returns on each of six new starts | Own legal returns | Both sides’ total | Points with at least ten |
| --- | --- | ---: | ---: | ---: |
| Original frozen planner | 0, 0, 2, 10, 2, 4 | 9 | 18 | 1/6 |
| Same edge and net penalties with zero correction | 0, 2, 2, 10, 2, 4 | 10 | 20 | 1/6 |
| Online optical correction | 4, 4, 4, 2, 0, 10 | 12 | 24 | 1/6 |

Each pupil policy lost all six points. Against the zero-correction control, the online policy lengthened four points and shortened two. It extended seed `1030035593` from four to ten legal returns and shortened seed `2043668154` from ten to two. These are separate fresh physical rallies with five legal returns by each arm at the ten-return mark. With the 14-return development point, three distinct starts reached at least ten with the stock anchor. [Learning points](rally-learning-anchor.json.gz) · [Zero-correction control](rally-anchor-1505337903-zero-guard.json.gz)

After four practice points, the optical corrections were −7.5 mm lateral, −93.3 mm downcourt and −31.7 mm at the net. The learner finished the six assessed points with 18 optical observations. On the first assessed point, the learned correction selected a 0.92 m contact depth with higher forward paddle speed. Both controls selected 0.815 m and landed on the pupil’s side. The learned stroke landed on the rival’s side, starting a four-return exchange. All 22 recorded practice and scored branches reproduced exact referee results and physical events on replay. [Learning points](rally-learning-anchor.json.gz)

## Post-bounce replanning check

On two development starts, the receiver checked the remaining shot in three simulations after a new optical sighting of the rising ball. It replaced the prepared stroke only when fewer than two simulations predicted a legal return and a new proposal predicted one. The original rally lengths were eight and eight. The revised policy reached eight and two. Both replays reproduced the original scored events. [Replanning receipt](rally-anchor-612423769-refined.json.gz)

## Reproduction

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_rally_trial --rival anchor --count 6
python -m scripts.experimental.tt22_rally_trial --rival anchor --learn --practice 4 --count 6
python -m scripts.experimental.tt22_rally_trial --rival anchor --seed 1505337903 --count 6 --zero-guard
```
