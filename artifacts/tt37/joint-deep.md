# A camera-to-joint pupil for the deeper centered receiving posture

The frozen camera-to-joint pupil landed 10 legal near returns on 24 completely unseen measured releases, including two later returns. With familiar openings and independently redrawn replies, it landed 26 legal returns and reached a best rally of nine alternating legal strokes. No tested pupil sustained longer play. [Complete paired results](joint-deep-results.json)

## What the pupil learned

The mentor collected successful physical strikes from a different, deeper centered home. We kept one strongest positive tape per exact combination of measured launch, opening seed and reply seed, across six completed mentor archives. Twenty-eight contexts from sixteen different measured launches contained 86 legal near returns. Exact action replay verified the complete physics referee stream and re-rendered stereo images around each verified legal strike, giving 4,410 camera/encoder samples. Only observations from those successful windows entered the model. Nothing from the earlier left-ready arm was used in this pupil. [Frozen observation receipt](joint-deep-observations.json)

On each tick, a local linear fit delivers one joint setpoint from current visible ball position and velocity and the measured arm angles. It does not receive a release ID, a target, a contact time, opponent or ball simulator state, or a whole sequence of actions. Incoming and outgoing observations use separate fits based on the measured direction of ball motion. Weak local support moves the arm back toward its ready pose. The pupil uses the same centered starting posture and motor limits throughout. [Controller](../../scripts/experimental/tt37_joint_deep.py)

We compared two label choices on the familiar physical training contexts before running the held tests. Copying the current mentor setpoint gave 28 legal near returns and 12 later returns. Using a two-tick leading setpoint and tighter weighting on current depth gave ten legal near returns and two later returns. The first variant was selected by the prespecified score of legal near returns plus later returns, then frozen and compared with a stationary controller at the exact same centered pose. That stationary arm is a pose control, not an alternative trained method. The pupil did not reproduce most of the mentor's late returns in the calibration contexts. [Practice selection](joint-deep-choice.json)

## Live physical comparison

One held deck had 24 independently drawn releases absent from all six mentor archives, including unsuccessful training attempts. The other retained one familiar opening per trained release and independently redrew two continuation sequences per release. The latter tests whether returns can continue once the pupil receives a new reply after a familiar first shot. All points retained the live opponent, physical table and legal referee. Up to 200 alternating legal returns were permitted on each held point. [Pinned opening deck](opponent-plan.json)

| Physical test | Offered openings | Legal first | Legal later | Best near / alternating |
| --- | ---: | ---: | ---: | ---: |
| Frozen pupil, 24 novel releases | 22 | 8 | 2 | 2 / 5 |
| Stationary centered control, novel | 22 | 0 | 0 | 0 / 1 |
| Frozen pupil, 32 new continuation sequences | 32 | 20 | 6 | 4 / 9 |
| Stationary centered control, continuations | 32 | 0 | 0 | 0 / 1 |

All 54 delivered opening trajectories matched exactly between the stationary and learned arms, as did the two novel releases that never delivered a legal opening. The pupil made physical opening contact in 16 of 22 offered novel shots, then landed three of nine left, two of six middle, and three of seven right. It landed two of the ten later offers in that deck. In the redrawn continuation deck it landed six of 25 later offers and four separate points reached a second own return. The similar left-ready reach experiment scored seven opening returns and no later returns on the same nominal novel cards, but its different body pose led to more points without an offer. That comparison does not isolate either posture or learning. [Complete paired results](joint-deep-results.json)

Exact action replay of the highest-scoring familiar, novel, and redrawn-reply points reproduced the scores and physical deliveries. Their ten legal near contacts all used the negative paddle face. The audits found zero near-arm penetrations in 10,185 physics samples, with at least 15.3 mm clearance. This checks those three points, not all swipes. [Action-only CAD and blade audit](joint-deep-audit.json)

## Reproduce

Use the pinned measured launches from the [regulation reproduction instructions](../tt29/endurance.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_joint_deep capture artifacts/tt37/clinic-{deep,search-deep,search-deep2,guided-0,guided-10,guided-20}.json.gz
for mode in local depth; do
    python -m scripts.experimental.tt37_joint_deep run /tmp/tt25-rallies.json familiar "$mode" "artifacts/tt37/joint-deep-familiar-$mode.json.gz"
done
python -m scripts.experimental.tt37_joint_deep_report select
mode=$(python -c 'import json; print(json.load(open("artifacts/tt37/joint-deep-choice.json"))["mode"])')
for deck in held_openings fresh_replies; do
    for arm in "$mode" ready; do
        python -m scripts.experimental.tt37_joint_deep run /tmp/tt25-rallies.json "$deck" "$arm" "artifacts/tt37/joint-deep-$deck-$arm.json.gz" --cap 200
    done
done
python -m scripts.experimental.tt37_joint_deep_report summary
python -m scripts.experimental.tt37_joint_deep_report audit
```
