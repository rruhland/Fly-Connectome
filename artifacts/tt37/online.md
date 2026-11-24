# Direct learning between exchanges on the random court

The learner used newly earned feedback before later approaches in the same physical point. That update did not improve held scoring. It made 6 legal arm returns on 24 unseen releases versus 7 for the frozen visual-reach policy, and both made 14 over 16 changed continuations. [Paired results and per-point records](online-results.json) · [Controller](../../scripts/experimental/tt37_online.py)

The direct joint pupil receives current stereo ball position and velocity and the joint encoders. An online selector indexes a bounded outcome memory by the current ball relative to the estimated blade center, its velocity, and measured joint pose. It chooses from six mixtures of the prior direct readout, live empirical body feedback, a small inherited joint correction, and a local reach hint. The live body map comes from earlier encoder and RGB pairs. A visible ball-to-blade gap can teach a signed correction by solving a damped local inverse at the measured encoder posture. That inverse acts on an observed, past miss, not a predicted interception. The current stroke is credited as soon as the referee awards a return or the observation records a finished miss. The next stroke can retrieve the new evidence before the point ends. [Original body map and relative alignment observations](../tt36/reach.md) · [Learning commands and feedback](online-practice-learn.json.gz) · [Final checkpoint](online-practice-learn-checkpoint.json)

| Physical course | Matched control, near returns | Learner, near returns | Later near returns, control / learner |
| --- | ---: | ---: | ---: |
| 18 practice games, same exploration seeds | 8 | 8 | 2 / 2 |
| 24 new randomly delivered releases | 7 | 6 | 0 / 0 |
| 16 fresh continuations after familiar openings | 14 | 14 | 2 / 2 |

Eight following approaches looked up the newly updated memory within a point. Practice took 38.5 process CPU seconds and supplied 21 scored or optical outcomes, five with a usable signed miss correction. The matched exploration control chose with the same per-point random seeds but could not retain outcomes. All practice point scores tied. On the unfamiliar course, the opponent itself never supplied a legal first ball in six matched points. None were dropped from the totals. On changed continuations one point grew from two to three returns and another dropped from two to one. The held memory did not update. [Matched exploration commands](online-practice-frozen.json.gz) · [New releases, body](online-held_openings-body.json.gz) · [New releases, carry](online-held_openings-carry.json.gz) · [Changed continuations, body](online-held_replies-body.json.gz) · [Changed continuations, carry](online-held_replies-carry.json.gz)

Every changed held score replayed from the issued commands alone. Full-CAD replay checked 5,153 solver states across three games, with no penetrations and a minimum clearance of 18.6 mm. [Replay verification](online-audit.json)

## Separate model-assisted ceiling

The older controller uses RGB and encoders but also has a stroke planner, numerical body kinematics and a private forecast simulator. We ran it on three TT36-style openings with fresh independently drawn later target coordinates, using the same shots across conditions. Starting from an empty motor archive and adding actual rewarded motions in play made 13 alternating returns, versus 11 with an empty frozen archive. With the older 238-example archive loaded but frozen, it made 13. The last point grew from five alternating returns without memory to seven with live retention. This pilot does not establish a long rally or a general advantage and must not be grouped with the direct result. [Empty/frozen](online-planner-pilot-cold.json.gz) · [Empty/live retention](online-planner-pilot-carry.json.gz) · [Inherited/frozen](online-planner-pilot-warm.json.gz)

## Reproduce

Use the pinned measured feeds and shared `opponent-plan.json`.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_online practice learn
python -m scripts.experimental.tt37_online practice frozen
for course in held_openings held_replies; do
  python -m scripts.experimental.tt37_online "$course" body
  python -m scripts.experimental.tt37_online "$course" carry
done
python -m scripts.experimental.tt37_online summary
python -m scripts.experimental.tt37_online audit
for mode in cold carry warm; do
  python -m scripts.experimental.tt37_online_planner pilot "$mode" --limit 3
done
```
