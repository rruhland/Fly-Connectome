# Learning from the shot that comes back

Learning from the following exchange did not improve return counts. From identical acquired experience, the continuation-ranked and immediate-ranked actors both matched the unmodified readout at 25 legal arm returns over 20 new continuations and 7 over 24 unseen releases. [Matched receipts](online-pace-results.json)

An outgoing stroke can change what the physical opponent sends next, even when the opponent's placement goal is independently drawn and never visible to the near arm. This test asked whether a direct controller could learn a more sustainable stroke by crediting the next actual exchange. It separately measured learning from the current shot alone. [RGB and encoder control](../../scripts/experimental/tt37_online_pace.py) · [Pinned practice and test draws](online-pace-plan.json)

The near arm used the same camera and encoder body-reach controller as the earlier experiment. On an incoming observed ball it chose one of three late-phase variations. The default kept the existing joint readout. Two small signed edits combined a shift in its present-depth phase coordinate with an orientation adjustment projected to reduce displacement of the paddle center. The live body estimate and the ball-relative gate still read the unchanged current camera observation. No estimated intercept, future trajectory, far aim, seed, or release ID entered the actor. These are local stroke variations, not a separate search planner. [Direct body model](../tt36/reach.md)

The practice course repeated five already returnable opening families four times. Each point continued against a fresh independently drawn reply sequence. It physically tested the three alternatives through seeded exploration. A legal return received one unit immediately in the logged outcome, but the continuation learner stored that return after observing or failing to observe its next exchange. It also received 0.14 when there was another observed approach and up to 0.60 for the next measured own outcome. If the following strike landed legally it earned the full additional 0.60. Otherwise outgoing RGB motion or the measured ball-to-blade gap could give smaller credit. The two held treatments loaded exactly the same experience. One used the continuation value and the other used only the initial score or optical grade. An unmodified joint readout supplied a matched fixed control. [Practice outcomes](online-pace-practice-learn.json.gz) · [Frozen practice control](online-pace-practice-explore.json.gz) · [Retained experience](online-pace-checkpoint.json)

The edits physically changed some outgoing balls. On the same opening of release 14832, a signed edit moved the measured forward velocity from 3.36 to 3.15 m/s while both strokes landed. Both subsequent balls ended in a near-arm fault. This measures a realizable change in the outgoing hit, not a reliable improvement. [Optical and physical measurements](online-pace-practice-learn.json.gz)

| Course | Control | Immediate ranking | Continuation ranking |
| --- | ---: | ---: | ---: |
| 20 practice points | 15 with exploration but frozen state | — | 14 while learning |
| 20 held continuations after the five practiced openings | 25 | 25 | 25 |
| 24 held independent opening releases | 7 | 7 | 7 |

Both practice branches used 11 actual late edits. The continuation-ranked controller did not choose any of the edits on the held courses. The immediate ranking tried one on a held unseen point without changing its score. The held result therefore shows that the learned test preferred the original stroke, not that applying either signed edit is always harmless. The 20 continuation games in each branch made five later legal arm returns after 20 openings, with at most two per point. None of the unseen releases produced a later legal arm return. [Paired result breakdown](online-pace-results.json) · [Command-only replay](online-pace-audit.json)

## Reproduce

Use the measured feeds and the shared `opponent-plan.json` for anchored or unseen releases.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_online_pace plan
python -m scripts.experimental.tt37_online_pace practice learn
python -m scripts.experimental.tt37_online_pace practice explore
for course in later new; do
  for mode in fixed immediate carry; do
    python -m scripts.experimental.tt37_online_pace "$course" "$mode"
  done
done
python -m scripts.experimental.tt37_online_pace summary
python -m scripts.experimental.tt37_online_pace audit
```
