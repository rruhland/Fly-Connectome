# Receding joint search from the camera and encoders

The elastic direct-joint controller with live learning reached 49 alternating legal strokes in a natural simulated rally, with 24 returns by the near arm. On 12 novel releases it landed 78 legal near returns, compared with six from raw feedback using the same camera readout. Live reward memory raised later returns from 18 to 35 on the held continuation deck, but the frozen search produced a longer individual rally of 53 strokes on a novel release. None of these runs reached 100. [Paired physical results](joint-mpc-results.json) · [49-stroke replay](joint-mpc-extended-audit.json)

The controller revises short joint commands in a separately built nominal body model. The live game supplies stereo images, joint feedback, the controller’s own previous commands, and the actual referee signal for learning. The nominal model is never given the live physics state, marked ball spin, an opponent target or the incoming release identity. It is reset from its own calibrated plant and tethered to the measured joints. Ball position and velocity come from the same camera readout used for the paired raw control. [Implementation](../../scripts/experimental/tt38_joint_mpc.py)

The instruction archive contributes candidate fragments of at most 21 control ticks, grouped by the actual rewarded training strikes so a nearby cluster of nearly identical frames does not flood the search. A candidate can be the current raw controller, a shifted previous winner, a nearby joint-command fragment, or a bounded perturbation in position and timing. The simulated moment of closest ball–blade approach supplies a local motor correction. No motion is aimed at a fixed receiving plane, generated from an inverse-kinematics stroke chart, or issued as a complete picked stroke. The short physics screen compares contact and outgoing direction, then a smaller set receives full landing and optical uncertainty checks. The winning action can influence at most five ticks before becoming stale. The elastic candidate also permits a shorter, bounded preparatory correction if every candidate misses but the independent model finds a substantial reduction in separation. [Frozen course and sources](joint-mpc-protocol.json)

The live learner starts with an empty reward memory on each evaluation deck. When the actual referee awards a legal near return, it can store a few recent, camera-observed and already executed joint fragments around that approach. A bounded memory holds at most 192 fragments. A retained fragment still has to pass physics screening before it can change an action. Every point records memory content hashes before and after the point. The frozen control preserves the same search, camera and preexisting instruction bank with no real-outcome credit. [Paired score and memory audit](joint-mpc-results.json)

## Physical comparisons

The original and elastic searches are compared on six development points with new reply draws. The method is selected by later legal returns before held scoring, then evaluated on a pinned 12-release novel opening slice and an eight-point continuation slice of the prior random-opponent roster. Novel opening releases are held out of the whole rewarded instruction archive. Continuation cards retain familiar opening measurements but get separate reply seeds. The frozen/live comparison is on the continuation slice. The paired raw control uses the same readout as the search. The original RGB readout of the raw pupil is also reported where an exact archived counterpart exists. [Selection rule and breakdown](joint-mpc-choice.json)

| Six development points | Legal near | Legal later | Best near / alternating |
| --- | ---: | ---: | ---: |
| Raw joint field, matched camera readout | 2 | 0 | 1 / 3 |
| Joint-fragment search | 21 | 16 | 7 / 15 |
| Elastic joint-fragment search | 33 | 28 | 18 / 36 |

The elastic controller reached the development cap at 36 alternating legal strokes on release 5142. A separately executed natural extension raised the cap to 200 and reran the same four-point prefix from an empty memory. It reproduced the first three complete points and all commands and events through the prior cutoff, then finished at 24 near returns and 49 alternating legal strokes. That 40.67-second simulated rally used 42.0 CPU seconds for 99 search refreshes and 83.5 CPU seconds overall. The final opponent shot was delivered legally down the middle. The model found nominal contacts on successive refreshes but no legal action, and the actual simulated arm missed. Independent action replay reproduced the complete referee history and deliveries. All 24 legal contacts used the negative face and spanned 59.4 cm laterally. Full arm CAD found zero penetrations across 19,521 physics samples, with minimum clearance of 15.63 mm. [Extended action tape](joint-mpc-extended.json.gz) · [Replay and CAD receipt](joint-mpc-extended-audit.json)

| Held games and controller | Legal first | Legal later | Legal near | Best near / alternating |
| --- | ---: | ---: | ---: | ---: |
| 12 novel releases, raw with matched readout | 6 | 0 | 6 | 1 / 3 |
| 12 novel releases, raw with original camera readout | 6 | 3 | 9 | 4 / 9 |
| 12 novel releases, elastic with live memory | 9 | 69 | 78 | 15 / 31 |
| 8 held continuations, raw with matched readout | 4 | 0 | 4 | 1 / 2 |
| 8 held continuations, raw with original camera readout | 4 | 2 | 6 | 2 / 5 |
| 8 held continuations, elastic with memory frozen empty | 4 | 18 | 22 | 10 / 21 |
| 8 held continuations, elastic with live memory | 4 | 35 | 39 | 18 / 37 |

Exact paired opening trajectories matched on all 19 delivered first shots. The one novel no-offer point reproduced the same events. Novel elastic beat the same-readout raw controller on eight points, tied four, and lost none. Its later legal returns covered the left, middle, and right lanes. The continuation slice used two launch recordings and had a clear failure mode. All four 5530 openings scored zero with elastic and one with raw. All four 14832 openings scored at least two with elastic and zero with raw. Updating reward memory improved three of the four successful repeat continuations over the same frozen search and lost none. These totals separate learning from running search, but the live arm can also learn across points. [Per-point outcomes](joint-mpc-results.json)

The search used 141.1 CPU seconds on the novel deck, inside 284.1 total CPU seconds for 140.0 simulated seconds. On continuations the live search used 76.6 CPU seconds inside 150.9 total CPU seconds for 73.6 simulated seconds. This is a physically simulated result, not a hardware trial or real-time application. [Timing and candidate counts](joint-mpc-results.json)

Action-only replay reproduced the strongest novel point, live continuation, and frozen continuation. It found 43 legal near contacts on the negative paddle face, no arm penetrations over 34,646 physics steps, and a minimum clearance of 15.0 mm. This audit covers the selected strong points, not every assessed volley. [Held CAD and referee replays](joint-mpc-audit.json)

## Isolate updates within one point

The two leading novel cards were replayed with the memory frozen. The first point had no legal near return in either run and exactly matched every command and event. Both controllers therefore entered the chosen second point, release 4640, with empty memory. Commands first diverged at 4.58 simulated seconds, after two actual near returns could seed the live memory. Frozen search ultimately made 26 near returns in 53 alternating strokes. Live search made 13 near returns in 27 alternating strokes. This counterexample was chosen after observing the novel live result to isolate the effect of within-point updates. It is not another independent generalization trial. The reward memory helps some continuation sequences but can hurt others. [Within-point receipt](joint-mpc-within-results.json)

## Reproduce

Use the measured launches from the [regulation reproduction instructions](../tt29/endurance.md) and the preserved TT37 evidence, including the [joint archive receipt](../tt37/long-rgb.json). Set the native physics and numerical workers to one.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
optic=artifacts/tt37/rebound.npz
protocol=artifacts/tt38/joint-mpc-protocol.json
python -m scripts.experimental.tt38_joint_mpc protocol development "$protocol" --optic "$optic"
python -m scripts.experimental.tt38_joint_mpc raw development artifacts/tt38/joint-mpc-development-raw.json.gz --optic "$optic" --cap 36
python -m scripts.experimental.tt38_joint_mpc live development artifacts/tt38/joint-mpc-development-live.json.gz --optic "$optic" --cap 36
python -m scripts.experimental.tt38_joint_mpc live development artifacts/tt38/joint-mpc-development-elastic.json.gz --optic "$optic" --cap 36 --variant elastic
python -m scripts.experimental.tt38_joint_mpc_report selection
python -m scripts.experimental.tt38_joint_mpc live development artifacts/tt38/joint-mpc-extended.json.gz --optic "$optic" --cap 200 --limit 4 --variant elastic
python -m scripts.experimental.tt38_joint_mpc_report extension
variant=$(python -c 'import json;print(json.load(open("artifacts/tt38/joint-mpc-choice.json"))["variant"])')
for arm in raw live; do
    python -m scripts.experimental.tt38_joint_mpc "$arm" held_openings "artifacts/tt38/joint-mpc-held_openings-$arm.json.gz" --optic "$optic" --protocol "$protocol" --limit 12 --variant "$variant"
done
for arm in raw live frozen; do
    python -m scripts.experimental.tt38_joint_mpc "$arm" held_replies "artifacts/tt38/joint-mpc-held_replies-$arm.json.gz" --optic "$optic" --protocol "$protocol" --limit 8 --variant "$variant"
done
python -m scripts.experimental.tt38_joint_mpc_report comparison
python -m scripts.experimental.tt38_joint_mpc_report audit
python -m scripts.experimental.tt38_joint_mpc frozen held_openings artifacts/tt38/joint-mpc-within-frozen.json.gz --optic "$optic" --limit 2 --variant elastic
python -m scripts.experimental.tt38_joint_mpc_report within
```
