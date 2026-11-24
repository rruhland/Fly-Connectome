# A broader direct joint field against a continuously varied opponent

The recovery field preserved the reach controller’s results, and live updates did not improve them. Both learned versions matched the fixed controller on every held point. The full approach replacements worsened strike quality on practice. [Matched physical results](joint-results.json) · [Controller](../../scripts/experimental/tt37_joint.py)

## Learning from rewarded frame samples

The training source was 46 verified action replays from 41 distinct measured releases. Those replays contained 84 legal near-arm returns and yielded 4,486 stereo and encoder observations. A training sample stored the observed ball position and velocity, the measured joint angles and their rates, and the actuator setpoint issued two ticks later. Only observation windows associated with a legal training return were recorded. The actor evaluated one joint setpoint each tick from the current sensory input. Deployment had no planned stroke or access to the opponent's target. [Capture and replay driver](../../scripts/experimental/tt37_joint_capture.py) · [Training receipt](joint-observations.json)

Four uses of the field were evaluated on the same 18 physical practice points. The plain field used ball motion and encoders. The body field added blade-relative position, its face normal and blade velocity computed from encoder kinematics. The gated field reduced oncoming corrections where the old direct pupil had support. The recovery field changed commands only while the tracked ball moved away after an observed approach. Every field fell back continuously to the fixed visual reach policy when its local data had weak support.

| Practice controller | Near legal returns | Later near returns | Longest alternating rally |
| --- | ---: | ---: | ---: |
| Existing visual reach | 10 | 3 | 7 |
| Plain field | 3 | 0 | 3 |
| Body field | 4 | 0 | 3 |
| Gated body field | 8 | 2 | 5 |
| Recovery field | 10 | 3 | 7 |

Recovery was selected before held scoring. The direct field is most useful here as a candidate for recovery because fitting more rewarded frame samples did not preserve strike quality when the output replaced approach commands. [Practice selection](joint-field-choice.json)

The active condition retained a bounded set of its own recent visual and motor samples whenever the physical referee awarded a legal return. It chose the credit window using the last observed inbound motion. The update could affect the rest of that point and later practice points. The related held treatment carried this memory and froze it for assessment. The fixed field used the same training bank and the same architecture with an empty live memory. [Live practice and updates](joint-practice-active.json.gz) · [Carried memory](joint-carried.json)

## Live physical comparison

The course paired each actor with the same measured launch and opponent seeds. The opponent selected continuously varied goals for its live physical strikes. The 24 novel-opening release IDs were disjoint from the demonstrations and practice. The 16 varied-continuation points used four familiar opening releases and fresh reply seeds. Each held point could reach 200 alternating legal strokes. Scores exclude the launch. [Pinned course](opponent-plan.json) · [Opponent implementation](../../scripts/experimental/tt37_opponent.py)

| Held deck | Fixed reach | Frozen recovery field | Practice-updated recovery field |
| --- | ---: | ---: | ---: |
| 24 novel opening points | 7 near / 0 later | 7 / 0 | 7 / 0 |
| 16 familiar openings with random continuations | 14 near / 2 later | 14 / 2 | 14 / 2 |

All three controllers had 14 contacts from 25 offers on novel points and 25 contacts from 30 offers on continuation points. The maximum was two consecutive near returns on both recovery treatments and the reach control. All 34 delivered opening placements matched exactly across treatment and control, and all six no-offer points reproduced the same referee stream. Live practice kept 301 visual and motor frames from ten legal events. The carried update was queried on 225 and 526 recovery ticks in the two held decks. It did not change any paired score. A field confined to recovery cannot improve first interception, and the result does not test whether a more expressive update can improve the strike itself. [Matched physical results](joint-results.json)

Replaying the highest-scoring point for each arm and deck reproduced the complete event stream and opponent placements. All six audited runs had zero arm penetrations across every physics step, with minimum CAD clearance 18.6 mm. One reply point included an unlanded rim touch after two legal face returns. This is a replay audit of representative best points, not of every evaluated point. [Action-only CAD audit](joint-audit.json)

## Reproduce

Use the pinned measured launches from the [regulation reproduction instructions](../tt29/endurance.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_joint_capture
for mode in reach plain body gated recovery; do
    python -m scripts.experimental.tt37_joint /tmp/tt25-rallies.json artifacts/tt37/opponent-plan.json practice "$mode" "artifacts/tt37/joint-practice-$mode.json.gz"
done
python -m scripts.experimental.tt37_joint_report select
python -m scripts.experimental.tt37_joint /tmp/tt25-rallies.json artifacts/tt37/opponent-plan.json practice recovery artifacts/tt37/joint-practice-active.json.gz --learn --save artifacts/tt37/joint-carried.npz
for deck in held_openings held_replies; do
    python -m scripts.experimental.tt37_joint /tmp/tt25-rallies.json artifacts/tt37/opponent-plan.json "$deck" reach "artifacts/tt37/joint-$deck-reach.json.gz" --cap 200
    python -m scripts.experimental.tt37_joint /tmp/tt25-rallies.json artifacts/tt37/opponent-plan.json "$deck" recovery "artifacts/tt37/joint-$deck-field.json.gz" --cap 200
    python -m scripts.experimental.tt37_joint /tmp/tt25-rallies.json artifacts/tt37/opponent-plan.json "$deck" recovery "artifacts/tt37/joint-$deck-carry.json.gz" --cap 200 --carry artifacts/tt37/joint-carried.npz
done
python -m scripts.experimental.tt37_joint_report summary
python -m scripts.experimental.tt37_joint_report audit
```
