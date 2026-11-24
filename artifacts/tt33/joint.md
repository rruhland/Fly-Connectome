# Joint feedback from rewarded table-tennis practice

A frozen continuous controller produced six legal SO-101 returns across eight new measured releases against a live opponent. Two points reached five alternating legal returns. A second controller used optical ball position and velocity and the near arm’s measured joint positions. It produced three legal returns without computing time or position at a receiving plane. Both controllers sent a new vector of joint-rate commands every control interval. [Implementation](../../scripts/experimental/tt33_joint.py)

## Training and execution

The demonstrations came from 30 rewarded movements on five earlier releases, all using the same calibrated left-ready pose. During training, replay reproduced the archived physical events and recorded the RGB-derived optical state, near-arm encoders and the demonstrated joint command targets. A fit retained Gaussian basis centers and weights over individual observations. Weighted ridge regression emphasized frames near legal impact and used no backpropagation. The targets used for regression were the teacher’s individual actuator setpoints. The deployed inputs were optical observations, measured joint positions in the raw-input variant, and the pupil’s own command reference. The checkpoint stored only the observation basis and the fitted joint coefficients. [Rewarded practice](../tt31/ready-practice-staged.json.gz) · [Recovery demonstrations](../tt31/confirm-recovery.json.gz) · [Side-switch demonstrations](../tt31/transition-6810-reference.json.gz) · [Additional practice](../tt31/transition-practice.json.gz)

The plane-based controller estimated arrival at a fixed observation plane 1.29 m from the net using stereo RGB at 30 Hz. At 60 Hz it evaluated a smooth joint feedback surface over time to the plane and recent optical crossing estimates. The raw-input controller used the fitted ball’s present position and velocity along with measured joint positions. It evaluated the current observation directly on each control tick. In both versions the same plant enforced the arm’s joint limits and torque envelope. [Plane checkpoint](joint-plane.json.gz) · [Raw-input checkpoint](joint-raw.json.gz)

For corrective practice, the pupil controlled the first five active ticks of two earlier points. The optical teacher took over from the actual encoder and command state and landed two returns on each point. A second fit added 126 observations of teacher commands from rewarded exchanges after the handoff. On the eight-release pilot, both plane fits landed two near returns. On the independent fresh roster, both landed six and the corrective fit produced one more alternating legal return. [Corrective practice generator](../../scripts/experimental/tt33_joint_corrections.py) · [Handoff tapes](joint-corrections.json.gz) · [Corrected checkpoint](joint-corrected.json.gz) · [Plane pilot](joint-pilot-plane.json.gz) · [Corrected pilot](joint-pilot-corrected.json.gz) · [Raw-input pilot](joint-pilot-raw.json.gz)

## Independent fresh games

The fresh roster used eight release IDs absent from training. Its first physical deliveries included three left, two middle and three right. The fixed recovery teacher, neutral arm and pupils received identical first replies at the opponent contact, near bounce and reference-plane crossing. Subsequent balls were produced by the live opponent responding to each arm’s physical return. Counts exclude the recorded launch. [Fresh roster](../tt32/fresh-plan.json.gz) · [Teacher tapes](../tt32/fresh-fixed.json.gz)

| Controller | Near legal returns | Alternating legal returns | Longest point | Blade contacts |
| --- | ---: | ---: | ---: | ---: |
| Stationary near arm | 0 | 8 | 1 | 2 |
| Fixed optical teacher with recovery | 4 | 16 | 5 | — |
| Frozen plane-based pupil | 6 | 19 | 5 | 7 |
| Plane-based pupil after corrective practice | 6 | 20 | 5 | 8 |
| Frozen raw-input pupil | 3 | 14 | 3 | 6 |

The plane-based pupil landed five left-side and one right-side deliveries. Its two-return games started with releases 14993 and 8239. On 14993 it landed a right-to-left switch. The raw-input pupil also landed a right-side ball on 14993 and made two legal left-side returns in other games. Middle deliveries produced zero legal pupil returns. The assessment covers eight points from a shared ready pose. [Plane games](joint-fresh-plane.json.gz) · [Corrective games](joint-fresh-corrected.json.gz) · [Raw-input games](joint-fresh-raw.json.gz) · [Stationary games](joint-fresh-empty.json.gz)

The plane-based pupil moved at a normalized command magnitude above 0.01 on 610 of 1,327 ticks and reached a peak measured joint speed of 4.78 rad/s. Its 95th percentile absolute command change by axis was 0.39, 0.29, 0.31, 0.19, 0.33 and 0.00. Command-only replay matched every referee event and counted zero penetrations across 10,586 full-CAD physics steps. Minimum clearance was 18.6 mm. All six legal returns used the same blade face. The raw-input pupil issued active commands on 582 of 1,136 ticks and reached 4.69 rad/s. The corrected pupil issued active commands on 682 of 1,389 ticks and reached 4.73 rad/s. Full-CAD replay found zero penetrations in both variants, with 18.6 mm minimum clearance for the raw-input pupil and 15.9 mm for the corrected pupil. The fresh game files include action activity, rates, forces and process CPU time. [Plane replay audit](joint-fresh-plane-audit.json) · [Corrected replay audit](joint-fresh-corrected-audit.json) · [Raw-input replay audit](joint-fresh-raw-audit.json)

## Reproduce

Use the pinned measured launches from the [endurance reproduction instructions](../tt29/endurance.md). The action replay checks the physical referee stream and placement measurements independently of the saved policy.

```sh
export PYTHONPATH=src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt33_joint capture --output /tmp/tt33-plane.npz
python -m scripts.experimental.tt33_joint capture --raw --output /tmp/tt33-raw.npz
python -m scripts.experimental.tt33_joint fit /tmp/tt33-plane.npz /tmp/tt33-plane.json.gz --features context --width .9 --ridge .015 --impact-weight 4
python -m scripts.experimental.tt33_joint fit /tmp/tt33-raw.npz /tmp/tt33-raw.json.gz --features raw-proprioception --width 1.05 --ridge .015 --impact-weight 4

python -m scripts.experimental.tt33_joint fit /tmp/tt33-plane.npz /tmp/tt33-correction-pupil.json.gz --features height --width .9 --ridge .015 --impact-weight 4
python -m scripts.experimental.tt33_joint_corrections /tmp/tt25-rallies.json artifacts/tt31/confirm-recovery.json.gz /tmp/tt33-correction-pupil.json.gz /tmp/joint-corrections.json.gz
python -m scripts.experimental.tt33_joint capture --sources /tmp/joint-corrections.json.gz --output /tmp/tt33-corrections.npz
python -m scripts.experimental.tt33_joint fit /tmp/tt33-plane.npz /tmp/tt33-corrected.json.gz --additional /tmp/tt33-corrections.npz --features context --width .9 --ridge .015 --impact-weight 4

for model in plane raw corrected; do
    python -m scripts.experimental.tt33_joint evaluate /tmp/tt25-rallies.json artifacts/tt32/fresh-plan.json.gz /tmp/tt33-fresh-$model.json.gz --checkpoint /tmp/tt33-$model.json.gz
    python -m scripts.experimental.tt33_joint verify /tmp/tt33-fresh-$model.json.gz /tmp/tt33-fresh-$model-audit.json --cad
done
python -m scripts.experimental.tt33_joint evaluate /tmp/tt25-rallies.json artifacts/tt32/fresh-plan.json.gz /tmp/tt33-fresh-empty.json.gz
```
