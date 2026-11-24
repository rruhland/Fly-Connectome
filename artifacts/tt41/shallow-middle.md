# A grip-branch reach limit at the bottom of the normal stroke

Moving the unchanged, already-prepared arm to an intermediate contact depth did not rescue the moderately low terminal balls in this practice-prefix pilot. The interior of the blade was locally reachable on one of three actual arrivals and not the other two. The common elbow stop, not the modeled joint travel deadline, separated those cases in a same-branch geometric probe.

## Uninterrupted physical corrections

The three receptions were taken from practice/development points with different release IDs from the replication. They had already produced 6, 5 and 11 legal arm returns, respectively. All three terminal arrivals belonged to the same moderate band when projected at the stock reception line. The paired replays reproduced the full native event prefix before issuing a trial motion. The existing target and measured joint positions and rates stayed continuous. The actual opponent used the original release seed, reply seed and continuous distribution throughout. The selected reception and candidate cue ticks were scheduled retrospectively. One fork fit an optical twin from rendered stereo and body encoders. The other privileged comparison initialized the same search from the exact actual state. Neither fork is an end-to-end deployable policy.

| Release | Projected height | Actual stock-depth touch | First-impact-gated result |
|---|---:|---:|---|
| 7038 | 1.024 m | −1.549 m | No legal proposal at the requested depth |
| 5864 | 0.997 m | −1.555 m | No aligned seed passed the broad earlier screen |
| 4563 | 1.005 m | −1.573 m | No legal proposal at the requested depth |

The bounded search tried five planned depths from −1.325 to −1.480 m, with a smooth handoff from the real stock command and the stock target slew guard. On 7038 and 4563, the corrected native screen performed 1,856 dynamics trials at three cue times per point. The first physical impact ball center was in the intermediate window on 138 trials. This window allows small axial and discretization tolerance for the ball center and does not use outgoing separation as an impact proxy. No candidate in that window returned legally, so neither corrected replay changed the stock action prefix or score. The two optical forks took about 19 process CPU seconds combined. [7038 corrected fork](shallow-middle-strict-7038.json.gz) · [4563 corrected fork](shallow-middle-strict-4563.json.gz)

An earlier screen was deliberately checked against native contacts and rejected. It used the separation position, which can be several centimetres farther forward than first touch. The stock-depth state-initialized 7038 trial was physically legal at **−1.542 m**, and the real far opponent answered it at 12.88 seconds, advancing from 6 arm and 7 far returns to 7 and 8. It is a positive same-depth reach control, not an intermediate result. The selected optical forks for 7038 and 4563 touched at −1.541 and −1.548 m and did not return. The 5864 fork never found a seed. Full-CAD replay of the positive fork reproduced all events with zero penetrations across 6,184 solver states. [7038 preliminary](shallow-middle-7038.json.gz) · [5864 preliminary](shallow-middle-5864.json.gz) · [4563 preliminary](shallow-middle-4563.json.gz) · [7038 physical audit](shallow-middle-7038-cad.json)

## Why the deep branch touches the rim

The original command tapes were replayed without editing. The table rebound and ensuing ball ray are from the actual native referee, not a fitted camera view. The physical solver contact point is near the 76 mm blade edge in every case. The projected ball center in the tangent plane can lie farther out than the solver contact point, which is expected for a finite ball meeting a rounded edge. The actual elbow angle at contact was at the mechanical hard stop in all three, with up to 1.2 mrad of simulation compliance past it.

| Release | Actual solver contact radius | Actual ball-center tangent | Shoulder angle above lower stop | Best centered plane at y≈−1.48 | Best centered plane at y≈−1.56 | Deep plane with virtual extra lift/elbow |
|---|---:|---:|---:|---:|---:|---:|
| 7038 | 75.6 mm | 76.7 mm | 45.9 mrad | 62 mm | 60 mm | <1 mm |
| 5864 | 73.6 mm | 89.1 mm | 3.0 mrad | 89 mm | 83 mm | 16 mm |
| 4563 | 75.0 mm | 78.0 mm | 10.9 mrad | 79 mm | 77 mm | 10 mm |

The local same-signed-branch endpoint fits start from the actual prepared and contact poses. They test five face and speed families at seven positions on the real post-bounce ray. An accepted endpoint has 11–37 mm projected axial spacing, face alignment above 0.925, no CAD overlap, and a clear interpolated path from the stock preparation at an optimistic 4.16 rad/s. The available times were 0.25–0.35 s. Numerically best joint-constrained solutions in the table sit at the shoulder and elbow lower bounds. Removing the deadline does not change the radii. Toward y = −1.32 to −1.44 m no tested pose reaches far enough along the face normal. Extending only the shoulder and elbow range by ±0.18 rad produces the final-column values, which are FK sensitivity probes without CAD or actuator validity. No body or physics was changed in play. This is evidence for a local joint-envelope issue, not proof that an alternate signed branch or handle seating cannot create clearance. [Exact stock contact and pose probes](shallow-middle-radius.json)

## Reproducing this pilot

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_shallow_middle 7038 /tmp/strict-7038.json.gz
python -m scripts.experimental.tt41_shallow_middle 4563 /tmp/strict-4563.json.gz
python -m scripts.experimental.tt41_shallow_middle 5864 /tmp/first-5864.json.gz --separation
python -m scripts.experimental.tt41_shallow_middle 7038 /tmp/cad.json --audit artifacts/tt41/shallow-middle-7038.json.gz
python -m scripts.experimental.tt41_shallow_radius /tmp/radius.json
```
