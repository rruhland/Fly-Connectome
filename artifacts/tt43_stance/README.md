# A single intermediate posture can reach new shots, but does not yet sustain broad rallies

The stock-clamp SO-101 has a larger practical range than the 38-return record's deep, negative-face motion suggests. A single intermediate position, reached during the preceding visible outbound flight, admits a clean shallow rescue on a low ball the stock stroke missed. On other balls that same position supports player-driven stereo returns at stock-like depth. The first optical pilot nevertheless shortened both ordinary rallies versus stock. Getting into the new posture is necessary but is not yet a reliable strike policy.

## Concentration and a conditional limit

The reference 38-return rally used a negative face every time. The three clean low rescues in its separate development course also admit negative-face solutions, but they are not close to the record's negative-face branch in actuator coordinates. Record leg 25 hit with wrist roll −1.504 rad, versus +2.641 rad for prior low rescue 4322. Their clocks from actual far separation were 612 and 365 ms. At 4.16 rad/s a single stationary joint reference cannot cover both of those archived strike positions even with zero delay after far separation. The best possible slack for those exact endpoints is −10 ms. After actual table separation the corresponding bound is −341 ms. This argument bounds reference travel to fixed archived joint positions. It does not bound every dynamically achievable contact on those arcs and does not treat the reference-rate bound as a limit on physical overshoot. [Native witness envelope](witness-envelope.json.gz) · [38-return contact audit](../tt42/random-38.json) · [Clean earlier low rescues](../tt40/long-low.md)

## Alternative contact geometry

The geometric screen used 38 actual ordinary incoming arcs from the record and three actual development lows. It sampled seven depths and three contact-normal/pace families on each independently propagated post-bounce flight. Both physical faces were tested. It admitted finite blade-center fits within a 71 mm tangential radius and then required collision-screened links, blade and table with 2 mm minimum clearance on the interpolated path. Actual clean witnessed poses were retained in the candidate sets. None of the geometry results alone claim a legal return.

| Fixed ready option and contact branch | Least slack from exact far separation | Least slack from exact table separation |
|---|---:|---:|
| Stock ready, either face | −58 ms | −306 ms |
| Best sampled common ready, either face | +141 ms | −101 ms |
| Best sampled common ready, negative face only | +23 ms | −277 ms |
| Best sampled common ready, positive face only | −65 ms | −313 ms |

The best sampled q is approximately [−0.45, −0.35, −0.28, +1.17, +0.55] rad at nominal grasp. Its routes reached all 41 rays in the travel screen from far separation and hit the negative face on 37 of 38 preferred ordinary geometric endpoints. Two of the three best low endpoints were negative and one positive. This is one readiness and a choice of strike after observing the incoming ball. It is not a second wing stance or a forced blade-face flip. The earliest inbound fits in the four archived clean low sensory examples appeared 58–77 ms after far separation. Their first visible bounce fits left only 73–104 ms before the low strokes. The numbers favor beginning to move on the incoming estimate and updating with the observed bounce, not waiting in a hub for bounce confirmation. [Actual-ray chart](chart.json.gz) · [Low optical records A](../tt40/long-low-cad-clean-optical.json.gz) · [Low optical records B](../tt40/long-low-alternate-strict-optical.json.gz)

## Native feasibility and real optical play

An exact-state development coach moved to the common hub through physical bounded commands starting just after the prior near shot. No foreknowledge was used to change the far opponent, which delivered exactly the original far impact and near-table bounce. The body was within 0.0008 rad of the common hub at the chosen request tick, 83–100 ms after far separation. A 64-candidate, at most three-round local request used both signed poses and the unchanged rollout. On low 4322 the selected new stroke from that hub landed legally, unlike stock. The physical near collision was a positive-side rim hit at y = −1.053 m and z = 0.881 m, followed by a live opponent reply. The paired point rose from five to six physical arm landings. Independent tape replay reproduced every event and placement, with zero penetrations in 4,864 full CAD states and a 15.77 mm minimum distance. The same search missed the two chosen ordinary control cases. A shorter backswing on the original two-case roster yielded no usable native solution and is included as a negative. The successful physical correction is only an offline teacher example. [First matched forks](native.json.gz) · [Teacher and second normal control](native-teacher.json.gz) · [Shortened-backswing control](native-direct.json.gz)

The optical actor starts in the original boot state. Outbound stereo motion triggers the move to this same hub on each reception. From inbound RGB, its isolated predictor gets only stereo and own encoders/issued targets. It computes bounded early strike proposals around the predicted ray and optionally gives its normal-depth trajectory to the regular stock ensemble after a visible bounce. The ordinary fallback only replaces the early drive on its usual robust validation rule. Neither actor is told a shot ID, target, true bounce, or native ball state. All physical games use the stock clamp, motor envelope, paddle and same independent continuous opponent draw schedule.

The first frozen pilot used the first two direct-course assessment cards and the stock checkpoint after 12 live practice games. Pilot scores were 1 and 1 versus 5 and 10 stock. The optical policy reached within 0.02–0.06 rad of the new hub on recorded inbound cues in 9550 and returned its first ordinary ball on a new early negative-face movement. It also made a physical first return in the second game. One of its next nominally legal early plans on 9550 actually sent the ball back into its own half. The initial pilot commits a pre-bounce plan without adjusting it to the rebound, an avoidable limitation. In the follow-up on those identical cards, giving the deep trajectory back to the stock robust planner at observed rebound raised 9550 from 1 to 2 and left 15665 at 1. This does not preserve ordinary performance. A separate live practice card, 10329, started with the physically earned 4322 correction and made no legal arm return, so it did not update that correction bank. These are small pilots, not a score-improving qualification. [Original fixed-hub games](held.json.gz) · [Deep bounce handoff](handoff.json.gz) · [Disjoint practice](practice.json.gz) · [Matched frozen control](../tt42/after12.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt43_stance
python -m scripts.experimental.tt43_stance
python -m scripts.experimental.tt43_stance_chart
python -m scripts.experimental.tt43_stance_native --limit 2 --output "$out/native.json.gz"
python -m scripts.experimental.tt43_stance_native --schedule direct --limit 2 --output "$out/native-direct.json.gz"
python -m scripts.experimental.tt43_stance_native --cases 1 2 --output "$out/native-teacher.json.gz"
python -m scripts.experimental.tt43_stance_live
python -m scripts.experimental.tt43_stance_audit "$out/held.json.gz" "$out/held-audit.json"
python -m scripts.experimental.tt43_stance_live --handoff --output "$out/handoff.json.gz"
python -m scripts.experimental.tt43_stance_audit "$out/handoff.json.gz" "$out/handoff-audit.json"
python -m scripts.experimental.tt43_stance_live --handoff --teacher "$out/native-teacher.json.gz" --mode live --deck practice --start 12 --limit 1 --output "$out/practice.json.gz"
python -m pytest -q tests/test_tt43_stance.py
```
