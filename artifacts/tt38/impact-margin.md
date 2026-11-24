# An impact margin for the physical paddle

The earlier camera rescue got the paddle to the ball within 2.1 ms of prediction on all five issued strokes. Four still landed illegally. This follow-up changes what the grounded arm is asked to do at contact, and measures whether those changes survive plausible error in the private prediction. It is a model-based motor controller with scored movement memory, not a distilled direct learner or a privileged physical teacher. [Earlier rescue](camera-rescue.md)

The proposed face normal points along the difference between a centered outgoing flight and the observed approaching velocity. For a nominal normal restitution of 0.76, the desired blade normal speed is `(desired_normal_velocity + 0.76 * incoming_normal_velocity) / 1.76`. The wrist and other joint velocities come from a damped five-joint solve for center velocity with a penalty on face rotation. An eased sweep maintains the controlled face through a small arrival-time band. A second family trims tilt and normal speed around the already moving legacy stroke. It solves for the tilt change while minimizing blade-center translation, so the motion need not resolve the entire reach again.

The planner ranks these schedules in the unlaunched court initialized from RGB-derived ball flight, joint encoders and issued actuator references. It prefers center contact, clearance, a broad interior landing and an approachable far-side bounce. Six antithetic perturbations stress the position, velocity and unknown spin of the camera-estimated ball. The perturbations are fixed engineering stress cases informed by the observed camera/contact errors, not a fitted posterior or a claim to span all departure errors. A physically selected replacement must have a nominal legal landing and beat the stock candidate by a fixed score margin, with at least four legal predictions in seven settings. A failed review preserves the stock proposal. Nothing in the decision receives launch cards, opponent random seeds, real MuJoCo data, or referee geometry. Real returns still reward the ordinary actuator and value memories, but the new schedules are excluded from their incompatible original ten-parameter movement bank.

Whole games retain the independently reacting broad-range far paddle and the unchanged 200 alternating-return cap. The two paired comparisons reuse the chronological closed-loop school control and independent cold-value control. Both conditions begin each roster with identical compared memory. Within a roster learning from physically scored results can diverge naturally when it sees different balls. Real native contact and legal results, and whether the inherited late servo altered an issued candidate, are read only after play for the audit.

## Whole-game results

The controller produced accurate physical departures when it selected a replacement, but did not make rallies longer overall. All 31 issued replacements made physical contact. Thirty were legal and received a legal reply. The largest absolute error in predicted vertical departure among the 31 was 0.27 m/s. None was overwritten by the inherited late servo. In the seven replacements where the archive and intervention had identical actual command prefixes, all seven stock strokes also landed legally. Six stock strokes and all seven replacements were answered. Moreover, 30 of the 31 stock strokes were nominally legal in the private model before replacement. The selected motions mostly solve easy alternatives, not rescue genuinely failing receptions.

| Conditions | Arm legal returns, stock → margin | All alternating returns | Longest point, arm / both | Process CPU |
| --- | ---: | ---: | ---: | ---: |
| 12 matched school | 81 → 77 | 174 → 166 | 14 / 29 → 15 / 31 | 209.8 s → 290.0 s |
| 24 independent cold-value | 68 → 37 | 157 → 97 | 18 / 37 → 7 / 15 | 206.6 s → 187.3 s |

On 35 points the last legal stroke was the opponent's. All 35 had a final review. Twenty-six had no nominally legal replacement at that final reception, eight still made no replacement, and one replacement was physically attempted and failed. The risk ranking was more reliable than the ability to generate a viable trajectory when the arm was already in trouble. The next test protects already-viable stock movements and tries real earned approach histories earlier. [Protected early follow-up](../tt39/margin-practice.md)

School release 3920 improved from eight to eleven arm returns. The physical inputs were identical before the first replacement. Both replacement strokes landed and were answered. Full CAD replay reproduced stock and margin exactly, and verified 7,051 and 9,540 solver states respectively without penetration. The longest margin point, 13250, reached fifteen returns against two in stock, but earlier reward memory had already changed its motor commands before the first replacement on that point. It does not isolate the effect of the four later accepted trims. [3920 CAD replay](impact-margin-3920-cad.json)

[School physical tape](impact-margin-closed.json.gz) · [School comparison](impact-margin-closed-summary.json) · [Independent physical tape](impact-margin-independent.json.gz) · [Independent comparison](impact-margin-independent-summary.json)

## A separate correction for rim strikes

In the previous contact-identification receipts, absolute vertical departure error for predicted contacts at a radius up to 40 mm had an upper quartile of 0.28 m/s on its fit sample (6) and 0.15 m/s on held strokes (3). Above 40 mm it was 2.12 m/s (21) and 2.35 m/s (12), respectively. Only cases with both an optical outcome and a predicted touch enter this conditional comparison. The small sample describes a possible near-rim failure mechanism, not a calibrated global error law. [Fit measurements](contact-id-fit-observations.json) · [Held measurements](contact-id-held-observations.json)

A separate intervention tests eight identified last failed strokes. The design replays the archived point from its original launch and verifies the physical events and camera/encoder context up to the intervention. It then reconstructs the incumbent collision in another isolated snapshot of the optical court. The actual real ball is never used to pick a direction. Only the hypothetical point of contact relative to the modeled blade center determines an in-plane correction. A joint solve translates the paddle along that vector and checks alternatives with smaller apparent contact radius, some with a small tilt change. It only issues nominally legal schedules whose private correction remains better under directional uncertainty. These are targeted forks on known failures, not evidence from a blind whole-game rollout.

On independent release 11171 the archive had 18 arm returns and the final arm stroke went out. At the same tick and on the same camera and encoder data, the centering motion moved the requested blade center about 6.5 cm within the face plane, with a 0.082-radian tilt offset. The private contact radius decreased from 73.3 to 46.8 mm and legal outcomes increased from four to six of seven tested states. The real ball left the revised blade at (−0.774, 2.841, 3.514) m/s, compared with (−0.791, 2.795, 3.461) m/s in the private forecast. After play, the exact native replay measured the actual contact radius drop from 75.6 to 52.8 mm. The ball landed as return 19 for the arm. The independent opponent physically replied and landed return 20, for 39 exchanges with no reset inside the point. The diagnostic intentionally stopped at that reply, so 19 is a fork endpoint and not the natural maximum of a new full controller. [Independent rim forks](impact-margin-center-independent.json.gz) · [Native radius replay](impact-margin-center-native.json)

This was the only intervention issued in the eight forks. Six other monitored strikes touched in the private court but had no replacement that was both materially closer to the center and nominally legal. The eighth did not touch in that court. Late recentering is therefore a selective correction, not a reliable way to recover every rim hit. [School rim forks](impact-margin-center-school.json.gz)

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt38_impact_margin artifacts/tt37/value-closed.json.gz artifacts/tt38/impact-margin-closed.json.gz
python -m scripts.experimental.tt38_impact_margin artifacts/tt37/independent-cold-value.json.gz artifacts/tt38/impact-margin-independent.json.gz
python -m scripts.experimental.tt38_impact_margin CONTROL SUMMARY --compare TESTED
python -m scripts.experimental.tt38_impact_margin_center school artifacts/tt38/impact-margin-center-school.json.gz
python -m scripts.experimental.tt38_impact_margin_center independent artifacts/tt38/impact-margin-center-independent.json.gz
python -m scripts.experimental.tt38_impact_margin_center audit AUDIT --tested FORK_ARCHIVE
```

`--stage stock --count 2` verifies that the benchmark driver reproduces the first two closed-loop control points exactly, including commanded rates and physical events. `--stage nominal` is an optional ablation that scores the same candidate family without uncertainty margins. [Impact planner and comparison audit](../../scripts/experimental/tt38_impact_margin.py)
