# A far-court practice carriage for the table-edge SO-101

The far carriage landed 41 of 43 legal incoming balls from 48 previously untested match launches. The complete native contact and referee event sequence reproduced on all 48 replays. The far replies reached the intended near depth at a median longitudinal speed of 4.17 m/s. They passed below the proposed high receiving corridor. [Event and launch records](partner-validation.json.gz)

| Incoming launch group | Tested | Legally reached far side | Far blade contacts | Legal far replies |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 8 | 8 | 8 | 8 |
| Fast | 8 | 6 | 6 | 6 |
| Oblique | 8 | 8 | 7 | 7 |
| Strong spin | 8 | 5 | 5 | 5 |
| High launch | 8 | 8 | 8 | 8 |
| Slow | 8 | 8 | 8 | 7 |
| Total | **48** | **43** | **42** | **41** |

The audit rotates each recorded position, velocity and spin 180° around the vertical axis at launch. A stationary SO-101 stays on the near side of the regulation table. Five launches fail before reaching the far side legally, including two stopped by the near blade. The far player misses one oblique ball that bounced legally, and one slow-ball reply lands on the far side. All 41 legal replies clear the net without contact and pass their assigned near depth, 1.04–1.09 m behind the net. [Validation implementation](../../scripts/experimental/tt29_partner_validation.py) · [Raw event data](partner-validation.json.gz)

| Measurement for legal far replies | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Speed at far blade separation (m/s) | 5.19 | 6.55 | 7.57 |
| Far blade separation to near bounce (s) | .198 | .354 | .560 |
| Near bounce depth from net (m) | .291 | .440 | .559 |
| Longitudinal speed at assigned near depth (m/s) | 2.83 | 4.17 | 4.70 |
| Ball center height at assigned near depth (m above floor) | .920 | .992 | 1.103 |
| World x-axis spin at far blade separation (rad/s) | −131.2 | −44.7 | 41.5 |

The physical ball stays in the native MuJoCo solver after the opening launch. The carriage carries the original rigid far paddle on a simulated free joint with 0.65 kg effective mass. Its private planner reads simulator state, tests impacts and flights on independent solver states, and applies bounded force and torque to the carriage in the live arena. Rail geometry represents the support visually. The controller limits target speeds to 3.45 m/s and 10 rad/s. Measured carriage speed peaked at 3.23 m/s, while impact impulses produced angular speeds up to 12.89 rad/s. The simulated carriage and its state access make it an idealized training opponent. [Opponent implementation](../../scripts/experimental/tt29_partner.py)

The near arm held a fixed command in this audit. The results establish the far opponent's ability to return varied shots over the regulation table. The high corridor required |x| ≤ .13 m, z from 1.095 to 1.29 m, and nearward longitudinal speed from 3.35 to 5.5 m/s at the assigned depth. Zero replies met all three requirements. Spin on separation was predominantly backspin, despite the planner's preference for topspin. [Per-launch diagnostics](partner-validation.json.gz)

## Deeper receiving targets

The same 48 launches were tested twice with receiving targets 0.27 m deeper, 7.5 cm lower and 0.45 m/s slower. This second configuration also landed 41 of the 43 legal incoming balls and cleared the net on every legal reply. Twenty-six replies met the deeper geometric corridor of |x| ≤ .13 m, z from 1.015 to 1.19 m, and nearward longitudinal speed from 3.0 to 4.85 m/s at the assigned depth. [Deeper launch and replay records](partner-deep-validation.json.gz)

| Measurement for legal far replies with deeper targets | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Speed at far blade separation (m/s) | 5.14 | 6.07 | 7.53 |
| Far blade separation to near bounce (s) | .175 | .456 | .629 |
| Near bounce depth from net (m) | .333 | .629 | .843 |
| Longitudinal speed at assigned near depth (m/s) | 2.69 | 3.52 | 4.68 |
| Ball center height at assigned near depth (m above floor) | .980 | 1.055 | 1.126 |

An optional clearance check screens the sampled carriage approach and follow-through before choosing a stroke. It keeps the planned blade's rim above the table whenever its center is over the playing surface. The centerline audits use the original planner. [Carriage trajectory checks](../../scripts/experimental/tt29_partner.py)

The forehand preset shifts the receiving window to the right of the near arm's base. An optional approach check simulates the commanded motion from before first contact through blade separation, then ranks the resulting flight. Two guarded forehand development launches reproduced legal replies with the check enabled. On launch 6971 the reply rose from 1.017 m to 1.047 m at y = −1.325 m. Launch 5906 used the same reply in both modes. The two-launch comparison measures the different selected returns. [Opposing trajectory and approach code](../../scripts/experimental/tt29_partner.py)
