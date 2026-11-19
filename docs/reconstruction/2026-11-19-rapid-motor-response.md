# Rapid visual response recovery in native Pong

The controller retains the frozen relational steering intent. A bank of six request orderings predicts the controlled entity’s image position from past requests and measured camera elapsed time. Each model tracks position, velocity, and a lagged actuator response. The likelihood of each observed position shifts probability between the orderings, allowing a new ordering to inherit the motion already in progress. When evidence rules out the previous ordering, the controller can pick a useful request using the probabilities for the desired effect before all three requests are resolved.

The response template was fit on the observed positions and actuator requests from 12 ordinary native starts. The other four starts were held back from fitting. The normalized position RMS error was 0.00370 on the fitting starts and 0.00353 on the held-back starts. Calibration used 4,800 transition cameras and 16 initial cameras. The fitted time constants were 0.0765 and 0.2407 seconds. The existing frozen visual tracker, encoder, and relational policy supplied the controlled entity. [Calibration runner](../../scripts/experimental/fit_pong_rapid_motor_response.py) · [Calibration receipts](/workspace/m1b-experiments/pong-rapid-motor-calibration-20261119-r1)

## Development comparison

The native evaluator changes the execution mapping during an existing rally. The controller receives the existing public entities, its own request, and camera elapsed time. The evaluator alone knows the switch time, mapping, and exact ball-contact tick. The controller was frozen at `7ff273976d8f25d109f33a716cf455dc49ea1ff9` for independent qualification.

| Development condition | Controller | Next return | Later returns | Full ordering by next contact | Full ordering by game end |
| --- | --- | ---: | ---: | ---: | ---: |
| Reused four-second cycle starts | Rapid | 21/24 | 76/76 | 16/24 | 23/24 |
| Reused four-second cycle starts | Shorter probes | 10/24 | 22/26 | 3/24 | 6/24 |
| Five permutations at 3–7 seconds | Rapid | 39/40 | 127/127 | 32/40 | 40/40 |
| Five permutations at 3–7 seconds | Shorter probes | 15/40 | 7/21 | 2/40 | 3/40 |

Every pair in both comparisons had identical public prefixes and identical first changed-contact clocks. The rapid controller saved 11 additional returns in the reused comparison and 24 in the broader comparison. Median time to correct ordering in successful rapid identifications was 0.433 and 0.467 seconds, respectively. The broader comparison assigned one switch time per permutation. The timings and mappings were varied together, with independent held-out evaluation able to test other combinations. Longer survival produced more later opportunities in the rapid arm. This model treats the changes as permutations of three familiar effects. A new effect magnitude or observation delay requires separate evaluation. [Development receipts](/workspace/m1b-experiments/pong-rapid-motor-development-20261119-r1)

## Carrying the response between points

`RapidMotorMemory` serializes all six probabilities and the currently confirmed ordering. A new controller restores those values and initializes its physical motion state from the first bound visual position in the next point. A frozen copy retains the same probabilities while visual steering continues.

A single native camera session ran three capped points over 615 transition cameras. The first point ended 0.3 seconds after a hidden cycle, while the two leading orderings had probabilities 0.518 and 0.277. The next point began with those exact probabilities and confirmed the cycle from continued visual evidence. The final point restored normal controls, and the controller confirmed the normal ordering again. A live controller and a controller recreated from JSON agreed on all requests, state estimates, and probabilities at every camera. The frozen copy preserved its restored probabilities throughout. An earlier diagnostic at 0.2 seconds left 0.966 probability on the original map, so the carry check used 0.3 seconds to exercise partial identification. [Memory implementation](../../scripts/experimental/pong_rapid_motor_memory.py) · [Carry check](../../scripts/experimental/check_pong_rapid_motor_carry.py) · [Native carry receipts](/workspace/m1b-experiments/pong-rapid-motor-carry-20261119-r2)
