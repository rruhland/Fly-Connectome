# Recovering a slower actuator from visual motion

The controller compares six actuator rise times and nine request delays using the same camera-position innovations that identify button effects. Each combination retains its own velocity and actuator-state estimate. The controller interpolates rise time from the local evidence and carries that estimate across points. With an inferred slower response, it aims at the same visual crossing while accounting for residual paddle travel. With the acquired fast response selected, it uses the existing arrival controller. [Controller](../../scripts/experimental/pong_actuator_response.py)

The development comparison reused 12 slower starts from the initial inertia scout. The response learner returned 46/48 opportunities and saved 11/12 first balls. Its matched response-time-frozen copy returned 9/20 and saved 5/12 first balls. The frozen copy still learned request delay and button effects. Its request sequence matched the original timing-only learner on all 18 slower and restored games. [Development receipts](/workspace/m1b-experiments/pong-actuator-response-dev-20261119-v1)

## Fresh paired native games using timing v2

The simulator changed actuator rise time from 0.08 to 0.32 seconds. The change applied from the opening of each slower point. Both arms used the same 64×64 visual tracker, goal controller, and timing updates. The learner received public visual entities, elapsed camera time, and its request history. Six fresh restored points immediately followed the slowdown with each arm carrying its own learned state. Separate normal and FIFO phases started from the acquired fast model. Points ended at a score or the ten-second limit.

| Condition | Live first | Response frozen first | Live later | Response frozen later |
| --- | ---: | ---: | ---: | ---: |
| Slower response | 10/12 | 5/12 | 20/20 | 3/6 |
| Restored response | 6/6 | 6/6 | 12/12 | 12/12 |
| Ordinary controls | 6/6 | 6/6 | 10/10 | 10/10 |
| FIFO 12 cameras | 3/3 | 3/3 | 6/6 | 6/6 |
| FIFO 17 cameras | 2/3 | 2/3 | 4/4 | 2/3 |
| FIFO 21 cameras | 2/3 | 2/3 | 4/4 | 4/4 |

On the slower starts, the live arm gained five first returns with zero paired losses. It inferred zero queue delay on all 12 and finished with response estimates of 0.297–0.344 seconds. Its two first misses occurred on direct opening serves. The evaluator found that full legal drive from the first tick still left gaps of 0.158 and 0.162 image units at those contacts, exceeding the 0.1125 hit tolerance. The two arms used identical requests in all six ordinary-control games. The 17-camera FIFO fell between the supported lag values. The live arm represented it as 15 cameras plus a slower response and the frozen arm used 18 cameras. This exposes ambiguity in identifying an off-grid delay and response time from the same movement. [Fresh paired receipts](/workspace/m1b-experiments/pong-actuator-response-paired-20261119-r1)

The fresh run used 26,719 transition cameras and 66 initial cameras in 78.26 seconds. The fitted rise-time estimate and delay were restored jointly from checkpoints. The evaluation covers changes across points and separate FIFO phases. Abrupt changes to physical response within a point and simultaneous slowdown plus FIFO remain open. [Runner](../../scripts/experimental/run_pong_actuator_response.py)

## Timing v3 restore smoke test

Timing v3 reconsiders a carried delay when the paddle moves before its first relevant queued command could arrive. In the opt-in response observer, that evidence applies to the rejected lag at every rise time. The controller selects lag zero with the current rise-time class and clears any pending joint switch.

The response observer and scalar timing v3 issued identical requests on nine fresh games covering ordinary controls, a retained 12-camera FIFO, and removal of a carried 12- or 21-camera FIFO. Both returned all 21 opportunities. In the three restored games, both discarded the old delay 0.20 seconds after the serve and seven cameras after the first movement request. Neither arm discarded the true retained FIFO.

On four further starts with a 0.32-second actuator and no scoring-based placement, the live response learner and its response-frozen counterpart each saved 3/4 first balls. The live arm saved 4/4 later balls versus 0/3 frozen and inferred zero delay throughout the four end-of-point checkpoints. This smoke check used 9,536 transition cameras and 26 initial cameras. The earlier v2 paired receipts retain their original source hashes and conditions. [Smoke runner](../../scripts/experimental/check_pong_actuator_response_restore.py) · [V3 receipts](/workspace/m1b-experiments/pong-actuator-response-v3-smoke-20261119-r1)
