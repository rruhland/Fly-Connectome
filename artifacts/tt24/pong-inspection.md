# Learning when to inspect a changed actuator from native Pong

Four independent native Pong reward fits selected inspection thresholds of 3.3, 2.3, 3.3 and 2.3 permitted per-frame excursions. The four within-launch reward shuffles selected 1.3, 0.20, 0.35 and 0.35. On 200 fresh points, the reward-selected thresholds raised net score by 68 compared with shuffled rewards, by 45 compared with the arm monitor's original equivalent threshold, and by 35 compared with never inspecting. The gains came from avoiding losses before the 12-second cap. All conditions won three points. [Registered courses and method](pong-inspection-plan.json) · [Held native point records](pong-inspection-held.json.gz)

During play, the observer predicts the paddle coordinate from its own commands and current sign estimate. It divides the absolute optical position residual by the permitted travel per camera frame. A residual above the threshold adds one unit of evidence. Other frames subtract half a unit with a floor of zero. Four accumulated units trigger a 15-frame identification sequence. The actual native ball and opponent continue for the entire half-second sequence. The paddle receives generic signed pulses during this time and estimates the new actuator direction from the resulting RGB motion. The paddle tracker and sign estimator receive no switch flags. Hidden courses include a stable sign, lasting and transient reversals, and unexpected restoration. [Source runner](../../scripts/experimental/tt24_pong_inspection.py)

The optical nominal motor fit used 2,260 frames of unscored exploration and 904 separate test frames. Its held tracking error was 0.245 times the permitted per-frame excursion. The pilot correctly identified the sign in 92.4% of completed probes with constant wiring. On held play, the reward-selected model recovered the correct sign in 215 of 235 completed probes with constant sign, including inspections where the sign was already correct. Once the optical model was fixed, the four threshold fits used 2,400 real native point trials, ten thresholds and only terminal point scores. The learner received native resets, visual frames and the outcome. True command gains and sign schedules stayed outside the learner. [Pilot and calibration](pong-inspection-pilot.json.gz) · [Fit and test data](pong-inspection-held.json.gz)

| Inspection policy | Net point score | Losses | Timeouts | Checks | Alarms on 40 stable points | Live time spent inspecting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pong reward fit | −93 | 96 | 101 | 290 | 16 | 8.9% |
| Shuffled reward fit | −161 | 164 | 33 | 787 | 158 | 37.0% |
| Original arm equivalent, 0.785 | −138 | 141 | 56 | 630 | 109 | 24.2% |
| Fixed 1.0 | −133 | 136 | 61 | 528 | 72 | 19.4% |
| Fixed 0.5 | −154 | 157 | 40 | 818 | 150 | 35.3% |
| Never | −128 | 131 | 66 | 0 | 0 | 0.0% |
| Periodic | −170 | 173 | 24 | 654 | 156 | 36.3% |
| Always | −191 | 194 | 3 | 1,309 | 240 | 100% |

The paired per-point improvement was +0.340 over shuffled rewards with a 95% bootstrap interval of +0.265 to +0.415, +0.225 over the original arm equivalent with an interval of +0.155 to +0.295, and +0.175 over never checking with an interval of +0.110 to +0.240. The reward-selected threshold still prompted ten of the 40 stable points to inspect at least once. [Paired outcomes and per-inspection event records](pong-inspection-held.json.gz)

Fixed high thresholds were registered after the first fit reached the upper grid boundary. The follow-up replayed the same held launches without selecting another setting. Fixed 1.3 scored −108 with 471 checks, fixed 2.3 scored −92 with 300, and fixed 3.3 scored −95 with 267. The reward-selected threshold averaged −0.005 points against fixed 2.3, with a paired interval of −0.040 to +0.030, and +0.010 against fixed 3.3, with an interval of −0.025 to +0.045. The repeated reward fits recovered the high-tolerance region. Both fixed thresholds within that region reproduced the reward-selected performance. [Follow-up plan](pong-inspection-high-plan.json) · [Matched fixed-threshold trials](pong-inspection-high-held.json.gz)

The dimensionless scale supplies a direct comparison to the five-axis arm monitor. Pong divides optical position error by 0.0233 court units. The arm divides RMS joint error by its allowed 5.2/60 radians per tick. The original 0.068-radian arm threshold corresponds to 0.785 on this scale. The two selected source thresholds correspond to 0.199 and 0.286 radians on the arm. The source trials evaluate when to inspect. Their sign estimator and paddle policy both remain local to native Pong. [Source plan and units](pong-inspection-plan.json) · [Arm monitor](../../scripts/experimental/tt24_autonomous_drive.py)
