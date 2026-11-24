# Why the arm held still on some incoming balls

In four continuously varied points, the SO-101 saw the incoming ball, attempted a stroke, and held its last joint targets when the planner could find no valid motion. Instrumented runs reproduced the archived actions, contacts and measured placements. The first search began 77–83 ms after the physical far paddle hit. Across the four points the planner tried 29 times and accepted no tape. The arm's largest measured joint excursion was under 0.001 rad per point. On active control 13159 it issued commands over 1% of nominal rate on 131 control ticks and landed three returns. Its final incoming ball also went unanswered after six failed searches. [Decision and encoder traces](idle-trace.json.gz) · [Original eight points](../tt31/recovery-full.json.gz)

| Measured opening | Actual first arrival x at the receiving plane | First search after far impact | Searches over point | Largest generated batch |
| --- | ---: | ---: | ---: | ---: |
| 10898 | −0.002 m | 83 ms | 8 | 315 |
| 3286 | +0.187 m | 77 ms | 7 | 840 |
| 4869 | +0.060 m | 77 ms | 7 | 45 |
| 8557 | +0.150 m | 83 ms | 7 | 120 |

The planner first screened remembered motor tapes, then fitted fresh strokes. Before the near bounce it generated candidate motions on all four points, and the private dynamics checks rejected all of them. After the camera recognized the bounce, the available fits all failed the geometric, face-alignment or joint-travel checks. The trace shows both bottlenecks. The pre-bounce simulated rejection does not identify a single cause such as a collision, speed limit or predicted bad landing. [Full batch and fit counts](idle-trace.json.gz)

## Expanding the contact window

The fallback keeps the original memory, camera tracker, motor limits, solver and ready pose. It runs only when the stock search returns no tape. It proposes contacts at six additional depths between 1.10 m and 1.60 m from the net, plus one lower contact when the forecast supports it. It keeps the same local inverse kinematics and dynamically verifies the complete motor tape before acting. Its inputs are the successive camera and encoder observations and the near arm's own joint targets. [Implementation](../../scripts/experimental/tt32_idle.py)

| Measured opening | Original arm returns | Wider window arm returns | Original alternating returns | Wider window alternating returns |
| --- | ---: | ---: | ---: | ---: |
| 10898 | 0 | 0 | 1 | 1 |
| 3286 | 0 | 0 | 1 | 1 |
| 3855 | 1 | 1 | 3 | 3 |
| 14183 | 1 | 1 | 3 | 3 |
| 4869 | 0 | 0 | 1 | 1 |
| 13159 | 3 | 3 | 7 | 7 |
| 8557 | 0 | 2 | 1 | 5 |
| 13621 | 1 | 1 | 3 | 3 |
| **Total** | **6** | **8** | **20** | **24** |

The original and wider-window controllers issued identical command tapes on six points. On 8557, the first incoming flight matched and the wider search landed from 1.52 m from the net, beyond the stock search's deepest proposed interception at 1.38 m. It hit 15.1 cm right of the shoulder, then landed the next reply with the original search 21.4 cm left of the shoulder. Both used the same blade face. On 3286 the wider search issued 52 meaningful command ticks and missed the paddle. The eight wider-window replays checked 12,365 physics states with zero CAD penetrations and a minimum gap of 15.9 mm. The wider trial used 140 CPU seconds and the original used 122. [Eight paired points](idle-wider.json.gz) · [8557 contact and CAD replay](idle-8557-audit.json)

## Fresh launches and opponent seeds

On eight independently selected launches, the original controller landed four arm returns and the wider controller landed six. The alternating scores were 16 and 20. Both controllers held the same learning checkpoints fixed. All eight first deliveries matched through the far paddle contact, near bounce and receiving-plane crossing. Actions matched exactly on six points. The arm had issued zero movement commands on both points recovered by the wider search. [Frozen plan](fresh-plan.json.gz) · [Original controller](fresh-fixed.json.gz) · [Wider-window commands and exact CAD replays](idle-fresh-wider.json.gz)

| Measured opening | Original arm returns | Wider window arm returns | New physical contact |
| --- | ---: | ---: | --- |
| 8886 | 0 | 1 | 1.534 m from net, 10.9 cm right of shoulder |
| 14993 | 0 | 0 | |
| 3026 | 0 | 0 | |
| 8239 | 2 | 2 | |
| 13131 | 0 | 0 | |
| 13213 | 0 | 1 | 1.478 m from net, 23.2 cm right of shoulder |
| 9532 | 1 | 1 | |
| 10485 | 1 | 1 | |
| **Total** | **4** | **6** | |

The eight fresh wider-window replays checked 11,254 physics states with zero CAD penetrations. The closest checked gap was 15.7 mm. The fresh original runs used 134 process CPU seconds and the wider runs used 161. Across both eight-point courses the wider search added four arm returns in three games. All three points that started with a stationary miss in the middle still missed. [Earlier wider-window tapes](idle-wider.json.gz) · [Fresh wider-window tapes](idle-fresh-wider.json.gz)

## Reproduction

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt32_idle /tmp/tt25-rallies.json /tmp/idle-trace.json.gz
python -m scripts.experimental.tt32_idle /tmp/tt25-rallies.json /tmp/idle-wider.json.gz \
  --widen --ids 10898 3286 3855 14183 4869 13159 8557 13621
python -m scripts.experimental.tt32_idle /tmp/tt25-rallies.json /tmp/idle-fresh-wider.json.gz \
  --reference artifacts/tt32/fresh-fixed.json.gz \
  --widen --ids 8886 14993 3026 8239 13131 13213 9532 10485
```
