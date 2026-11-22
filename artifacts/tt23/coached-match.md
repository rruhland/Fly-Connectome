# Full games with the learned physical coach

Two of the three complete first-to-11 games included a rally beyond 20 legal returns. The continuously updating SO-101 reached **23** while serving at 0–2. The unpracticed arm won a **22**-return point on its next serve. All three games finished under the same strict win-by-two referee, and both motor tapes reproduced all 34 points exactly. [Frozen design](coached-match-design.md) · [Full games and replay checks](coached-match.json.gz)

The same learned five-axis coach played each game. It used its frozen continuation score and physical checks to rank replies. The SO-101 used its normal physical service, fixed stereo RGB, inverse flight fit and guarded factory stroke planner in all conditions. The two practiced arms started with seven optical observations from four earlier points. Online play added 33 observations, bringing its stored count to 40. Every serve was legal, and the games had zero lets or timeouts. [Learned coach](coach.md) · [Optical practice](../tt22/rally.md)

| Optical calibration | Final score | Point attempts | Near legal returns | Far legal returns | Longest | Points with 10 or more | Points with 20 or more |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unpracticed | 1–11 | 12 | 45 | 50 | 22 | 4 | 1 |
| Seven observations, frozen | 0–11 | 11 | 32 | 37 | 18 | 2 | 0 |
| Seven observations, online updates | 0–11 | 11 | 40 | 45 | 23 | 3 | 1 |

The first 11 attempts shared the launch seed, serving side and exact physical reset in all three branches. The unpracticed arm's extra near serve at 1–10 added one far return. On the 11 common starts, the near totals were 45, 32 and 40 for unpracticed, frozen and online, and the far totals were 49, 37 and 45. Online learning gained eight near returns over the frozen memory on this schedule. The highest shared near-return total came from the unpracticed arm. [Common resets and unmatched serve](coached-match.json.gz)

Each cell gives the near and far legal return counts, followed by the longest alternating run in parentheses.

| Attempt | Server | Unpracticed | Frozen | Online |
|---|---|---:|---:|---:|
| 1 | Far | 0–0 (0) | 0–0 (0) | 0–0 (0) |
| 2 | Far | 9–9 (18) | 8–8 (16) | 8–8 (16) |
| 3 | Near | 3–4 (7) | 3–4 (7) | 11–12 (23) |
| 4 | Near | 11–11 (22) | 2–3 (5) | 2–3 (5) |
| 5 | Far | 5–5 (10) | 2–2 (4) | 2–2 (4) |
| 6 | Far | 3–3 (6) | 3–3 (6) | 3–3 (6) |
| 7 | Near | 0–1 (1) | 0–1 (1) | 0–1 (1) |
| 8 | Near | 2–3 (5) | 3–4 (7) | 2–3 (5) |
| 9 | Far | 3–3 (6) | 2–2 (4) | 3–3 (6) |
| 10 | Far | 9–9 (18) | 9–9 (18) | 9–9 (18) |
| 11 | Near | 0–1 (1) | 0–1 (1) | 0–1 (1) |
| 12 | Near | 0–1 (1) | — | — |

The 23-return point occupied 491 control frames, or 16.37 simulated seconds. The 22-return point occupied 458 frames, or 15.27 simulated seconds. Replaying both arms reproduced all point lengths, strict referee events, score transitions and the serving schedule. Replaying the recorded optical updates also recovered the online memory at every point boundary. [Frozen seed and launch sequence](coached-match-plan.json.gz)

```sh
python -m scripts.experimental.tt23_coached_match
```
