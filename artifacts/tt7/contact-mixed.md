# Contact timing under changing racket response and transport

The all-soft timing memory won fewer points than the fixed aligned lift on both mixed receiving installations. It opened one additional point and landed four more legal returns across the 25 starts shared by all three groups. Three of its points at the second installation included at least two legal returns. [Complete games](contact-mixed-archived.json.gz)

| Player | Site 192827117 | Site 237048329 | Opening returns on common starts | Legal returns on common starts | Wins on common starts |
|---|---:|---:|---:|---:|---:|
| Fixed aligned lift | 6–11 | 5–11 | 13 | 15 | 8 |
| Carries all-soft practice | 2–11 | 3–11 | 14 | 19 | 4 |
| Starts with empty memory | 0–11 | 5–11 | 12 | 18 | 4 |

The all-soft memory chose to omit the lift in 14 of 20 timing decisions on the first installation. It landed nine legal returns through 13 starts and won two points. The aligned lift landed eight returns and won five points over those same starts. Every group had the same eight public RGB motor probes, inherited hall experience, and hidden schedule of racket responses and transport disturbances. Each shuffled four-point block contained one ordinary point, one point with a softened racket, one with a softened racket and command disturbances, and one with the stock racket and command delay. Games used a target of 11 and a two-point lead. All 87 recorded points reproduced the applied commands, events, and referee result on replay.

## Learning from matched lift practice

Two development installations played 12 starts each with three lift phases. The earlier lift opened 13 of 24 points and landed 19 legal returns. The aligned lift opened nine points and landed 12 legal returns. The later lift opened six points and landed six legal returns. They won four, two, and three points, respectively. [Matched practice and physical replays](contact-mixed-matched-practice.json.gz)

The camera fit supplied a departure estimate on 29 of the 72 practice openings. The other 43 included 12 legal openings. Those observations got lower weight and used the public score for the last timing attempt, with neutral credit for earlier attempts in a rally.

The paired learner compares the visual flight and public point scores from the first timing opportunity in each matched start. It weights the differences by how closely the new incoming ball and inferred delay match the practice observations. It starts from the aligned lift and gives another phase the stroke when the local score estimate exceeds that reference by a margin. It updates scores as it plays, using each new observation to adjust the estimate for that phase. The cold player uses the same rule with empty practice and explores any phase without recent feedback before following its live estimates.

The paired learner won more points than the aligned lift and the cold learner at both fresh installations. The earlier fixed lift had a one-point advantage across both sites and led the learner at the first site. The learner landed the most legal returns and had the most points with a second return on the 27 shared starts. [First installation](contact-mixed-paired-a.json.gz) · [Second installation](contact-mixed-paired-b.json.gz) · [Cold games](contact-mixed-paired-cold.json.gz)

| Player | Site 284016793 | Site 319603771 | Opening returns on common starts | Legal returns on common starts | Points with a second return | Wins on common starts |
|---|---:|---:|---:|---:|---:|---:|
| Paired timing learner | 5–11 | 3–11 | 15 | 21 | 4 | 8 |
| Fixed earlier lift | 7–11 | 2–11 | 16 | 19 | 3 | 9 |
| Fixed aligned lift | 4–11 | 2–11 | 13 | 17 | 3 | 6 |
| Starts with empty practice | 3–11 | 2–11 | 13 | 16 | 3 | 5 |

The learner chose the earlier lift on all 30 openings and used the aligned lift on eight later rally attempts. On the sixth start at site 319603771, it landed four legal returns and won the point. The earlier fixed lift landed one return and lost that start. The cold player in the original site receipts could skip an exploratory phase when a rally produced multiple attempts. The cold games in the table use the corrected rule, which checks for phase feedback. All 116 comparison points replayed against the applied commands, events, and referee results.

## Reproduce

Use the [TT6 environment](../tt6/README.md). The optional probe cache reuses the public camera images and replays the motor probes. The cached and uncached rotary fits and commissioning receipts matched in a separate check at development site 158747803.

```sh
python -m scripts.experimental.tt7_contact_trial /tmp/contact-mixed-archived.json.gz \
  --practice-seeds --test-seeds 192827117 237048329 \
  --resume artifacts/tt7/contact-player-lift.json.gz \
  --conditions fixed carried cold --race-to 11 --max-steps 420 \
  --phases ordinary soft mixed delay --probe-cache /tmp/contact-probes

python -m scripts.experimental.tt7_contact_trial /tmp/contact-mixed-practice.json.gz \
  --practice-seeds --test-seeds 158747803 126081551 \
  --conditions fixed fixed-early fixed-late --test-points 12 --max-steps 420 \
  --phases ordinary soft mixed delay --probe-cache /tmp/contact-probes

python -m scripts.experimental.tt7_contact_trial /tmp/contact-mixed-paired.json.gz \
  --practice-seeds --test-seeds 284016793 319603771 \
  --matched-practice /tmp/contact-mixed-practice.json.gz \
  --conditions carried fixed-early fixed cold --race-to 11 --max-steps 420 \
  --phases ordinary soft mixed delay --probe-cache /tmp/contact-probes
```
