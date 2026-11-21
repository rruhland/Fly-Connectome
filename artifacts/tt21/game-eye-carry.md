# Carrying game-learned actuator delay between cameras

The source is the first ordinary delayed receiving point in each single-eye source game. The nominal and adaptive source runs saw identical RGB, actions, applied commands and referee events through point 5 of game seed `1920024739`. Four original points preceded the change. The outside camera identified a four-frame delay after 44 ordinary control frames on the changed point. The wrist camera identified it after 34. The estimate was made at the point boundary, using the issued actions and RGB motion trace with a separately rendered nominal arm. No dedicated motor action or target calibration point was used. [Source games](game-delay.md) · [Source receipt](game-delay.json.gz)

Both destination runs use the same strict unmarked clamp, same factory policy, a fresh single-eye registration at each start, and the unchanged four-frame command FIFO. The source eye is absent during destination play. The carried condition installs the estimated lead before the first destination point. The discarded condition keeps the original zero lead after the same source exposure. The only transferred value is the integer delay. Both directions use 12 matched fresh receiving starts drawn before evaluation, disjoint from the full ordinary source games and from each other. No updates, fitting, warm-up actions or nominal renders occur in the destination. [Evaluation code](../../scripts/experimental/tt21_game_eye_carry.py)

| Source eye | Ordinary control frames through recognition | Ordinary frames since change | Nominal renders on changed point | Changed-point inference wall time |
| --- | ---: | ---: | ---: | ---: |
| Outside | 243 | 44 | 28 | 0.64 s |
| Wrist | 230 | 34 | 20 | 0.41 s |

Four frames correspond to 133 ms of simulated transport. The nominal renders and matching through all five source points cost 4.71 seconds for the outside camera and 5.07 seconds for the wrist camera. The ordinary RGB captures already served gameplay and required no additional source controls.

| Transfer direction | Matched receives | Point score, discarded → carried | Legal returns, discarded → carried | Learner paddle contacts, discarded → carried |
| --- | ---: | ---: | ---: | ---: |
| Wrist → outside | 12 | 1–11 → 1–11 | 1 → 2 | 9 → 7 |
| Outside → wrist | 12 | 0–12 → 0–12 | 0 → 3 | 6 → 9 |

The carried lead changed the action tape on all 24 starts and increased legal returns in both directions. Aggregate point wins held steady. On the outside camera, carry added a winning legal return on seed `1356800682` and lost seed `600021732` after a legal return that both versions made. The three legal returns added on the wrist camera all ended in point losses. Each aggregate score covers 12 receiving points under the fixed delay.

Independent arenas reproduced the physical outcome, all referee events, and the clipped FIFO action tape on all 48 points. Rendering the first matched pair per camera again reproduced all 142 original native policy RGB frames. The source and target receipts preserve the player requests, applied commands, observation histories and hashes. [Matched replay receipt](game-eye-carry.json.gz)

Carrying the estimate costs one integer at the destination. End-to-end target processing, including native rendering, tracking and stroke planning, averaged 95–120 ms per simulated 33 ms tick across the four branches. Full target control-frame counts were `468 → 511` on the outside eye and `464 → 567` on the wrist. Independent verification, including the RGB sample, took a further 5.23 seconds and was outside the player clock.

```bash
python -m scripts.experimental.tt21_game_eye_carry --count 12 --output /tmp/tt21-game-eye-carry.json.gz
python -m scripts.experimental.tt21_game_eye_carry --verify
```
