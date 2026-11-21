# Follow-up after an own serve

The five-axis player often reached the reply to its own serve and sent the ball back short. Across four archived reply opportunities using the fixed velocity serve on offset arms, it hit three and returned none legally. The corresponding four gantry opportunities produced four hits and three legal returns. The archive combines one alternating game per body with held service trials. The small number of replies limits comparisons between bodies and between own-service replies and ordinary receiving. [Event audit](recovery-archive-audit.json)

The fixed player reached all seven legal replies to its own serve across two fresh games. Three became legal returns. The same controls returned 13 of 17 legal rival serves and six of eight next legal rival replies on rival-service points. Six of the seven own-service follow-up contacts had an RGB ball sighting one to three frames before impact. The seventh had a seven-frame gap and still produced a legal return. On the new gantry all three failed first follow-ups bounced on the player's side. On the five-axis arm the failed contact sent the ball back toward the player's end. Some earlier gantry traces lost the ball for six or more frames around the near-side bounce. This did not consistently predict a lost or illegal follow-up. [Offset game and RGB traces](recovery-match-offset.json.gz) · [Gantry game and RGB traces](recovery-match-gantry.json.gz)

## Two complete alternating games per installation

Both players started from the same carried [checkpoint](../tt8/span-after-four.json.gz) and used the same frozen 0.132 m/frame velocity serve. The experimental player recognized an own toss from stereo and looked for the outbound ball to reverse direction on the far half. It applied the existing early upward stroke during that rally, shifted by the estimated actuator delay. It could extrapolate up to four additional frames when the approaching ball disappeared near a planned collision. Only the evaluator read the server, physical contacts and serve legality. The players continued to update inherited visual and public-reward estimates during play.

| Installation | Controller | Score | Legal own serves | First legal return after own serve when a reply landed | Points with sustained exchange |
| --- | --- | ---: | ---: | ---: | ---: |
| Five-axis offset 315700676 | Fixed | 8–11 | 4/12 | 1/2 | 4 |
| Five-axis offset 315700676 | Early lift and additional coast | 3–11 | 5/8 | 2/4 | 4 |
| Gantry 1647285554 | Fixed | 6–11 | 6/9 | 2/5 | 5 |
| Gantry 1647285554 | Early lift and additional coast | 3–11 | 5/8 | 2/4 | 5 |

A sustained exchange required a legal learner return after an own serve or at least two legal learner returns on an opposing-service point. Both games on each installation used the same ordered seed and disturbance schedule. The offset control had two service lets. Points of different duration and reward changed the carried estimates and later serve actions. The same seed, server and condition occurred on 8 starts on offset and 14 on gantry. Across 13 shared own-service starts, four action prefixes matched through the first rival contact or the point end. All 66 point tapes replayed with identical applied commands, events and scores. [Offset](recovery-match-offset.json.gz) · [Gantry](recovery-match-gantry.json.gz)

On rival-service points, the experimental player never activated the follow-up intervention.

The development comparison tested extra forward and upward blade velocity, higher pitch and longer extrapolation after observed service replies. On three archived gantry starts the current fixed drive held two answers legally. The stronger forward velocity objective and that objective with higher pitch each held one. Lift velocity alone, higher pitch alone, and longer extrapolation with either pitch setting held none. The fresh complete games tested the earlier contact-phase lift from the prior six-joint experiment together with the longer extrapolation. The complete games support keeping the fixed follow-up controller. [Development trials](recovery-development-gantry.json.gz) · [Earlier contact-phase experiment](../tt7/contact-timing.md)

## Replay with identical prior point history

Each of the seven fixed-game points with a legal reply was replayed with both controllers starting from the fixed player's state immediately before that point. The baseline reproduced its actions from both full games. Every paired service had identical actions through the opponent's contact. The overlay produced four legal first follow-ups versus three. Both offset trials kept their original first-follow-up outcome. On gantry one short shot became a legal return, while two illegal contacts became misses. Across the seven points, total legal returns fell from ten to eight and the one fixed-player win became a loss. [Paired offset points](recovery-paired-offset.json.gz) · [Paired gantry points](recovery-paired-gantry.json.gz)

## Reproduce

Use the [TT6 environment instructions](../tt6/README.md). Both runs record the player's reconstructed ball and blade tracks along with the separate referee event tape.

```sh
python -m scripts.experimental.tt9_recovery_trial /tmp/recovery-offset.json.gz \
  --seed 641172389 --policies fixed early-lift --probe-cache /tmp/tt8-cache --trace
python -m scripts.experimental.tt9_recovery_trial /tmp/recovery-gantry.json.gz \
  --seed 462190827 --body gantry --policies fixed early-lift --probe-cache /tmp/tt8-cache --trace
python -m scripts.experimental.tt9_recovery_trial /tmp/recovery-development-gantry.json.gz \
  --seed 128753901 --development artifacts/tt8/service-drive-gantry.json.gz --section fixed_drive \
  --policies fixed velocity lift both loft coast coast-loft --probe-cache /tmp/tt8-cache --trace
python -m scripts.experimental.tt9_recovery_audit /tmp/recovery-offset.json.gz /tmp/recovery-gantry.json.gz
python -m scripts.experimental.tt9_recovery_paired /tmp/recovery-offset.json.gz /tmp/recovery-paired-offset.json.gz \
  --probe-cache /tmp/tt8-cache
python -m scripts.experimental.tt9_recovery_paired /tmp/recovery-gantry.json.gz /tmp/recovery-paired-gantry.json.gz \
  --probe-cache /tmp/tt8-cache
```
