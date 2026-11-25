# Swap the earned bank on the same terminal incoming ball

The whole-game transfer score fell from 196 to 130 after the 63-return practice point. In the 18 held pairs whose issued commands changed, legality on the first changed same-incoming ball did not change. This study instead swaps banks on the hard balls that actually ended those games. It asks whether one bank can return the other bank's terminal miss when the previous rally, native incoming ball, and receiving body approach are all preserved. [Original paired transfer](../tt51_point_transfer/transfer.json)

The comparison is the first eight pairs by the original preassigned order, both directions. Seven assignments with an incoming terminal ball contribute 14 candidate swaps. The remaining assignment has no native opponent offer in either arm and stays in the ledger as two verified no-offer replays. No later case replaces it. Each ordinary actor runs closed loop from launch with the originating bank and full RGB and encoder history. The harness verifies every issued command and native event in the prefix. Immediately after the last far-racket separation, it substitutes only the 768 online motor fragments from the opposite exact assessment boundary, `after-09` or `after-10`. It does not reset or replace the actor, incumbent, optical state, encoder history, random generator, or native opponent.

The harness knows the terminal cut from the archived event tape. That retrospective timing is not an actor feature or a proposed deployment trigger. The native ball and body state at the cut are recorded separately for diagnosis, not given to the controller. A rescue counts only on the original native far impact, same requested placement and identical near bounce. Any rescue must replay the factual arm without the swap and reach a true new legal landing. Both arms continue to the normal native terminal disposition. [Frozen roster and source hashes](protocol.json)

## The late bank swap did not rescue any of the 14 matched misses

All 14 terminal swaps reproduced the original far impact, separation, near bounce and measured receive line exactly. The earlier bank rescued **0 of 7** of the later bank's terminal misses, and the later bank rescued **0 of 7** in the reverse direction. Six swaps changed issued motion without changing whether the arm contacted the ball or scored. The two no-offer cells exactly reproduced their factual command and event tapes. The summary includes native terminal reasons, and the action archive preserves accepted joint goals and the actual native commands. [Paired results](comparison.json) · [Commands and native events](actions.json.gz)

| Release | Whole-game legal returns before → after | Earlier bank changed commands on later miss | Later bank changed commands on earlier miss |
| --- | ---: | --- | --- |
| 4888 | 0 → 0 | Yes | Yes |
| 5335 | 1 → 3 | Yes | No |
| 8883 | 6 → 6 | No | Yes |
| 7398 | 22 → 18 | No | Yes |
| 7738 | 0 → 0 | No offer | No offer |
| 5222 | 23 → 3 | No | No |
| 15035 | 0 → 0 | No | No |
| 12265 | 7 → 10 | No | Yes |

The factual games had different terminal incoming balls in five of the seven eligible pairs, even though the near arm had essentially reset by far separation. The commanded references were exactly equal at the two factual cuts in all seven pairs, and the measured blade-center positions were at most 0.061 mm apart. The differing terminal receive lines varied by up to 78.3 mm in height and 1.20 m/s in velocity. On 5222, the largest sampled decline, the terminal ball moved from x = +0.134 m to −0.089 m at the receive line. Neither opposite bank changed a single command on either exact terminal ball. [Separate evaluator-only cut diagnostic](hidden-cut.json)

This argues against a late retrieval failure as the explanation for these terminal misses. It does not rule out bank effects earlier in the rally. An earlier bank can select an already legal but physically different outgoing stroke, which gives the native opponent a different ball and can change both the subsequent offer and how the arm approaches it. A same-history fork at the first changed outgoing choice, followed through the very next native offer, would separate that pathway from the capabilities of the two banks on a common incoming ball.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt52_terminal_bank_crossplay run
python -m scripts.experimental.tt52_terminal_bank_crossplay report
python -m scripts.experimental.tt52_terminal_cut_report
python -m pytest -q tests/test_tt52_terminal_bank_crossplay.py
```
