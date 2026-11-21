# A complete game with the optical rally receiver

The stock SO-101 played one first-to-11, win-by-two game on preselected seed `403607823`. Service rotated every two scored points and every point after deuce. The match used the same gentle standard rival releases, stock clamped body, strict referee and physical anchor partner as the rally experiments. The partner cooperatively targeted playable returns. [Match program](../../scripts/experimental/tt22_match.py)

On pupil service, its physical paddle executed the previously qualified serving program 3. The left camera measured the tossed ball at the established three times. On points still live at frame 35, the native stereo rally receiver took over with its full optical history and an independent arm model driven by every earlier joint-rate command. The same physical point continued through the handover. The pupil kept its visual landing and clearance correction across all points, beginning with the four-point practice checkpoint and updating from visible outgoing arcs. [Service qualification](../tt21/serving.md) · [Rally checkpoint](rally-learning-checkpoint.json.gz)

Both arms’ actual rate commands and each referee event are recorded. The physical replay uses both command tapes, the original seeded releases and the original referee. The optimizer’s simulated paths are separate from the executed point. [Match receipt](match-403607823.json.gz)

## Result

The rival won **11–2** over 13 points. The pupil served legally on 6/6 attempts and the rival on 7/7. The pupil made 15 legal returns and the rival made 19. The two pupil wins followed its own legal serves.

The consecutive legal-return counts, with the serve excluded and every point retained, were **8, 6, 0, 1, 4, 4, 1, 0, 0, 4, 1, 5, 0**. The longest point contained four legal returns by each side. The pupil’s visual correction grew from seven carried observations to 21 by the end of the game. All 13 points and the final scoreboard reproduced exactly from 1,261 recorded frames of both physical arms’ commands. [Match receipt](match-403607823.json.gz)

## Reproduction

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_match
python -m scripts.experimental.tt22_match --verify artifacts/tt22/match-403607823.json.gz
```
