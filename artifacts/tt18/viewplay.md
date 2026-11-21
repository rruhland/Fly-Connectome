# Scored table play through camera changes

On a held game schedule, fitting the active cameras from RGB brought the score to 9–11 with ten legal returns. The player with its initial camera fit frozen lost 1–11 and made no legal returns after the first camera change. Keeping the original cameras throughout produced a 9–11 loss with eleven legal returns. Each adaptive game had two points with multiple legal returns.

| Camera condition | Player–rival score | Legal returns | Points with ≥2 returns | Control frames | Replayed points |
| --- | --- | --- | --- | --- | --- |
| Stable cameras, RGB registration | 9–11 | 11 | 2 | 1,555 | 20 |
| Changed cameras, RGB registration | 9–11 | 10 | 2 | 1,357 | 20 |
| Changed cameras, frozen registration | 1–11 | 4 | 1 | 619 | 12 |

Both changed-camera branches received the same stream schedule and initial RGB calibration opportunity. The stable branch kept its original two streams. The first two point attempts used identical RGB, requested commands and outcomes in all three branches. Each branch started from the same archived TT17 coupled-gantry travel checkpoint, with its motor fit and rally and service experience. The held game seed was `1253086157`. The opponent and toss seeds, service order, and phases with ordinary, softened and delayed control followed the same scoring protocol and seed schedule. No branch spent a control frame on a dry camera probe. Games ended at 11 with a two-point margin, so their durations diverged with scoring. [TT17 checkpoint](../tt17/motor-gantry-coupling-travel-player.json.gz)

| Attempt | Change presented to the two changed-camera branches | Adaptive point record | Adaptive legal returns | Frozen point record | Frozen legal returns |
| --- | --- | --- | --- | --- | --- |
| 1–2 | Original two views | 1–1 | 4 | 1–1 | 4 |
| 3–4 | Swap the streams and remount one camera | 2–0 | 3 | 0–2 | 0 |
| 5–6 | Add a third camera | 0–2 | 0 | 0–2 | 0 |
| 7–8 | Remove two cameras | 0–2 | 0 | 0–2 | 0 |
| 9 onward | Restore a side camera and add a high camera | 6–6 | 3 | 0–4 | 0 |

The first RGB frame of the swap verified both camera changes. Three legal returns followed on that point. The third camera registered from its first frame, and the player lost both points in that stage. During the two single-view attempts, the player received 55 RGB captures and reported zero metric ball or blade sightings. It stayed with its existing ready-pose behavior and lost both points. When multiple views returned, the new high camera registered on the opening frame and the player made a legal return and won that point. The stable game recorded one unnecessary refit of an unchanged view on attempt 19.

The controller received synchronized RGB under opaque stream IDs, its previous requested commands, and scalar point rewards. The scene used surveyed colored landmarks, an orange ball, a magenta blade and a colored rim marker used to estimate wrist pitch. Camera positions, angles and lenses were private to the renderer. The tracker checked landmarks at point start, on a stream-set change, and every eight frames. It triangulated current object detections with at least two geometrically consistent views. The frozen branch reused the original RGB-derived projections on available known IDs throughout. The physical model is the same TT17 gantry and table. Its full 20-point TT17 action tape produced exactly the same physics events with the added visual geometry.

The receipts include requested and applied actions, referee events, RGB frame hashes, frame-level sighting flags, and a lossless opening RGB strip for each used stage. The verifier replays all 52 action tapes through MuJoCo with rendering disabled, checks matched-prefix RGB and action digests, and validates the sample images and checkpoint hashes. Each portable checkpoint stores its RGB-derived fits and can be opened with `load_checkpoint` without a renderer. [Adaptive receipt](viewplay-1253086157-adaptive.json.gz) · [Frozen receipt](viewplay-1253086157-frozen.json.gz) · [Stable receipt](viewplay-1253086157-stable.json.gz) · [Adaptive checkpoint](viewplay-1253086157-adaptive-player.json.gz) · [Trial and verifier](../../scripts/experimental/tt18_viewplay_trial.py)

```bash
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt18_viewplay_trial artifacts/tt18 verify
```
