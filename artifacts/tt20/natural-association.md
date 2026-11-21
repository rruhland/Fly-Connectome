# TT20 opt-in RGB association around colored distractors

`NaturalViewTransport(..., associate=True)` keeps multiple connected RGB components per role. After natural court registration, it tests apparent extent against the known ball and paddle radii and the existing role bounds. When several components remain plausible, their scores use hue agreement and a soft court-position prior. A recent optical track adds a short motion prediction reprojected into the current image. It retains the previous color choice where no metric candidate qualifies, which preserves a bearing for the optional temporal tracker. The selector never reads tag locations, scene IDs, joint state or referee labels. The default selector remains the original single color component.

The measurement uses the two whole held points from the native tag factorial, one own-serve tape (632487885) and one rival-serve tape (1556439130). Both points repeat in clean, tags-only, posts-only and posts-plus-tags scenes. Each tracker sees only its own RGB. Lens, pixels, commands, full physics traces and referee events match the archived counterpart. The first-pass association uses size-only ranging and leaves the original one-eye paddle-angle ambiguity intact. [Archived factorial](arm-tag-factorial.json.gz)

Each cell shows legacy → associated metric estimates, with the number under 50 mm error in parentheses.

| Survey posts | Arm tags | Single camera | Ball estimates (under 50 mm) | Paddle-center estimates (under 50 mm) |
|---|---|---|---:|---:|
| Off | Off | Wrist | 22 (15) → 22 (15) | 61 (58) → 61 (58) |
| Off | Off | Outside | 68 (38) → 68 (38) | 94 (76) → 94 (76) |
| Off | On | Wrist | 0 (0) → 22 (15) | 58 (57) → 58 (57) |
| Off | On | Outside | 4 (4) → 68 (38) | 94 (76) → 94 (76) |
| On | Off | Wrist | 21 (14) → 21 (14) | 50 (50) → 60 (57) |
| On | Off | Outside | 68 (38) → 68 (38) | 0 (0) → 94 (76) |
| On | On | Wrist | 0 (0) → 21 (14) | 50 (50) → 57 (56) |
| On | On | Outside | 4 (4) → 68 (38) | 0 (0) → 94 (76) |

On clean input, all emitted positions and abstentions were identical for both roles and cameras. The outside paddle-center measurements recovered with posts alone are also identical to their clean counterparts. With arm tags alone, all wrist ball estimates match clean exactly. The outside camera recovers the same estimate count and median/p90 error as clean (40.6/167.3 mm), with one frame’s error rising from 177.0 to 403.1 mm. On the 4 marked outside ball frames the old selector retained, the new selector returns identical positions.

When posts and arm tags are both installed, associated wrist ball errors have a 36.0 mm median and 66.5 mm p90. Associated outside blade centers have a 10.3 mm median and 75.3 mm p90. Neither one-eye tracker emits a unique paddle plane.

The pilot comprises 94 frames per scene and camera. The metrics describe perception on frozen motions. They leave score effects, behavior under larger appearance changes, and distractors with equally plausible shape and motion open.

[Per-frame receipt](natural-association.json.gz) · [Association code](../../scripts/experimental/tt20_natural_association.py) · [Reproduction script](../../scripts/experimental/tt20_natural_association_trial.py)
