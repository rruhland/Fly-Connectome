# Wrist-driven lateral drift correction

This extra practice pass started from the retained four-drop checkpoint. The original wrist motion readout and motor weights stayed frozen. A fall-to-rise transition visible in wrist RGB triggered a short estimate of post-rebound lateral velocity. The controller used that estimate and the ball’s position relative to its home pose to aim a small opposing tilt at the next rebound. The optical arm geometry translated the tilt into bounded joint offsets while penalizing displacement of the blade center. [Controller and experiment](../../scripts/experimental/tt29_bounce_drift.py)

Twelve new releases explored tilt gains of 0, 0.40 and 0.85. Nine used the harder lateral, height and spin distribution from the previous continuation. Once each gain had played two hard releases, the selector used its own completed hard-drop reward above the first rebound to choose subsequent gains. The zero-gain incumbent made eight qualified rebounds on one hard practice release. Each harder experiment at either positive gain made one. The active gains tied, so the lower gain became the frozen challenger for validation and assessment. [Practice actions and optical corrections](tt29_bounce_drift.json.gz)

The promotion rule required a positive paired streak win balance, at least two additional rebounds overall, a median streak at least as high as the incumbent’s, and a positive rebound change on the difficult subgroup. The six gate releases and eight subsequent held releases each balanced ordinary and difficult releases. All three decks were disjoint from one another and every recorded earlier bounce course.

| Deck | Incumbent rebounds | Active drift rebounds | Difficult subgroup, incumbent / active | Longest, incumbent / active |
| --- | ---: | ---: | ---: | ---: |
| Six promotion releases | 21 | 18 | 3 / 3 | 13 / 10 |
| Eight untouched releases | 20 | 22 | 2 / 2 | 13 / 16 |

The gate rejected the challenger. On the untouched assessment it gained one ordinary paired streak and lost none. The difficult subgroup tied on total rebounds and on every paired streak. All 40 command tapes reproduced their physical outcomes. The deployment copy matches the original incumbent byte for byte. [Assessment and replay receipts](tt29_bounce_drift.json.gz) · [Retained deployment](tt29_bounce_drift_deploy.json)
