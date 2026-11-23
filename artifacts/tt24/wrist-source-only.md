# One wrist camera for adaptation and play

Two generic motions on the altered arm produced 96 wrist images across 192 control frames, or 3.2 seconds. An existing algorithm fitted the camera mount and corrections to link geometry using the encoders and four visual boundaries from a previously surveyed scene. The same frozen donor-trained motor weights and consistent reference were then used on 16 fresh wide launches. This test uses the previously calibrated altered body and fresh game starts. [Calibration budget and geometry](wrist-egomotion.md) · [Physical comparison](wrist-source-only.json.gz)

| Optical model | Mean longest streak | Maximum | Ten consecutive |
| --- | ---: | ---: | ---: |
| Stock observer | 5.00 | 21 | 3/16 |
| Mount alone from wrist | 3.00 | 11 | 1/16 |
| Mount and links from wrist | 9.88 | 19 | 7/16 |
| Previously calibrated using outside cameras | 9.63 | 22 | 7/16 |

The wrist-only joint fit beat the stock map on ten starts, tied four and lost two. Its paired bootstrap interval for the gain in longest streak was 0.81 to 8.81. Against the paired-camera map the corresponding interval was −3.44 to 3.63. On four fresh stock revisits wrist-based recognition recovered the complete original commands and scores of 5, 18, 18 and 19. All 72 physical replays matched. No scored receiving-arm practice was used. Earlier stock calibration supplied metric scale and the familiar scene. [Runner](../../scripts/experimental/tt24_wrist_source_only.py) · [Motor training](tt24_donor.md)
