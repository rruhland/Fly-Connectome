# TT85 | The stock wrist view loses useful overlap during receiving

On 16 ordinary moving-arm replays, the centered SO-101's stock wrist camera saw no ball at an actionable lead. The known incoming ball was outside the geometric camera image cone in all 55 postbounce frames taken 67–125 ms before actual impact or the virtual contact depth. Fifty ball centers were behind the lens and five were in front but outside its field of view. The natural red blade was visibly in frame. The views often point at the ceiling.

| Archived receiving outcome | Independent motions | Wrist RGB sightings in the 67–125 ms window |
| --- | ---: | ---: |
| Low center, missed without contact | 4 | 0 / 14 frames |
| Low center or wing, touched without a legal return | 4 | 0 / 13 |
| Low, legally returned | 4 | 0 / 15 |
| Routine, legally returned | 4 | 0 / 13 |

The result was not sensitive to that narrow deadline. Across 171 postbounce frames 50–235 ms before impact or virtual crossing, none even entered the lens cone. Three low contacts showed an RGB ball for a single frame 14.6 ms before impact, two of them clipped. One of those was a stock legal return. Two routine returns had a brief view beginning at 29.2 and 35.4 ms. None of the four untouched central balls was visible before the virtual contact plane. The seven valid pre-contact bearings were late and had errors of 2.17–8.95 mm against the true line of sight. These are bearing errors only, not full 3-D errors.

## Optical and physical boundary

The arena and full native physical routes were unchanged. A 384×288 raw RGB wrist frame was captured every 60 Hz control tick around the archived receiving phase. The actual stock rigid board remained at the documented (+6,+54,−35) mm gripper aperture with its existing 124° vertical optics, bracket, opaque jaw and blade. All far-hit, bounce and measured near-contact packets reproduced exactly. A known ball pose was used only to check whether the unassisted orange-component detector really saw the ball and to assess geometric visibility. The native encoder camera supplied the bearing measure. No hidden labels, geometry, or RGB overlays changed any movement or camera view. On no-touch paths, the actual incoming ball's crossing of y=−1.53 m gave the comparison clock.

The earlier TT29 wrist-only 16-return example used a different offset mount and ready posture. It does not demonstrate late visual overlap on this fixture. On the tested ordinary body motions, a detector or fusion change cannot recover footage that does not enter the camera. No corrected arm point was attempted.

[Protocol](protocol.json) · [All frame and physical receipts](visibility.json.gz) · [Summary](summary.json) · [Actual wrist RGB 81 ms before an unscored touch](frames/central-or-wing-low-unscored-touch-3355-tick81.png) · [Same wrist RGB 14.6 ms before contact](frames/late-3355-tick85.png)
