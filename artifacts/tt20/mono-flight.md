# One-camera flight and receiving

The same 12 held opponent-service starts ran with the wrist camera and the left outside camera, one stream per run. Each view used the unmarked SO-101 end-clamp station at 480 × 360 and 30 Hz. Both conditions used the same frozen TT19 factory receiver, normal painted court lines, orange ball silhouette and declared lens intrinsics. Each point started with empty optical history and no target practice.

| View | Ranging | Starts with paddle contact | Starts with legal return | Total legal returns | Points won |
|---|---|---:|---:|---:|---:|
| Outside | Ball size | 9/12 | 2/12 | 2 | 2/12 |
| Outside | Temporal fit | 11/12 | 5/12 | 6 | 0/12 |
| Wrist | Ball size | 12/12 | 8/12 | 9 | 5/12 |
| Wrist | Temporal fit | 12/12 | 6/12 | 7 | 0/12 |

Five outside starts produced a legal return only with the temporal fit and two only with ball size, giving an exact paired p-value of .453. One wrist start returned only with the temporal fit and three only with ball size, giving a p-value of .625. The outside return counts favor the temporal estimate in this sample. The wrist return counts and both point-win comparisons favor ball size.

On actionable frames with both estimates available from the same RGB, the outside median position error fell from 36.4 to 26.3 mm and velocity error fell from 1.473 to .115 m/s across 115 frames. The wrist position error grew from 4.1 to 8.0 mm while velocity error fell from .094 to .050 m/s across 49 frames. These comparisons use the RGB captured in the temporal branch, including frames during paddle motion and later exchanges.

The temporal fit uses up to 14 recent ball bearings with an apparent-size prior and competing smooth and table-bounce solutions. The table dimensions, 20 mm ball radius and gravity set metric scale. Drag and bounce are nominal physical approximations. The receiver accepts the fit's current velocity directly since successive fitted positions can shift as the history changes. It falls back to the original four-sighting speed estimate when the current position comes from ball size alone. The factory controller retained the published CAD stroke catalogue and dry model, including the original mount position.

The policy read RGB and its own command history. Simulator ball state was recorded for error measurements and the simulator referee produced the scores. All 48 action tapes reproduced the recorded score and complete physics event trace from the same point seeds.

- [Outside-camera actions, optical estimates and replay receipt](mono-flight-outside.json.gz)
- [Wrist-camera actions, optical estimates and replay receipt](mono-flight-wrist.json.gz)
