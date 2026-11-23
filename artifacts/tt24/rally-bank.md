# Broader demonstrations for direct joint feedback

Forty-four existing TT23 receiving archives contain at least two strict legal near returns on 109 distinct eligible releases after the previous demonstration and assessment releases are excluded. The expanded bank selects one teacher tape per release. It interleaves four strata based on launch direction and lane width, drawing from the sampled strata in a fixed order. It includes the two longest tapes per stratum, then draws without replacement. None of the new E/F launches occurs in the TT22 or TT23 JSON archives. [Harvest and independent release generator](../../scripts/experimental/tt24_rally_bank.py)

The 48 selected teacher points cover 13,107 control frames, or 436.9 simulated seconds. Each full two-arm tape reproduces its terminal frame, score, referee events and return counts. At fixed 240×180 stereo resolution, the extractor renders 5,706 frames and keeps 5,124 observation/command pairs around the first landing and the first, middle and last successful continuation contacts. The earlier frames in each window initialize visual history without using future frames. Teacher scoring selects the fit windows. Its only output to the policy fit is the observation and issued rate for one frame at a time. [Teacher exposure and exact replay receipts](rally-bank-demonstrations.json.gz) · [Extracted training pairs](rally-bank-demonstrations.npz)

The original 12-tape fit and the broader fit both drive every frame from release on the fixed 24-case E/F roster. Both use live RGB, proprioception and their own command history to calculate five joint rates. The older law retains its original local neighborhood of 20 and the wider fit uses 28, so the comparison covers both the dataset and modest local smoothing change. The camera and physical partner are shared. The match keeps all launches and includes hold and a six-frame visual-delay control. None of the four branches receives an assessment teacher command.

## Held results

| Outcome on 24 new releases | Original 12 tapes | Broader 48 tapes | Delayed RGB | Hold |
| --- | ---: | ---: | ---: | ---: |
| First near return | 11 | 14 | 0 | 0 |
| Second near return | 2 | 8 | 0 | 0 |
| Fifth near return | 0 | 2 | 0 | 0 |
| Tenth near return | 0 | 0 | 0 | 0 |
| Total near returns | 13 | 32 | 0 | 0 |
| Longest alternating exchange | 4 | 10 | 0 | 0 |

Seven paired launches reached the second return with the broader controller alone. One reached it with the original controller alone. The other shared success reached the fifth return with the broader controller. The broader pupil reached the second return on five moderate and three wide launches, compared with two moderate and zero wide for the original pupil. At least one near-arm command changed on 885 of the 1,288 overlapping control frames. Across the five posing joints, the per-channel change counts were 855, 847, 852, 859 and 862. Every one of the 96 complete physical command tapes reproduced its recorded referee sequence and terminal result. [Matched outcomes and joint commands](rally-bank-held.json.gz)

The two fifth-return points each had five legal near returns and five legal far replies. Seed `1176636968` reached its fifth near landing at frame 178, on the fifth point of the course and after 483 cumulative control frames, or 16.1 simulated seconds. Its near landings occurred at frames 37, 74, 110, 142 and 178. Seed `1682479719` landed at frames 36, 72, 109, 148 and 183, and the original pupil missed its first return on that release. The longest exchange reached ten alternating returns. [Ten-return held video](rally-bank-held.mp4) · [Video verification](rally-bank-held.json)

## Matched local smoothing

The two additional branches use the same held schedule, RGB size and body-state weighting. Each changes the neighborhood size of one fitted policy while keeping its demonstration bank fixed.

| Demonstrations and neighbors | First near return | Second near return | Fifth near return | Total near returns | Longest alternating exchange |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 tapes, 20 neighbors | 11 | 2 | 0 | 13 | 4 |
| 12 tapes, 28 neighbors | 12 | 6 | 0 | 20 | 8 |
| 48 tapes, 20 neighbors | 14 | 7 | 0 | 26 | 6 |
| 48 tapes, 28 neighbors | 14 | 8 | 2 | 32 | 10 |

The broader bank increased second-return frequency at both neighborhood sizes. The two fifth returns occurred with the broader bank at 28 neighbors. All 48 additional physical tapes passed strict command replay. [Broader fit at 20 neighbors](rally-bank-matched-locality.json.gz) · [Original fit at 28 neighbors](rally-bank-original-wide-locality.json.gz)

## Reproduce

Use the [TT23 environment](../tt23/tt23_sequence.md#reproduce) and the earlier [rally training banks](rally.md#reproduce).

```sh
python -m scripts.experimental.tt24_rally_bank --workers 2
python -m scripts.experimental.tt24_rally_bank_locality
python -m scripts.experimental.tt24_rally_bank_locality --original
python -m scripts.experimental.tt24_rally_movie --bank
```
