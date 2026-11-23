# Learning when to reuse an old reference

The original native Pong experiment found that a familiar reference was much better than resetting the internal origin on independently tilted starts. Here the receiving arm can change that decision during online play when the release location changes. Six independent streams each played twenty releases. The first twelve were followed by a return to each of the two stations. Four streams used the stock body and two used the altered body. All use the original stock optics in order to compare reference choices. [Replicated native transfer](reference-transfer.md) · [Online runner](../../scripts/experimental/tt24_station_switch.py)

One station holds the ball briefly over a fixed installation while the arm starts tilted. At the second station the ball immediately falls over a mobile, nearly level starting pose with up to 0.34 radians of shoulder displacement. In a development screen the more distant station failed to support long play. At the moderate station, the starting-pose convention averaged 9.25 consecutive rebounds against 5.63 for always using the familiar reference on eight starts. Both settings were frozen before online comparisons. [Wide station screen](station-switch-feasibility.json.gz) · [Moderate station screen](station-switch-immediate-feasibility.json.gz)

The arm can select 0, 0.5 or 1 as the degree of reference reuse. The same coefficients control both its motor equilibrium and visual origin. A kernel on public starting joint angles weights prior experience. Only the arm's own scalar rebound rewards update the expected value of each coefficient. A global version uses all experiences equally. After the first twenty plays each learner took eight fresh launches without exploration or further updates. The frozen comparator uses the same rounded native reference as the online grid. Identical controller decisions on the identical physical start reuse one actual on-policy rollout. [Rounding control and revisit runner](../../scripts/experimental/tt24_station_retention.py)

| Policy | Rebounds during 120 learning plays | 24 fresh fixed station | 24 fresh mobile station | Combined revisit |
| --- | ---: | ---: | ---: | ---: |
| Native Pong prior with local memory | 619 | 158 | 206 | 364 |
| Shuffled Pong reward with local memory | 483 | 110 | 189 | 299 |
| Neutral prior with local memory | 538 | 122 | 178 | 300 |
| Native Pong prior, global memory | 565 | 145 | 209 | 354 |
| Frozen same-grid native prior | 587 | 228 | 119 | 347 |

Online learning recovered 87 additional rebounds on mobile-station revisits while giving up 70 on fixed-station revisits. This weak overall increase reveals interference when old and new station contexts overlap in starting joint-angle space. The global adapter came close on fresh total. On the rounded frozen comparison, the online learner improved four of six learning streams and lost two. Performance at both stations must be tracked independently.

[Stream 1](station-switch-immediate-stream-0.json.gz) · [2](station-switch-immediate-stream-1.json.gz) · [3](station-switch-immediate-stream-2.json.gz) · [4](station-switch-immediate-stream-3.json.gz) · [5](station-switch-immediate-stream-4.json.gz) · [6](station-switch-immediate-stream-5.json.gz)

[Revisit 1](station-revisit-0.json.gz) · [2](station-revisit-1.json.gz) · [3](station-revisit-2.json.gz) · [4](station-revisit-3.json.gz) · [5](station-revisit-4.json.gz) · [6](station-revisit-5.json.gz)
