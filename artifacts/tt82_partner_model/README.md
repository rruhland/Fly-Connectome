# The previous arm stroke changes the opponent's next delivery

Two different legal arm strokes were physically executed from each of 49 matching incoming situations. In an independent test of each stroke, the real opponent reacted to six unfiltered goals, and an actual receiver attempted the next return. To forecast the difference between the strokes, a separate simulated partner received each achieved departure and ten *different* continuous random goals. It started with a neutral far carriage. A receiver model fitted on other releases estimated how often the next ball could be returned. The private simulation had the actual outgoing position, velocity and spin, information available to this diagnostic only after the arm had struck. [Design and source assignments](protocol.json)

| Forecast of the difference between two strokes | 32 development forks | 17 independent forks |
|---|---:|---:|
| Correct direction of average next height | 29 / 32 | 17 / 17 |
| Correlation with measured difference in height | .849 | .941 |
| Estimated gain in actual next return probability versus an uninformed choice | +7.0 percentage points | +4.4 percentage points |
| Release-bootstrap interval for that gain | +2.9 to +11.7 points | −0.5 to +10.8 points |

All 980 private draws produced a measurable next delivery. Knowing how a stroke really left the blade made differences in the following ball's height quite predictable. Predicting which stroke would score the next *actual* legal return was more difficult. The 17-fork result has six correct signs among nine nonzero contrasts. One earlier engineering pilot left an unrelated near paddle in the private scene, which could hit the imported departing ball again. The final forecasts move that passive paddle clear of the ball and reject an early double contact. Neither private launch nor simulated competence counts as a game. [All development and held forecasts](report.json) · [Physical actions, measured goals and real next returns](../tt55_paired_afterstate/README.md)

Learning a model of a cooperating partner from the arm's own experience is also part of i-Sim2Real. That work used an industrial arm on a linear stage. Its rally counts do not establish performance for this fixed SO-101. [Abeyruwan et al., CoRL](https://proceedings.mlr.press/v205/abeyruwan23a.html)
