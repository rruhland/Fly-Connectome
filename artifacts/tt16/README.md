# What carried motor experience buys on unfamiliar controls

A short simultaneous movement of all motor ports identified the new wiring on two gantry and two five-axis installations. All four original table players then won a full game. One five-axis player won its next game 15–13 after a checkpoint reload without further calibration. Every player started with a previously trained table skill and camera-based body model. The signed port mapping was inferred without backpropagation. [Watch the five-axis arm play all 18 points of its 11–7 game](five-axis-full-game.mp4) · [Resume the five-axis player](coded-five-resumed-player.json.gz)

| Starting the four held games | Wins | Legal returns | Distinct points with at least two arm returns | First legal return, median elapsed frames |
| --- | ---: | ---: | ---: | ---: |
| Direct simultaneous calibration | 4/4 | 83 | 18 | 127 |
| Learn the mapping during scored table tennis | 1/4 | 54 | 12 | 164.5 |
| Transfer hidden-source search plus wrist check | 2/4 | 57 | 13 | 524 among the three games with a return |
| Same wrist check alone | 0/4 | 0 | 0 | No return |

Each elapsed clock includes 68 optical setup commands with familiar wiring, all earlier search experience when used, all dry movements, and the preceding scored frames. Controls share full game seeds, shifting disturbances, service rules and optics. The direct movement took 12 control intervals and 13 stereo pairs. We measured 5.98 points per 1,000 total control frames with simultaneous calibration, 4.53 for direct online remapping and 3.37 with the search and wrist check. [Protocol, tapes and clocks](transfer.md)

- **Transfer with an existing history.** On two further installations the search-trained player asked about its two most uncertain ports. This reached exactly the same play as the full-port test in both games while using 85% and 88% less requested motion at the transition. With search time charged, it still took longer. The identical two queries without search produced zero returns. [Search-guided queries](sparse.md)
- **Changing delay during calibration.** Inferring the motor clock raised exact maps from 0/12 to 9/12 under a fixed delay and from 3/12 to 9/12 under a changing delay. On a further 24 wirings, extending the same movement to 24 commands when uncertain raised exact fits from 11 to 20. On four complete games from that last cohort, returns rose from 20 to 44 and wins from zero to one. [Controller, costs and counterexamples](calibration.md)
- **New game with an inertial puck.** The transferred table map shortened average time to the first rewarded hold from 109.25 to 100 frames across 32 fresh courses. Cold and carried learners both earned 128/128 rewards by the end. Both learned the unfamiliar puck response online. The benchmark supplies a marker decoder, a goal-seeking objective and a dynamics model family. [Puck experiment](payload.md)
- **Six-joint rotary.** First-shot depth learning kept the inherited stroke on all 46 evaluated receiving contexts. A lateral sweep increased the count of second-return points on the initial held serves and lost that gain on replication. Across all 60 matched rival serves, it produced four such points versus five for the inherited stroke. [Practice, replication and game scores](rotary.md)

The five-axis active-calibration and search-assisted gantry checkpoints each retained all 180 native Pong action tapes, including the same 41 points. The tests cover two known body geometries with new wirings and new camera setups. Arbitrary joint layouts and six-joint match wins remain open. [Five-axis Pong audit](coded-pong.json.gz) · [Gantry Pong audit](transfer-pong.json.gz) · [Next research pass](../../docs/reconstruction/HANDOFF.md)

All 164 table-tennis tests passed. The repository-wide run recorded 1,662 passes, 11 skips and three failures. The same three failures reproduced on the parent revision. They involved a curriculum exception message, an unavailable calibration cache, and a bitwise native-CPU voltage comparison differing by 1.2 × 10⁻⁷.
