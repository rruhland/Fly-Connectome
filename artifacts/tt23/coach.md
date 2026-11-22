# Choosing far replies by learned near-arm return probability

On eight fresh near serves, the physical coach that ranked legal replies by a learned continuation score increased the longest rally from 15 to 27 alternating legal returns. It extended six of the eight paired rallies, tied one and shortened one. Four learned points reached ten returns and two reached twenty. All 16 points replayed the full referee event sequence and point result from both arms’ joint-rate commands. [Native points](coach-held.json.gz)

Past physical replies from the five-axis practice partner and earlier physical practice rivals supplied 87 distinct episodes across 22 starts. The frozen optical SO-101 from the ordinary game contributed 32. Each episode begins with a strictly legal far return and uses the ball position, velocity and spin at separation from the near table. Its label is whether the near arm’s next shot earned a strict legal return. The near arm made 59 of those returns. Identical traces across archived policies were counted once, and validation held out every episode from the same initial seed together. [Practice and grouped predictions](coach-practice.json.gz)

| Return predictor | Whole-seed Brier error | Ordinary frozen player Brier error |
| --- | ---: | ---: |
| Lateral, height and pace error from the stock corridor | .20349 | .23642 |
| Geometry plus signed arrival height, velocity and timing | .11988 | .15671 |
| Geometry, flight and timing plus explicit spin components | .12213 | .16280 |

The best grouped score used the ridge-regularized flight and timing model. Estimated acceleration includes spin in all three projections. Additional spin terms yielded a slightly higher error on this archive. The width of the supported region and the influence of the probability on live selection were fixed before the fresh trial.

The stock coach and learned coach share the target profile `[0.0, -0.90, 0.20, 2.40]`, joint-rate library, six-candidate exact-check cap and strict referee. The learned coach subtracts a multiple of estimated near return probability from the stock geometry cost when the candidate is supported by recorded practice. Each attempted candidate must pass the physical legal return check before ranking. Simulator state is available to the coach. The unchanged near player uses the ordinary match's RGB optics, public servo commands, arm model and frozen optical calibration.

The comparison uses eight fresh starts from schedule `2099526147`, with the same physical near serve in each pair and standard opening pace and spread. Each point runs from release to its referee result. The study measures rallies from a shared near-service setup.

Across all eight pairs, the toss observations, toss-camera digest, first 35 near commands and full referee events through the legal serve match exactly. The far arm starts planning during the serve’s flight, at frames 24–26, before the referee certifies the serve at frames 36–38. Both far tapes agree before planning. Any far contact occurs strictly after service certification. Two pairs first differ in far commands during the serve’s flight. The receipt records the service event and first command differences for every pair. [Service audit](coach-held.json.gz)

| Physical far coach | Legal near serves | Near legal returns | Far legal returns | Rallies ≥10 | Rallies ≥20 | Longest | Near point wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stock geometric | 8/8 | 19 | 26 | 1/8 | 0/8 | 15 | 1/8 |
| Learned continuation | 8/8 | 44 | 51 | 4/8 | 2/8 | 27 | 1/8 |

On seed `1823220138`, the stock coach made one legal return and the near arm made zero. The learned coach and the same near policy exchanged 27 alternating legal returns over 516 frames, with 14 far and 13 near returns. The near arm then missed onto its own half. The near service tape and full events through the legal serve matched. The first far command diverged at frame 24 when the learned coach selected a different stroke. Both points replayed from their two joint-rate tapes. [Scored tapes](coach-held.json.gz)

Build the fit with `python -m scripts.experimental.tt23_coach`. Run and replay the native points with `python -m scripts.experimental.tt23_coach_trial` in the project table-tennis environment.
