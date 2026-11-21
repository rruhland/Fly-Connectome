# Learning return depth from stereo and point reward

The online learner won both fresh games on the original gantry, with 41 legal returns and eight sustained rallies. The registered player won one game, with 34 legal returns and eight sustained rallies. The gains in follow-up returns came after the gantry's own serve. Freezing the optically learned choice for matched incoming balls moved the median first landing deeper and yielded six wins in 13, compared with seven for the registered player. Carrying the placement memory to five-axis D and back to the gantry produced two losses. [Audited comparison](place-comparison.json.gz)

## Matched incoming balls

The learner practiced on 24 disjoint points, choosing the middle, short and deep targets 4, 4 and 18 times. Stereo supplied 11 landing measurements across 17 legal returns. Including depth in the reward estimate favored the deep target at the end of practice. The direct reward estimate from the same practice history favored the short target. The trial froze those two choices and compared them with the original stroke on 13 matched playable first balls, including four replies to the gantry's own serve. [Practice tape](place-practice.json.gz) · [Matched point tapes](place-pairs.json.gz)

| Policy | Point wins | Legal first returns | Legal later returns | Wins with a later ball | Median first bounce from net |
|---|---:|---:|---:|---:|---:|
| Registered player | 7/13 | 10/13 | 12/16 | 4/7 | 0.573 m |
| Frozen choice from stereo and reward | 6/13 | 8/13 | 8/11 | 3/5 | 0.673 m |
| Frozen choice from reward alone | 7/13 | 7/13 | 11/14 | 4/4 | 0.535 m |

The deeper choice gained two points the registered player lost and lost three points it won. On the eight matched balls where both landed a return, the median paired shift was 2.5 cm deeper. The shorter choice had a 1.7 cm median shift toward the net on seven such pairs. The differences across all legal bounces also reflect which balls each policy returned. On the four own-serve replies, the registered, deeper and shorter choices won two, zero and two points. Their legal first follow-ups were three, two and two.

## Full games and installation change

| Fixture | Policy | Score | Legal own serves | Wins on playable first balls | Legal first returns | Legal later returns | Wins with a later ball |
|---|---|---:|---:|---:|---:|---:|---:|
| Gantry A | Registered | 11–9 | 7/10 | 9/15 | 10/15 | 7/10 | 4/6 |
| Gantry A | Online placement | 11–9 | 8/12 | 8/15 | 11/15 | 11/15 | 3/7 |
| Gantry B | Registered | 10–12 | 10/11 | 8/19 | 10/19 | 7/11 | 2/5 |
| Gantry B | Online placement | 11–7 | 8/10 | 8/13 | 10/13 | 9/15 | 4/6 |
| Five-axis D | Registered | 11–4 | 6/7 | 10/13 | 12/13 | 4/6 | 3/5 |
| Five-axis D | Carried placement | 6–11 | 7/8 | 2/12 | 5/12 | 2/5 | 2/5 |
| Gantry revisit | Registered | 8–11 | 8/10 | 6/16 | 10/16 | 7/11 | 2/6 |
| Gantry revisit | Carried placement | 7–11 | 7/10 | 4/13 | 7/13 | 3/7 | 2/5 |

In the first two gantry games, online placement landed 7/10 first follow-ups and 16/20 later returns following the gantry's own serve. The registered player landed 4/13 and 5/7. The online player won 5/10 points with a playable own-serve reply versus 4/13. On rival serve the win rates were close, at 11/18 versus 13/21. Across both service roles the online player had 30 later opportunities versus 21. [Gantry A registered](place-game-a-fixed.json.gz) · [Gantry A online](place-game-a-online.json.gz) · [Gantry B registered](place-game-b-fixed.json.gz) · [Gantry B online](place-game-b-online.json.gz)

The transfer carried the placement memory from game A into the post-game registered checkpoint for D. The revisit imported the resulting placement memory into the original gantry checkpoint. Each branch used its installation's own motor model, registration and stereo tracker. Both branches had one complete game per fixture. Carrying the placement memory reduced first and later returns in each fixture. [D registered](place-game-transfer-fixed.json.gz) · [D carried](place-game-transfer-online.json.gz) · [Gantry revisit registered](place-game-revisit-fixed.json.gz) · [Gantry revisit carried](place-game-revisit-online.json.gz)

## Experiment

The experiment resumed the TT12 two-probe gantry checkpoint after its first complete game. Both branches use its registered toss controller, calibrated motor response, contact plane and incoming-flight estimator. The placement variant chooses among nominal landing depths of 0.76, 0.27 and 1.16 m from the net on the rival half. The 0.76 m choice calls the original stroke. The other choices change the requested blade inclination by up to 0.14 rad. The player recognizes receiving strokes from stereo, including after a rival replies to its own serve. The achieved bounce depends on the blade motion, impact and ball flight. [Starting checkpoint](../tt12/motor-gantry-recall-player.json.gz)

Stereo provides the depth of a visible rebound or a low ball sighting supported by two preceding outbound sightings. At point end the learner records the depth of each observed landing and assigns the public point reward to the last attempted stroke. A smoothed reward estimate over landing depth contributes 45% of each choice's value. The direct point reward estimate contributes the rest. The controller first samples each choice and then uses a small exploration bonus during play. The three target depths and the credit assignment are part of the controller design. [Learner](../../scripts/experimental/tt13_place.py)

The evaluator used ball state and contacts only after play. Across the receipts it associated all 109 optical landings with a legal return, out of 198 scored legal returns. The median absolute depth error was 2.8 cm. Visibility varied by fixture and shot, so the learner's depth evidence represents a subset of the returns. All 219 requested-command tapes were replayed through the physics and scorer. [Audited comparison](place-comparison.json.gz)

Each matched branch forks the same registered player before the point and checks equal legal incoming-bounce events, requested and applied commands, and ball sightings through the first placement decision. Each complete-game comparison uses a common registered start and prospective tosses and rival openings in separate streams. The practice supplies only placement memory to the online branch. The ordinary updates continue in both full-game branches. A playable first ball is a legal rival serve or a legal rival reply to the player's own serve. Each subsequent legal rival return contributes a later opportunity. The later win counts refer to final results on points with at least one such opportunity. [Trial and replay](../../scripts/experimental/tt13_place_trial.py)

## Reproduction

From the repository root, use the MuJoCo hall environment.

```sh
python -m scripts.experimental.tt13_place_trial practice artifacts/tt13/place-practice.json.gz --seed 431730923 --limit 24 --checkpoint artifacts/tt13/place-practice-player.json.gz
python -m scripts.experimental.tt13_place_trial pair artifacts/tt13/place-pairs.json.gz --seed 117284391 --limit 18 --memory artifacts/tt13/place-practice-player.json.gz
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-a-fixed.json.gz --seed 860912361 --policy fixed
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-a-online.json.gz --seed 860912361 --memory artifacts/tt13/place-practice-player.json.gz --checkpoint artifacts/tt13/place-game-a-player.json.gz
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-b-fixed.json.gz --seed 371850314 --first-server 1 --policy fixed
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-b-online.json.gz --seed 371850314 --first-server 1 --memory artifacts/tt13/place-practice-player.json.gz --checkpoint artifacts/tt13/place-game-b-player.json.gz
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-transfer-fixed.json.gz --seed 689410733 --resume artifacts/tt12/motor-d-recall-player.json.gz --policy fixed
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-transfer-online.json.gz --seed 689410733 --resume artifacts/tt12/motor-d-recall-player.json.gz --memory artifacts/tt13/place-game-a-player.json.gz --checkpoint artifacts/tt13/place-game-transfer-player.json.gz
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-revisit-fixed.json.gz --seed 521074251 --first-server 1 --policy fixed
python -m scripts.experimental.tt13_place_trial game artifacts/tt13/place-game-revisit-online.json.gz --seed 521074251 --first-server 1 --memory artifacts/tt13/place-game-transfer-player.json.gz --checkpoint artifacts/tt13/place-game-revisit-player.json.gz
python -m scripts.experimental.tt13_place_report artifacts/tt13 --additional transfer revisit
```

Replay any receipt with `python -m scripts.experimental.tt13_place_trial verify PATH`.
