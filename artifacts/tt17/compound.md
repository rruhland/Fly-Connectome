# Simultaneously changing port assignments, gains and coupling

The same 24-frame visual movement produced 37 legal returns when the player estimated gain and coupling, versus six when it learned only motor order and sign. It increased returns and score on all four installations. All eight games were losses. A two-stage query nearly won one gantry game but produced fewer returns in aggregate. [Motor fit](../../scripts/experimental/tt17_motor_fit.py) · [Full-game protocol](../../scripts/experimental/tt17_compound_games.py)

Each cell contains the final score, total legal returns and number of points with two or more legal returns.

| Installation | Signed only | Continuous fit | Two-stage query |
| --- | --- | --- | --- |
| [Five-axis A](compound/compound-five-0-games.json.gz) | 3–11 / 2 / 0 | 4–11 / 4 / 0 | 4–11 / 4 / 0 |
| [Five-axis B](compound/compound-five-1-games.json.gz) | 1–11 / 3 / 1 | 8–11 / 13 / 3 | 6–11 / 3 / 0 |
| [Gantry A](compound/compound-gantry-0-games.json.gz) | 0–11 / 1 / 0 | 6–11 / 7 / 2 | 3–11 / 4 / 0 |
| [Gantry B](compound/compound-gantry-1-games.json.gz) | 2–11 / 0 / 0 | 8–11 / 13 / 2 | 10–12 / 19 / 2 |

Four seeded, previously unplayed installations simultaneously changed port order, polarity and continuous cross-axis response. They reused two TT17 camera calibrations. All methods shared 68 frames of original-control optics, an archived arm model and 24 frames on the changed controls. The signed and continuous fits saw the same continuous movement. The two-stage query inferred an assignment from the first 12 frames, reset the simulated arm, then adapted the last 12 commands to reproduce the native movement. The reset gives this variant an extra intervention. Each fitted map stayed fixed during its game. Opponent, noise and point seeds were paired. As scores diverge, the number of points and the side serving later points can differ. Each game was replayed through the simulator. [Query implementation](../../scripts/experimental/tt17_compound.py)

On 12 independent physical pulse sequences per fixture, mean error predicting how the installation changed tool displacement fell from 79 to 47 mm, 62 to 46 mm, 99 to 52 mm and 73 to 57 mm in table order. Two-stage errors were 65, 52, 75 and 48 mm. The second query helped on one installation. The continuous one-pass fit is the better starting point for this family. This demonstrates rapid calibration of a previously trained player's arm. It uses a known reference movement and a supplied stroke controller. [Independent motion, five-axis A](compound/compound-five-0-motion.json.gz) · [Five-axis B](compound/compound-five-1-motion.json.gz) · [Gantry A](compound/compound-gantry-0-motion.json.gz) · [Gantry B](compound/compound-gantry-1-motion.json.gz)
