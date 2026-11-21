# Body learning during scored table tennis

On the compact gantry, the transferred body model produced a legal return at frame 245 and a legal own serve at frame 475. Starting with zero motor gains moved those milestones to frames 681 and 530. The frozen body fit reached them at frames 48 and 309 and won the most points on that fixture. On the extended five-axis arm with rearranged and reversed ports, the cold body learner reached the first legal return sooner than the transferred learner, and the separate service-variation experiment landed the fixture's only legal own serve. All 22 full games ended in losses. [Comparison summary](online-summary.json.gz)

## Two-game streams

Each condition starts independently from the same TT17 stroke and vision checkpoint for its fixture. The second game resumes the player's stored body and stroke experience from the first game on a disjoint point schedule. Every game uses alternating service, an 11-point target and a two-point lead. The target clock starts on the first scored command and all conditions spend zero frames on target setup. [Runner and replay](../../scripts/experimental/tt18_online_trial.py)

| Physical fixture | Body and service policy | Game 1 | Game 2 | Legal own serves | Legal returns | Points with two returns |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Compact gantry | Carried, active | 1–11 | 1–11 | 2/12 | 3 | 0 |
| Compact gantry | Cold, active | 0–11 | 1–11 | 2/11 | 2 | 0 |
| Compact gantry | Carried, passive | 1–11 | 2–11 | 3/12 | 5 | 0 |
| Compact gantry | Carried, frozen | 3–11 | 2–11 | 4/12 | 4 | 0 |
| Compact gantry | Carried, active, with service variations | 1–11 | 3–11 | 3/12 | 3 | 0 |
| Extended five-axis, changed ports | Carried, active | 0–11 | 1–11 | 0/11 | 1 | 0 |
| Extended five-axis, changed ports | Cold, active | 0–11 | 2–11 | 0/11 | 4 | 1 |
| Extended five-axis, changed ports | Carried, passive | 0–11 | 2–11 | 0/11 | 2 | 0 |
| Extended five-axis, changed ports | Carried, frozen | 0–11 | 1–11 | 0/11 | 0 | 0 |
| Extended five-axis, changed ports | Carried, active, with service variations | 1–11 | 2–11 | 1/12 | 4 | 0 |
| Extended five-axis, changed ports | Carried, active, with signed basis recall | 0–11 | 4–11 | 0/12 | 3 | 0 |

The compact gantry retains its earlier port convention through a fresh gain and coupling matrix. The extended five-axis arm draws a new hidden permutation and signs under a fresh gain and coupling matrix. Both fixtures change learner mass and joint travel, table dimensions, ball mass and left/right table contact damping. A mounted camera contributes mass on the wrist or proximal link. Both observation streams use their archived fixed stereo rig with supplied camera poses, plus backgrounds and tracking apertures from the earlier TT17 68-frame optical setup. The body prior also carries the earlier 24-frame optical fit. [Physical plant](../../scripts/experimental/tt18_plant.py) · [TT17 gantry donor](../tt17/motor-gantry-coupling-bounded-start-player.json.gz) · [TT17 five-axis donor](../tt17/motor-five-coupling-bounded-start-player.json.gz)

## From observed motion to play

The active learner adds a small stochastic motor adjustment to its currently planned action. It chooses a port based on uncertainty in its fitted movement and the rate remaining on that port. The adjustment decreases when the predicted blade motion spans three spatial directions and when a strike draws close. The body update regresses RGB blade position and inclination against accumulated, delay-aligned public actions with a quadratic feature model. The cold version starts with a generic table-ready pose and settling curve, and zero command gains. The passive version learns from the planner's commands. The frozen version keeps the carried body coefficients. The four share the inherited stroke controller, registration learning, visual timing and camera calibration. The policy receives stereo images, the prior requested action and the ordinary terminal point reward. [Learner and controls](../../scripts/experimental/tt18_online.py)

On the five-axis arm, median body residuals in the two games were 124 and 152 mm for frozen, 46 and 41 mm for active carried and 35 and 35 mm for active cold. These are RGB residuals measured before updating the coefficients with the current blade sighting. The servo aligns the commands to the image using its current visual delay estimate, which can incorporate that sighting. The policies' own commands generate different trajectories. The cold learner landed the only point with two legal returns at frame 426. [Five-axis active](online-five-extended-newports-198120436-carried.json.gz) · [Five-axis cold](online-five-extended-newports-198120436-cold.json.gz) · [Five-axis frozen](online-five-extended-newports-198120436-frozen.json.gz)

The service branch chooses among the inherited stroke and five small variations to toss contact height, blade pitch or forward speed, using earlier point reward and visible ball departure, bounce and net passage. The five-axis player landed its first legal own serve at frame 901 using the higher pitch variation. The same branch landed three legal serves on the compact gantry while the frozen baseline landed four. [Gantry service tape](online-gantry-compact-631040223-serve.json.gz) · [Five-axis service tape](online-five-extended-newports-198120436-serve.json.gz)

The signed basis recall branch matches the observed motor Jacobian to signed permutations of the related body's motor axes. It adopts a candidate only if the next scored point supplies enough new blade sightings and favors its prediction on the same commands. The five-axis player adopted one candidate and reached 4–11 with three legal returns in its second game. It landed zero legal own serves and reached zero points with two legal returns. [Basis recall tape](online-five-extended-newports-198120436-recall.json.gz)

## Reproduce

Run in the table-tennis simulation environment with MuJoCo 3.17. Use headless EGL for the saved run. Each receipt contains commands, their hashes, applied rates, image-derived sightings and referee events, with a separate checkpoint for each completed game. The evaluator reproduced all 270 scored point tapes on the physical plant. The RGB replay independently regenerated the decisions and final learner state for 98 points across both carried streams and the two five-axis extensions.

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for mode in carried cold passive frozen serve; do
  python -m scripts.experimental.tt18_online_trial artifacts/tt18 \
    --site gantry --plant compact --seed 631040223 --mode "$mode"
  python -m scripts.experimental.tt18_online_trial artifacts/tt18 \
    --site five --plant extended --fresh-ports --seed 198120436 --mode "$mode"
done
python -m scripts.experimental.tt18_online_trial artifacts/tt18 \
  --site five --plant extended --fresh-ports --seed 198120436 --mode recall
python -m scripts.experimental.tt18_online_trial artifacts/tt18 --rgb --verify \
  artifacts/tt18/online-five-extended-newports-198120436-carried.json.gz
python -m scripts.experimental.tt18_online_report artifacts/tt18/online-summary.json.gz \
  artifacts/tt18/online-{gantry-compact-631040223,five-extended-newports-198120436}-{carried,cold,passive,frozen,serve}.json.gz \
  artifacts/tt18/online-five-extended-newports-198120436-recall.json.gz
```
