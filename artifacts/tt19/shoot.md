# Shooting with a paddle model fitted from RGB

A controller that searched complete motor trajectories served and returned legally in a strict scored SO-101 game. It used the wrist and fixed stereo RGB, its own previous motor requests and the public service role. The nominal ball model produced a 1–11 loss with three legal returns and one legal own serve. Substituting the gantry flight law and impact matrix produced a 0–11 loss with two legal returns and zero legal own serves. The two branches used the same learned SO-101 body model and matched point schedule. [Nominal game](shoot-409676221-nominal.json.gz) · [Gantry transfer game](shoot-409676221-gantry.json.gz)

| Ball model | Score | Opponent-service points with blade contact | Legal returns | Legal own serves | Scored control frames | Points with ≥2 returns |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal flight and impact | 1–11 | 6/6 | 3 | 1/6 | 516 | 0 |
| Gantry flight and impact | 0–11 | 6/6 | 2 | 0/5 | 404 | 0 |

The policy searched 360 two-stage motor tapes per plan. A ridge quadratic predicted the paddle center and two normal components from filtered motor-request histories. The search tested a descending or arriving ball against each moving disk at four substeps per frame. It estimated the first approaching contact, including the rim, then used the local paddle velocity and tilt to predict a subsequent flight. It scored candidate landing position, net clearance and bounce sequence. During the scored run, fresh blade observations corrected the model’s current offset and new ball observations could revise the contact plan. The learner used the published court, ball and paddle dimensions. The vendor kinematics, joint state and referee events were unavailable to the planner. [Planning code](../../scripts/experimental/tt19_shoot.py)

The shared setup used 20 native 14-command optical probes, with 16 sequences for fitting and four for model selection. Both games were charged all 280 setup frames. The selected model used nominal settling. Its position error on the four model-selection sequences was 58 mm median and 140 mm p90. The best zero-extra-lag fit with gantry settling had 68 mm median and 113 mm p90. The selected body model missed relative blade displacement by 21 mm median over one frame and 56 mm over three frames. Full blade normals were available in only six of the 53 model-selection position sightings. The selected model’s median normal error on those six measurements was 10.5°. [Optical setup, fits and model selection](shoot-body-fit.json.gz)

The nominal branch made its first legal return on scored frame 38 and a legal own serve on scored frame 170. The gantry branch made its first legal return on scored frame 43. Both produced a predicted contact and nominally legal landing on every point, while the physical referee counted four legal shots in the nominal branch and two in the gantry branch. This prediction gap limits how much can be inferred from the selected trajectory before it is played. The flight and impact priors were switched jointly, so the comparison identifies their combined effect on this held schedule. [Gantry flight fit](flight-source.json.gz)

All 23 physical point tapes and all 23 planning decisions replayed exactly. Physical replay used the saved motors with rendering disabled. Decision replay reconstructed every requested action and selected plan from the recorded RGB-derived observations and the seed. [Nominal checkpoint](shoot-409676221-nominal-player.json.gz) · [Gantry checkpoint](shoot-409676221-gantry-player.json.gz) · [Replay entry point](../../scripts/experimental/tt19_shoot_trial.py)

```bash
PYTHONPATH=.:src:scripts/experimental python -m scripts.experimental.tt19_shoot_trial artifacts/tt19 verify
```
