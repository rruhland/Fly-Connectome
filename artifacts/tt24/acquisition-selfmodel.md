# Does a better paddle-angle model help the altered arm keep its first return?

On the existing two-minute stress course all six altered-arm drops qualified on their first return. Four died at 0.64, 1.12, 1.38 and 4.51 seconds after the launcher released the ball, with 1, 2, 3 and 9 qualified rebounds respectively. Exact early replay showed a systematic error in the inherited paddle-face estimate. On the first qualified contact the altered model underestimated the world (y) component of the upward normal by the equivalent of 3.9–6.2°. The estimated ball position during the first three seconds was typically within 2–7 mm median error. The original twelve trajectories supply this diagnosis only. No game observation or reward from them enters either calibration. [Original course](long-wrist.md) · [Exact early replay and metrology](acquisition-selfmodel-audit.json.gz)

## Independent contact-free calibration

Four 72-tick generic motor excursions per body supplied encoders and synchronized RGB from two fixed views. Two additional excursions per body assessed generalization. The ball stayed out of reach throughout. The fitter used no game rewards, moving simulator pose labels, desired paths or expert commands. It retains the carried link models and estimates small encoder-dependent corrections to the paddle's physical face orientation. The original carried wrist observer, camera-to-ball model and reward-trained motor checkpoint supply the remaining control. [Carried policy provenance](carried-wrist.md) · [Experiment code](../../scripts/experimental/tt24_acquisition_selfmodel.py)

The first route fit the unmarked paddle silhouette. On the altered arm its held projected width error went from 1.40 to 1.16 pixels, while independent physical face-normal error increased from 3.31 to 4.54°. It failed the predeployment geometry check and did not play a game. [Unmarked probes and frozen fit](acquisition-selfmodel-fit.json.gz)

The second route used three differently colored display sites in the known blade plane. The sites have no mass and create no collisions. Their two RGB views determine the plane direction by triangulation. The two held excursions per body are disjoint from the fit and from the game starts. The altered optically measured angle RMSE fell from 2.74 to 2.01°. It passed the geometry check, which required at least 10% improvement on held RGB-derived labels and 20% on held replay metrology. Simulator angles were accessed only to grade the already frozen fit. Angle errors in the table are RMS over the two reported horizontal components of the unit-normal vector, converted to component-equivalent degrees. [Marked probes and frozen fit](acquisition-selfmodel-marked.json.gz) · [Independent tool-response replay](acquisition-selfmodel-metrology.json.gz)

| Body | Physical angle RMS before → after | Physical angle-change RMS over 9 ticks |
| --- | ---: | ---: |
| Stock | 2.16° → 1.59° | 1.66° → 1.60° |
| Altered | 3.31° → 1.49° | 2.14° → 1.66° |

## Fresh physical launches

The matched course pairs 12 new altered launches and four stock retention launches. The launch protocol keeps widened joint starts, independently positioned held balls, 14–38 preparation ticks, randomized ball mass and drag, and two possible unseen gravity changes. The live cap is eight seconds after the dispenser lets go. One altered branch feeds the new absolute face estimate to the existing motor weights. A second cancels the measured source-side face bias from that estimate so the frozen weights receive the angle in their original stock sensory convention. The original wrist ball estimate, visual workspace reference, fixture schedule and motor head stay shared. Live control uses ordinary wrist RGB and encoders. The marked blade and fixed cameras are calibration equipment. Zero-command holding shares the same physical starts. [Paired actions and referee records](acquisition-selfmodel-held.json.gz)

| Frozen qualification on new altered starts | Total qualified | First | Second | Survived 5 s | Reached 8 s | Longest streak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original observer | 180 | 12/12 | 10/12 | 8/12 | 6/12 | 24 |
| Calibrated absolute normal | 190 | 12/12 | 11/12 | 7/12 | 5/12 | 29 |
| Calibrated to stock sensory convention | 126 | 12/12 | 10/12 | 4/12 | 2/12 | 26 |
| Zero command | 3 | 3/12 | 0/12 | 0/12 | 0/12 | 1 |

The absolute estimate turned three short frozen runs of one, three and two rebounds into 29, 23 and 27 qualifying rebounds. It also shortened four matched episodes from above to below the five-second mark. On total rebounds it won five, tied one and lost six. Across all new altered starts, greater geometric accuracy left median survival lower, 6.12 seconds versus 7.65 seconds with the original observer.

The correction activates only when the original wrist appearance selector recalls the alternate optical model. The same five encoder labels and radians are used for both body models. Applying the original unmodified stock branch on four retention drops yielded 88 qualified rebounds and three cap finishes. It uses a single copy of each stock action tape across the three actor conditions. All 56 distinct trajectories, including controls, reproduced their full original judge records and gravity events on replay. [Action tapes and scores](acquisition-selfmodel-held.json.gz)

## Reproduce

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_acquisition_selfmodel --audit /tmp/acquisition-audit.json.gz
python -m scripts.experimental.tt24_acquisition_selfmodel --calibrate /tmp/acquisition-outline.json.gz
python -m scripts.experimental.tt24_acquisition_selfmodel --marked /tmp/acquisition-marked.json.gz
python -m scripts.experimental.tt24_acquisition_selfmodel --metrology /tmp/acquisition-metrology.json.gz --fit /tmp/acquisition-marked.json.gz
python -m scripts.experimental.tt24_acquisition_selfmodel --evaluate /tmp/acquisition-held.json.gz --fit /tmp/acquisition-marked.json.gz
```
