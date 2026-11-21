# Physical arm transfer

Every fixture uses the same regulation table, ball, scripted opponent and stereo viewpoint. The four-channel gantry moves on three slides and tilts its blade on a wrist hinge. The five-channel offset receiver moves on three oblique slides, turns a tilted sweep hinge and tilts the wrist. A passive coupling counter-rotates the blade as the sweep joint moves. Both use ordinary red or black rubber and MuJoCo racket contacts.

The player receives two RGB images, the previous public command and the point reward. The evaluator chooses an actuator permutation and signs for each installation. The player fits the local response from six camera probes using stereo face positions and pitch. Each fitted motor channel has a settling curve so the player can account for onset, overshoot and inertia. The controller uses these responses to turn position and pitch targets from the shared stroke planner into bounded commands.

The practice curriculum starts with the first-person mini-table checkpoint. After seven stock-arm points, two independent branches practice seven points each on the gantry and offset fixtures. The held-out gantry receives experience from the stock and offset branches. The held-out offset receiver receives stock and gantry experience. The shared branch carries reward and timing statistics, observed bounce history and a prior on settling time. Each evaluation arm gets its own six RGB motor probes. The mini-only and fresh controls get those same probes and point seeds. The reversed-map control retains practice and reverses the fitted signs. The sweep intervention retains the fit and practice and fixes the command to the physical sweep joint at zero.

| Practice body | Points with a legal return | Legal returns | Points with at least two legal returns |
| --- | ---: | ---: | ---: |
| Stock | 7/7 | 25 | 5 |
| Gantry | 4/7 | 7 | 1 |
| Offset | 4/7 | 5 | 1 |

Each held-out body was tested at two installations with seven points per condition at each. Median position fit residuals on the RGB probe tapes ranged from 1.7 to 2.0 cm.

| Held-out body | Starting experience or intervention | Points with a legal return | Legal returns | Points with at least two legal returns |
| --- | --- | ---: | ---: | ---: |
| Gantry | Shared | 14/14 | 17 | 3 |
| Gantry | Mini only | 14/14 | 17 | 2 |
| Gantry | Fresh | 13/14 | 17 | 2 |
| Gantry | Reversed motor map | 0/14 | 0 | 0 |
| Offset | Shared | 13/14 | 22 | 6 |
| Offset | Mini only | 13/14 | 33 | 7 |
| Offset | Fresh | 11/14 | 25 | 6 |
| Offset | Reversed motor map | 0/14 | 0 | 0 |
| Offset | Sweep disabled | 11/14 | 19 | 3 |

The fitted sweep channel received commands above 0.05 in magnitude on 1,016 frames. Disabling it reduced returns at the first offset installation. The second produced equal return and sustained point counts with and without sweep. Earlier cross-body practice produced mixed results. The shared and mini-only conditions returned balls in the same number of held-out points, and mini-only produced more total returns on the offset body.

The complete receipt records probe costs, wiring, public commands, fit statistics, point scores, fault reasons and physical events. Replaying the 147 command tapes reproduced all 187 legal returns and the exact physics event records. The cached Hall pass includes live reflections and was checked against the original MuJoCo shaded pass on all three fixtures. [Trial receipt](body-transfer.json.gz) · [Renderer check](render-benchmark.json)

The offset replay uses the original live MuJoCo pass. Its recorded commands reproduce the 202-frame point and every logged contact event. The arm and opponent each land three legal returns. [First-person offset rally](body-offset-rally.mp4)

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt6_body_trial /tmp/body-transfer.json.gz \
  --seed 92056261 --practice-seed 86117024 --installations 2 --points 7 \
  --practice-points 7 --cache /tmp/tt6-body-cache
python -m scripts.experimental.tt6_body_replay artifacts/tt6/body-transfer.json.gz \
  /tmp/body-offset-rally.mp4 --course 1 --point 1 --condition shared
```
