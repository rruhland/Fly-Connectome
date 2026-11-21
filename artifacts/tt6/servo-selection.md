# Choosing a servo from the camera probes

`--auto-servo` measures the smallest singular value of the positional response fitted from stereo RGB and public motor commands. If a unit motor command can move the blade less than 0.8 blade radii in the weakest spatial direction, it uses the short-horizon controller with coherent camera feedback. Otherwise it uses the existing model predictive controller. The selector uses the same criterion for every installation.

The matched comparison re-rendered the original commissioning actions and reused the recorded starts and disturbance schedules from the two mixed-course receipts. Each solver started from the same mini-table experience and updated through each station’s points. The automatic column takes the result of the solver selected by the camera fit. The cutoff was explored using the original fits, so these results describe the selected starts.

| Body | Servo | Points with a legal return | Legal returns | Points with at least two legal returns |
| --- | --- | ---: | ---: | ---: |
| Gantry | Predictive or automatic | 5/7 | 7 | 2 |
| Gantry | Short | 5/7 | 7 | 1 |
| Offset | Predictive or automatic | 5/7 | 6 | 1 |
| Offset | Short | 4/7 | 7 | 2 |
| Rotary | Predictive | 1/14 | 1 | 0 |
| Rotary | Short or automatic | 5/14 | 5 | 0 |

The automatic policy returned balls in 15 of 28 points. The universal short and predictive controllers returned balls in 14 and 11 of 28 points respectively. On the eight points combining changed rubber, command delay and command noise, automatic and short each returned balls in three points, and predictive returned balls in two. The short controller produced one more total legal return than automatic selection. All three policies recorded zero multiple-return rotary points in this comparison. [Matched receipt](servo-selection.json.gz)

The deployed selector also played an independent course while carrying experience across three new installations. It returned one ball during seven points on the new rotary installation, where repeated returns remain an open problem. [Live course](servo-selection-live.json.gz)

| Body | Chosen servo | Points with a legal return | Legal returns | Points with at least two legal returns |
| --- | --- | ---: | ---: | ---: |
| Gantry | Predictive | 3/7 | 3 | 0 |
| Offset | Predictive | 4/7 | 5 | 1 |
| Rotary | Short | 1/7 | 1 | 0 |

Reloading the checkpoint at the third installation reproduced the selected solver, command checksum, score and exact physics event record for the first rotary point. [Checkpoint after the offset visit](servo-selection-checkpoint.json.gz)

On the same seven rotary starts with the same checkpoint, the predictive controller also landed one legal return. It won that point. [Matched continuation with predictive control](servo-selection-live-predictive.json.gz)

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt6_select_trial /tmp/servo-selection.json.gz
python -m scripts.experimental.tt6_continuous_trial /tmp/servo-selection-live.json.gz \
  --seed 24581214 --course gantry offset rotary --auto-servo --cached-render --reflections \
  --checkpoints /tmp/tt6-auto-checkpoints
python -m scripts.experimental.tt6_continuous_trial /tmp/servo-selection-resumed.json.gz \
  --seed 24581214 --course gantry offset rotary --auto-servo --cached-render --reflections \
  --start-visit 2 --resume artifacts/tt6/servo-selection-checkpoint.json.gz --phases ordinary
```

The carried state contains the verified motor maps, return-choice and bounce experience, stroke-response adaptation and actuator timing belief. At the next installation the policy validates an old map or fits a new one from the camera probes, then applies the selection rule again.
