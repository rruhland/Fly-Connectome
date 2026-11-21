# Shot memory conditioned on the observed motor response

The shot selector keeps point outcomes indexed by an anonymous signature of the camera-fitted motor model. Each signature contains the positional Gram matrices after 2, 5 and 8 frames and after the actuator response settles. The matrices are invariant to permutations and sign changes in the commands. The Gaussian weighting bandwidth comes from the median closest distinct practice signature. Four zero-reward observations per shot shrink estimates with little local evidence toward zero.

The training input contains the 84 public shot indices and point rewards from the two varied practice courses. At each held-out point the selector takes the shot with the largest locally weighted reward estimate and adds the resulting public reward to its memory. It carries the 84-point checkpoint’s body models, stroke response, latency and other state. Both policies continue learning through the held-out points. [Portable shot memory](local-shot-memory.json.gz) · [First 42 practice points](combined-carry.json.gz) · [Second 42 practice points](additional-practice.json.gz)

| Evaluation seed | 84-point checkpoint | Points with a legal return | Legal returns | Points with at least two legal returns | Wins | Capped points |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 54782309 | Global shot counts | 16/21 | 22 | 5 | 9 | 3 |
| 54782309 | Shot memory by motor response | 16/21 | 20 | 2 | 7 | 2 |
| 10932577 | Global shot counts | 11/21 | 16 | 4 | 3 | 4 |
| 10932577 | Shot memory by motor response | 12/21 | 20 | 6 | 7 | 5 |
| 39275001 | Global shot counts | 16/21 | 24 | 5 | 9 | 2 |
| 39275001 | Shot memory by motor response | 14/21 | 17 | 2 | 4 | 2 |

On the two primary courses, local shot memory raised opening returns from 27/42 to 28/42, below the 31/42 reached by the checkpoint after 42 practice points. Across all three courses, it returned 42/63 openings versus 43/63 with global shot counts, with fewer total legal returns and sustained rallies. The local policy returned 15/18 of its first two starts at each station versus 13/18 for the global policy. It returned 11/18 points with combined disturbances versus 13/18. Both policies paid the same calibration budget and fitted identical motor maps at every installation. [Primary A global](forward-extended.json.gz) · [Primary A local](forward-local-shot.json.gz) · [Primary B global](forward-repeat-extended.json.gz) · [Primary B local](forward-repeat-local-shot.json.gz) · [Independent global](newcourse-last.json.gz) · [Independent local](newcourse-local-shot.json.gz)

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt6_shot_memory /tmp/local-shot-memory.json.gz \
  artifacts/tt6/combined-carry.json.gz artifacts/tt6/additional-practice.json.gz
python -m scripts.experimental.tt6_continuous_trial /tmp/local-shot.json.gz \
  --seed 54782309 --course offset gantry offset-b --predictive --cached-render --reflections \
  --resume artifacts/tt6/continual-player-84.json.gz --shot-memory /tmp/local-shot-memory.json.gz
```

The matched continuation commands for the other courses use `--seed 10932577 --course gantry offset gantry-b` and `--seed 39275001 --course offset gantry offset-b`. `--checkpoints` saves the evolving shot memory alongside the other player state. The next resume reads the embedded shot memory directly.
