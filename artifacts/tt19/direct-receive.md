# Direct receiving transfer to the SO-101

With zero target practice, the gantry priors produced a first legal return on 17 of 24 fresh rival serves. The nominal model returned 15. Both took a median of 36 control frames from the start of a successful point to the legal landing. The sample leaves substantial uncertainty about the size of the receiving gain. [Paired tapes and precision estimates](direct-receive.json.gz)

| Outcome | Gantry prior, 95% Wilson interval | Nominal prior, 95% Wilson interval | Gantry only / nominal only | Exact paired p |
| --- | --- | --- | ---: | ---: |
| First legal return | 17/24 (51–85%) | 15/24 (43–79%) | 4 / 2 | .688 |
| Point won | 8/24 (18–53%) | 5/24 (9–40%) | 5 / 2 | .453 |
| Two legal returns | 3/24 (4–31%) | 2/24 (2–26%) | 2 / 1 | 1.000 |

The first successful return in the fixed schedule arrived after 114 elapsed control frames with gantry priors and 71 with nominal priors. Successful points in both branches had an interquartile range of 35–37 frames to the first legal landing. Thirteen starts succeeded in both branches, with a median paired timing difference of zero frames. The landed return rates were 70.8% and 62.5%, an observed difference of 8.3 percentage points. All 48 point tapes passed physical replay. [Receipts](direct-receive.json.gz)

This test pairs 24 new full-spread rival-service starts on the strict compact table. Each point starts from the factory-neutral arm with a fresh native wrist-plus-fixed RGB tracker. The published rollout player handles the entire point, including subsequent returns. Both branches use the same 2,048-movement factory catalogue and shortlist up to 32 candidates for physical simulation. The carried branch loads the gantry-trained flight law and racket impulse matrix. The nominal branch uses generic priors. The experiment uses zero SO-101 practice and target calibration frames, with the factory catalogue supplying all arm movements. [Trial](../../scripts/experimental/tt19_direct_receive.py) · [Published rollout player](../../scripts/experimental/tt19_model_rollout.py) · [Factory catalogue](factory-dry-catalog.npz)

The point seeds come from a frozen schedule and are disjoint from the previous game receipts. Execution order alternates. When both arms issue identical commands, the second branch receives the same native camera frames from their shared prefix. At the first command difference, each branch continues from its own physical trajectory. The stored tapes include the RGB-derived ball positions, actions, plan decisions and referee events. [Paired tapes and precision estimates](direct-receive.json.gz)

The individual percentages use 95% Wilson intervals. The paired test uses an exact two-sided binomial test on discordant starts. Frames-to-return are measured at the first legal landing in each point and summarized over successful starts. The simulated control clock runs at 30 Hz. The planner's process latency was measured separately. [Planning timings](fast-play.md)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt19_direct_receive \
  artifacts/tt19 artifacts/tt19/factory-dry-catalog.npz --starts 24 --workers 2
```
