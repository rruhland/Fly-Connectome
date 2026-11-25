# Does better private state initialization put 100 returns within reach?

Exact initialization did not make 100 returns look attainable for this controller. Across four selected physical points, ordinary frozen control landed **49** arm returns and exact initialization of all tested fields landed **22**. On the two strongest frozen controls, it shortened the point from **24 to 11** and **17 to 9**. Exact near-arm and paddle state did help one point, from 17 to 25, so the component results are not uniformly negative. [Matched physical scoring](audit.json)

This is a diagnostic, not a deployable player or a training source. Each condition runs one complete physical point on the same table-clamped SO-101 and randomized finite-mass opponent. Only initialization of the near controller's separate hypothetical simulator can change. The real plant, camera cadence and images, raw pose field, search gate, recalled direct joint sequences, robustness checks, action governors and opponent aim stream remain stock. The first physical offer and pre-intervention events are audited exactly. Once different sequences are issued, later *realized* offers can diverge even though the random target sequence is unchanged. [Isolation and replay code](../../scripts/experimental/tt42_private_init.py)

The experiment restores each release's exact previously earned memory from the checked carried checkpoint. Within-point credit is disabled in every arm, including stock. This avoids granting a motor update to privileged motion and is why 14390's stock comparison is **17 frozen**, not the earlier **35 with within-point updates**. On 8510 and 12109 we physically reran stock from the same memory. The long stock controls are existing complete frozen physical tapes, separately identified and replayable from commands. Nothing from these diagnostic runs is used as a policy checkpoint. [Earlier frozen and online pairing](../tt40/matched-within.json.gz) · [Original 35-return witness](../tt39/carried-35.json)

## Earlier evidence and what this measures

The earlier TT37 held-commands audit found 170 of 384 truly legal candidate motions recognized under optical state, 325 with true p/v, and 384 after adding spin. That test used many correlated actions at fixed physical decisions. It did not show the altered planner playing complete points. An optical spin predictor also improved mean departure prediction on some fixed TT39 commanded actions, without improving contact classification, and its four-point geometric-mentor live pilot fell from seven returns to four. The larger early-search oracle teacher reached 47 after a matched long prefix, but changed the proposal family and search budget as well. [Fixed-command substitution audit](../tt37/carry.md) · [Optical spin and angular controls](../tt39/spin.md) · [Teacher evidence](../tt38/motorcoach.md)

Here p/v means true ball position and translation velocity, with stock optical rebound and spin. Spin means the exact world angular velocity, including the component along the trajectory. Phase means the actual post-table-bounce referee state, in the private simulator only. Body means the true near joint position and velocity and grasped paddle pose/rate, with stock issued references. Ball combines p/v, spin and phase. All adds the truth body state. No true paddle or ball state is installed in the live plant, no future samples are supplied, and no opponent state or aim is written into the near model. **Even the all condition retains stock optical scheduling and the same robustness perturbations.** It tests a narrower, actionable question about improving the current private initialization, not the performance of every possible perfect-vision actor.

## Scores and failure mechanisms

| Release | Frozen stock | Exact p/v | Exact spin | Exact body | Exact phase | Exact ball | Exact all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12109 | 8 | 4 | 4 | 6 | 8 | 2 | 2 |
| 14390 | 17 | 13 | 18 | 25 | — | — | 9 |
| 5098 | 24 | — | — | — | — | — | 11 |

Each score is for a full physical point from the measured launch and ends naturally. The scheduled 14390 ball-only cell did not finish and is not counted. All completed conditions shared the exact first incoming far contact and near bounce. Physical event prefixes matched through the first different command, and every overlapping continuous opponent draw was unchanged. No row added any motor memory. [Both short points](short.json.gz) · [Long 14390 arms](point-long.json.gz) · [Peak frozen 5098 pair](peak-frozen.json.gz) · [Merged audit](audit.json)

On 8510 the full oracle and all component arms missed the same first delivery without a blade touch. Stock position error was only 1.1–2.2 mm and velocity error was no greater than 0.035 m/s at four search decisions. On 12109, stock median position and velocity errors were 1.22 mm and 0.022 m/s. The phase flag was already correct on all 34 stock decisions on that point. Exact phase reproduced every command and event on both short points. The 14390 all-state arm lost after nine returns. At four exact-state searches it found no legal option for the next low ball, which physically hit the wrist-camera foot instead of the blade. These are failures of the tested movement search, not proofs that no feasible action exists. [Per-decision residuals and actual events](audit.json)

The stock latent spin estimate was substantially less accurate than translation, with a 45.8 rad/s median error on 12109. Nevertheless, replacing spin with the exact complete vector gave 22 arm returns versus 25 across the three matched spin controls. This does not justify a marked-ball or extra-camera-rate spin experiment as the next route to 100. Exact body alone improved 14390, but reduced 12109 and did not touch 8510. It is not yet a general result for a new body observer. The separate real early-rebound readout is worth keeping as a research option, despite its 104 versus 109 online carried result. The sharper next comparison is between getting a legal movement into the candidate set and committing it through the landing. [Causal RGB rebound control](../tt41/early-optics.md)

Independent command-only replay reproduced all seven 8510 histories. The two 5098 arms and the 14390 body and all-state conditions also replayed every event exactly. Those four full CAD checks covered 56,063 physics configurations with zero penetrations. [5098 CAD](peak-audit.json) · [14390 CAD and score verification](audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt42_private_init
python -m scripts.experimental.tt42_private_init run "$out/point-long.json.gz" --ids 14390 --modes stock all pv spin body --archived-stock
python -m scripts.experimental.tt42_private_init run "$out/short.json.gz" --ids 8510 12109 --modes stock pv spin body all ball phase
python -m scripts.experimental.tt42_private_init run "$out/peak-frozen.json.gz" --ids 5098 --modes stock all --archived-stock
python -m scripts.experimental.tt42_private_init verify "$out/audit.json" --sources "$out/point-long.json.gz" "$out/short.json.gz" "$out/peak-frozen.json.gz"
python -m pytest -q tests/test_tt42_private_init.py
```
