# Anonymous wrist-camera vision

An appearance-and-motion tracker replaced the fixed hue masks in the same frozen wrist-camera controller for 14 paired held drops. The motor weights and joint-response readout came from `motor-rate-continued.json.gz`. That checkpoint was trained with the original hue masks and target reward. The anonymous tracker starts empty at each release and adds no motor practice. The frozen controller uses absolute mover coordinates, angular size and their history. The anchor affects relative retinal control and visual-consequence credit when those options are enabled. Neither was enabled in this frozen evaluation.

The two vision front ends saw 64 × 48 RGB sampled at 30 Hz. For the blind and passive controls, the simulator supplied joint observations without rendering RGB. Localization accuracy is the share of visible samples with a returned centroid within two native-resolution pixels of the renderer's separate segmentation pass. An object was visible for the audit if at least three labeled pixels were present. Segmentation IDs and scene state never entered the action or the detector. Each rebound had to satisfy the same physical judge, including a 4.5 cm rise.

| Condition | Qualified rebounds per drop | Mover localized | Blade localized |
|---|---:|---:|---:|
| Passive | 2.50 | | |
| Blind | 2.43 | | |
| Native RGB, fixed hue | 3.36 | 99.4% | 100.0% |
| Native RGB, anonymous | 3.79 | 83.3% | 50.1% |
| Native RGB, anonymous without action input | 3.64 | 84.1% | 48.9% |
| Cyclic RGB permutation, fixed hue | 2.43 | 0.0% | 0.0% |
| Cyclic RGB permutation, anonymous | 3.79 | 83.3% | 50.1% |
| Violet mover and green blade, fixed hue | 2.43 | 0.0% | 0.0% |
| Violet mover and green blade, anonymous | 3.71 | 83.5% | 50.4% |
| Reoriented wrist lens, fixed hue | 2.71 | 98.6% | 100.0% |
| Reoriented wrist lens, anonymous | 3.29 | 81.0% | 55.4% |
| External camera, anonymous on wrist-action replay | | 88.6% | 22.8% |

The anonymous tracker produced byte-identical actions on all 14 drops under the tested RGB permutation. With the simulated materials repainted, it beat the fixed hue controller on 10 drops and tied on four. In native RGB it beat the blind controller on 10 drops and tied on four. Relative to the fixed hue controller in native RGB, it won six, tied four and lost four. It reached eight rebounds on two drops and missed a drop that yielded three with fixed hue. On the rotated wrist lens, changed by 6° and −4° around two local axes, it beat fixed hue on five drops, tied on seven and lost on two. The 128 × 96 external camera assessment replayed native wrist actions through the same physical trajectories and measured perception only.

The front end groups connected pixels around chromatic modes measured in the current image. It associates successive regions by appearance, displacement and measured change in extent. A robust estimate of shared retinal drift and a small action-to-displacement regression discount collective and self-predicted motion when ranking a mover. For the anchor, persistent contrast, stable or action-predicted position, and proximity to abrupt turns in the mover's visible path add evidence. The front end has generic chroma and noise floors, continuity assumptions and one mover/anchor output pair. It needs sufficiently distinct connected regions and can confuse the blade with the gripper or background, especially before a visible turn or from the external viewpoint. Removing action input produced identical commands on 13 of 14 drops, so the main evidence here is for appearance and motion grouping. The detector averaged 3.24 ms per native RGB sample on one CPU thread, excluding rendering and the audit.

Run the paired held evaluation from the repository root.

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_anonvision_trial
```

`tt24_bounce_trial --vision anonymous` also runs the detector in reward-only target practice and visual-consequence credit experiments. The first-frame contextual portfolio requires immediate identities, so it uses the fixed hue detector.

- [Tracker](../../scripts/experimental/tt24_anonvision.py)
- [Experiment](../../scripts/experimental/tt24_anonvision_trial.py)
- [Per-drop results](tt24_anonvision.json.gz)
- [Frozen checkpoint](motor-rate-continued.json.gz)
