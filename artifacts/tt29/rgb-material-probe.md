# Material probe using controlled RGB drops

Short vertical drops measured the paddle’s effective normal response. The arm held its horizontal solo pose while two fixed RGB cameras observed the descent and first rebound at 60 Hz. The estimator triangulated the orange ball and fitted the trajectories on either side of contact. It extrapolated both fits to their intersection and compared the vertical speeds there. A separate nominal station supplied a lookup from observed rebound ratio and approach speed to the planner’s effective damping parameter. The lookup used two drop heights at each of five known simulator settings. The held probes used three other heights. [Probe implementation](../../scripts/experimental/tt29_material_probe.py) · [Stereo measurements](rgb-material-probe.json.gz)

| Held profile | Paddle measurements | Median observed rebound ratio | Effective paddle damping |
| --- | ---: | ---: | ---: |
| Stock | 3 / 3 | 0.871 | 20.5 |
| Changed ball and contact materials | 3 / 3 | 0.841 | 26.0 |
| Mixed material, arm and camera changes | 3 / 3 | 0.839 | 26.0 |

Both changed profiles shifted the effective paddle estimate by 5.5 relative to stock. The mixed profile used a ball 8% lighter than stock, table damping of 38, paddle damping of 39 and paddle friction of 0.39. It also increased the paddle mass by 12%, elbow mass by 14%, and shifted and rotated the fixed cameras. The effective fit pools these effects and has an absolute offset even on stock. The table drops lacked continuous stereo around the trough, so the evidence supports a paddle adjustment while leaving table response and paddle friction unresolved. [Captured tracks and held settings](rgb-material-probe.json.gz)

## Live comparison

The paired point used measured opening 11292 under the mixed profile. Both branches registered the moved stereo views from the white court lines and started with the same 40 frozen stroke examples. Opening 11292 was absent from that checkpoint. In the adjusted branch, the private stroke planner added the measured 5.5-unit change to its standard paddle damping of 25. Both branches used the same coupled motor envelope, actuator rate cap and fast physical practice carriage. The comparison capped play at two alternating legal returns after the measured opening. [Camera registration](court-registration.md) · [Live action tapes](rgb-material-probe.json.gz)

Both branches selected the same 50 joint commands. The near paddle struck the ball into its own half and the referee awarded zero legal returns in both branches. The measured shift altered the planner’s model, while the selected stroke stayed the same in this point. [Native referee events](rgb-material-probe.json.gz)

Run the survey and the paired comparison using the pinned measured launch file.

```sh
export MUJOCO_GL=egl PYTHONPATH=.:src:scripts/experimental LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt29_material_probe --feeds /path/to/rallies.json --condition compound
```
