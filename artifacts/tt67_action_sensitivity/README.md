# A local direct-joint response to changed incoming flight

The action-sensitivity readout landed **10 of 18** newly drawn physical replies versus **9 of 18** under time-aligned replay. Both modes got nine legal opponent answers. The readout recovered three otherwise missed shots and spoiled two otherwise legal shots. It preserved all six unshifted scored controls. This is local transfer around a familiar receiving stroke, not a new autonomous rally policy. [Frozen experiment](protocol.json) · [Physical scouts](scouts.json.gz) · [Issued tapes](held.json.gz) · [Paired outcome report](report.json)

The clearest mechanism witness is 5463. The new opponent reply moved 55.2 mm sideways and 22.5 mm down in the physical receive-line measure. Time-aligned replay touched at radius 74.5 mm and failed to score. The learned correction touched at radius 45.7 mm, returned legally, and the ordinary opponent answered. Its far-paddle contact and near bounce were identical to replay. The two other rescues moved the physical contact from 66.0 to 58.3 mm and from 74.6 to 66.3 mm.

| Source group | Replay legal / trials | Sensitivity legal / trials | Replay answered | Sensitivity answered |
| --- | ---: | ---: | ---: | ---: |
| Unshifted original controls | 6/6 | 6/6 | 6 | 6 |
| New physical replies across all sources | 9/18 | 10/18 | 9 | 9 |
| New replies at the two well-conditioned sources | 3/6 | 5/6 | 3 | 5 |
| New replies at the other four sources | 6/12 | 5/12 | 6 | 4 |

All scored branches repeated their referee events under CAD observation. The 21,985 sampled states had positive clearance, minimum 15.66 mm.

## What the transport law learned

Take six ordinary, scored and answered strokes with inner-face contact, three from relatively low source flights. At a pre-impact clock, issue each of five smooth joint offsets in both directions and estimate the motion of the blade center and normal. One exact source world and 18 nearby worlds per stroke expose sensitivity to joint position and velocity perturbations. There are 114 native worlds total and 1,254 nominal and paired-offset probes. No contact reward is differentiated, no backpropagation is used, and this source-only teacher never scores a held candidate. [Native fits](fit.json.gz)

The actor starts the same RGB observation clock in both arms and waits for the same public after-bounce optical gate. It extrapolates the incoming displacement from the current visible track to the source stroke's contact plane, compares the measured encoder/reference error, and solves the damped five-joint inverse of that learned action response. The outgoing commands stay inside ordinary actuator slew and joint limits. Both arms use the same optical time warp and source movement. Only the sensitivity arm adds the joint correction. Neither receives a held target, referee position, simulator pose, future image, or private bounce time.

The difference between sources matters. At 5463 and 5349 the weaker singular response in the lateral and vertical plane was .193 and .128 m/rad. The remaining four were .034 m/rad or lower at their attained, generally later body states. The standalone geometric distinction was not a decision gate in the primary experiment. It is an exploratory analysis of the primary results and a frozen source criterion in the wider follow-up.

## Wider independently drawn replies

The separately frozen wide check kept both sources whose minimum lateral/vertical response exceeded .05 m/rad, without changing their learned policy. Four additional, previously unused opponent seeds differed by 74.8–95.3 mm in the physical incoming measure. Replay made **no contacts**. Sensitivity made **three contacts**, landed one return, and received a legal answer from the original opponent. The 74.8 mm winner on 5463 touched at a 59 mm radius. Both modes still scored and got an answer on the two original controls. The evidence is useful for bounded practice and leaves wider cross-release transfer and full games open. [Wide protocol](wide/protocol.json) · [Issued wide tapes](wide/held.json.gz) · [Wide results](wide/report.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt67_action_sensitivity plan
python -m scripts.experimental.tt67_action_sensitivity scout
python -m scripts.experimental.tt67_action_sensitivity fit
python -m scripts.experimental.tt67_action_sensitivity held
python -m scripts.experimental.tt67_sensitivity_report
python -m scripts.experimental.tt67_sensitivity_wide plan
python -m scripts.experimental.tt67_sensitivity_wide run
python -m pytest -q tests/test_tt67_action_sensitivity.py
```
