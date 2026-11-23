# Native Pong calibration with bounded replay

The 240-window estimator qualified one of the two tested drives. On eight new wrist drops for that drive, it averaged **8.125 consecutive rebounds**, compared with **8.25** for the stored full batch and **1.50** for the stale map. The bounded estimator and full batch each reached ten rebounds on four drops. The other drive failed the same source gate as the strictly recursive estimator.

| Hidden drive | Full batch map error | Single-pass recursive map error | Bounded replay map error |
|---|---:|---:|---:|
| 0 | 13.32% | 35.19% | 47.18% |
| 1 | 5.29% | 14.43% | 13.44% |

Every held encoder row improved over stale for both bounded fits. Drive 0 exceeded the predeclared 22% full-matrix error gate, so it received no wrist test. Matrix error used the evaluator's known physical wiring. The learner received only live requested rates and subsequent encoder values.

| Eight matched wrist drops on drive 1 | Mean consecutive | Ten or more | Longest |
|---|---:|---:|---:|
| Bounded replay | 8.125 | 4/8 | 16 |
| Stored full batch | 8.250 | 4/8 | 17 |
| Stale map | 1.500 | 0/8 | 3 |

These starts were separate from the strictly recursive trial. All 24 action tapes replayed exactly. The saved bank reproduced a live wrist trajectory and a native held point after reloading. Recalling the original identity map reproduced all six previously archived stock runs.

## Memory and fitting

Each stream played its original twelve reward-trained native points in the physical arm simulation. The fitter received events once in their natural order. It used an eight-tick causal window and retained a uniform random sample of up to 20 eligible windows from each point. The two streams produced 3,461 and 3,997 eligible windows and retained 240 each. The live run used the source seed and body recipe and never opened a raw historical source clip.

The retained numeric memory was **55,464 bytes**, including the live causal window. This excludes array object overhead, temporary fitting arrays and the frozen game policies. The local robust regression reused only those retained windows at the final point boundary. It used 585 five-by-five solves and took about 20–21 ms per drive on this machine. Collecting the source moves took 4.63 and 5.13 seconds of wall time. The two streams contained 3,545 and 4,081 physical control frames. The independent native holdout updated neither reservoir nor estimator.

## Continuing beyond twelve points

The reservoir rotates one cleared slot for each new point and always reads the retained points in chronological order. The saved point number and random generator state remain absolute. The original checkpoint layout still loads, and both old checkpoints reproduced their original first-twelve fitted coefficients and lag choices exactly.

A separate continuation check fed each intact and save/reloaded branch the same live actuator and encoder events from actual points 13 and 14. After point 13 one branch saved and reloaded its state. Both branches retained points 3–14 after the next point and stored exactly 240 windows. They matched every retained value, selected lag, and fitted coefficient. On actual points 15–18 their predictions and qualification decisions also matched exactly. None of these continuation points changed the sealed first-twelve estimates or their wrist results.

Reproduce with `PYTHONPATH=src MUJOCO_GL=egl python -m scripts.experimental.tt24_online_tip_reservoir` and use `--continuity` to check the original checkpoints through points 13 and 14. [Sealed source fits](online-tip-reservoir-source.json.gz) · [Held wrist traces](online-tip-reservoir-held.json.gz) · [Continuation evidence](online-tip-reservoir-continuity.json.gz) · [Strict recursive comparison](online-tip-report.md)
