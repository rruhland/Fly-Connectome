# Solo blade practice followed by early outgoing-flight feedback

The combination landed 17 of 144 held mount–launch pairs, tying outgoing feedback from the original posture. The solo posture with three binary-reward probes landed 18. Solo practice helped with the handle rotation and hurt with the canted blade on this contact-supported roster. [Paired trials](combined-held.json.gz)

| Starting posture and regulation practice | Handle, 72 | Canted, 72 | Total, 144 |
| --- | ---: | ---: | ---: |
| Original, frozen | 2 | 12 | 14 |
| Original, three reward probes | 4 | 12 | 16 |
| Original, three outgoing-flight probes | 5 | 12 | 17 |
| Solo, frozen | 8 | 8 | 16 |
| Solo, three reward probes | 9 | 9 | 18 |
| Solo, three outgoing-flight probes | 9 | 8 | 17 |

Adding outgoing-flight probes to the solo posture gained four returns and lost three relative to the frozen solo posture. It gained eight and lost eight relative to outgoing practice from the original posture. With the same solo posture and the same three training balls, the outgoing learner gained one return and lost two relative to reward only. [Held launch outcomes](combined-held.json.gz)

On handle donor 7032, the original posture landed zero of six. Outgoing practice alone landed three, solo practice alone landed five, and the combination landed six. On the canted blade, the original posture landed all six launches from that donor and every branch using the solo posture landed zero. Both blade conditions recorded paddle contact on all six launches for every policy. [Practice traces and held tapes](combined-held.json.gz)

## Procedure

One policy started from its original posture. The other moved that posture using the blade orientation estimated from sixteen visual solo drops. Each starting posture had a frozen control, a binary-reward learner and a learner that used binary reward plus the early stereo estimate of outgoing flight. The experiment ran on the regulation table in MuJoCo with the far arm fixed after the feed. [Experiment runner](../../scripts/experimental/tt28_combined.py)

The starting action came from the TT27 coverage archive. Before testing either blade, the fixture selected the closest camera signature among each donor’s historically contacting joint goals. A fixed seed enrolled six donors from the nearby group and six from the unrestricted group, using contact history for eligibility and no landing outcomes. The twelve donors qualify as contact-supported cases. Each was run with the same new launch variations under the handle and canted blade changes. The three practice launches and six held launches per donor had independent perturbations of launch position, velocity and spin. The RNG was separate from the earlier outgoing-flight course. [Prior action coverage](../tt27/action-coverage.json.gz) · [Early outgoing-flight method](outgoing.md)

Solo transfer uses the archived 16-drop camera-derived rotation for each blade and the existing arm-geometry fit. The fit seeks the nominal paddle center and face normal with the changed grip while respecting the joint bounds. The regulation probes are fixed wrist-roll offsets of 0 and ±0.15 radians. Both learners with the same starting posture see the same probes and their binary landing rewards. The outgoing learner also sees the projection from its own camera frames. It can select an intermediate offset, while the reward learner selects the smallest tested offset with a legal return. Both policies freeze before the different held balls arrive. The harness selects the corresponding solo checkpoint and supplies camera calibration, motion and flight models, the shared starting motion and the probe axis and order. The learners use no simulator ball state or backpropagation. [Solo dose course](dose.md) · [Joint posture transport](../../scripts/experimental/tt26_optical_transfer.py)

Each starting posture yielded 41 usable forecasts across 72 actual practice attempts. Replaying all 450 tested held feed, blade and joint-goal combinations reproduced their referee events. The 36 practice trajectories and 72 held trajectories were distinct and absent from the preceding outgoing experiment. [Raw results](combined-held.json.gz)

```bash
MUJOCO_GL=egl PYTHONPATH=.:src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt28_combined /path/to/rallies.json
```
