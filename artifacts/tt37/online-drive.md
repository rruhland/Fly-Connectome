# Learning timing and swing drive from weak contacts

The controller learned a legal reply on one repeated weak incoming flight from trial outcomes measured in live play. Release 6208 scored zero with the inherited movement, and all four physical exploration points on that opening also scored zero. After fitting the observed outcomes, the frozen learner returned that first ball under two new continuation seeds. It used the timing and drive pair selected by a smooth response fit, not a replay of one successful stroke. [Practice command tapes](online-drive-practice-learn.json.gz) · [Frozen learned continuation tapes](online-drive-continuations-carry.json.gz) · [Inherited movement control](online-drive-continuations-known.json.gz)

This is a **geometric-scaffold experiment**, separate from the pure reactive TT37 tests. The receiving controller uses the clinic's connected joint posture surface, calibrated RGB, and an estimate of where and when the ball will reach a depth of −1.56 m. It can recall an inherited joint correction from the archived clinic. No candidate physics simulator, private motor search, or exact ball state is used by the learning or evaluating near arm. The actual physical opponent draws new placement goals independently on each live reception. [Scaffold implementation](../../scripts/experimental/tt37_motor_clinic.py) · [Two-parameter learning](../../scripts/experimental/tt37_online_drive.py) · [Pinned draws and releases](online-drive-plan.json)

On each chosen swing, the learner varies lead time by at most 55 ms and the geometric swing motion magnitude by at most 65%. Six matched nominal points started with no correction. A second six-point pass used the frozen inherited movement and produced the first outcomes the learner could use as references. During 24 additional real practice points, the learner selected bounded continuous probes around promising settings, then updated a joint sensory-context and action kernel after each point. A score was worth one and crowded out the smaller optical values. Nonlegal, visibly deflected balls supplied signed forward and vertical velocity and, when available, a stereo table-height cue. If the observer had no usable deflection and the shot did not score, the learner did not invent a label. The table-height cue is not an independent legal landing detector. [Nominal practice control](online-drive-practice-nominal.json.gz) · [Inherited practice control](online-drive-practice-known.json.gz) · [Retained physical experience](online-drive-checkpoint.json)

A matched native event read confirms that the forward-drive change was physical. On 11139 the inherited first shot separated at +0.71 m/s toward the opponent. Four attempted settings separated at +1.38 to +1.90 m/s, and the frozen selected setting also improved the viewed forward motion. These shots still scored zero. On 6208 the frozen chosen lead and drive changes were −27.5 ms and +16.25%. The RGB-signed forward velocity stayed near 3.3 m/s, while the vertical velocity moved from −1.23 to +2.95 m/s and the lateral drift fell from +1.51 to approximately zero. The inherited stroke bounced on the near half at y = −1.25 m. The learned stroke bounced legally at y = +0.26 m. Both new continuations physically reached the learner but it did not make a second legal return on this family. [Held command and optical outcome comparison](online-drive-results.json)

| Frozen held course | Inherited movement | With learned timing and drive |
| --- | ---: | ---: |
| 12 new continuations after trained openings | 5 | 7 |
| First 12 archive-disjoint TT37 openings | 0 | 0 |

The held gain was confined to the newly legal first stroke on 6208. As a matched ablation we replaced the 24-point practice archive's optical values by the binary referee score, leaving the observations used, tested parameters and fitting code unchanged. That control scored zero on both of the same 6208 continuations. The experiment identifies a contribution from graded evidence on this opening and with this fitting rule. It does not establish transfer to new incoming trajectories or sustained broad rallies. The 24 live training points took 60.0 process CPU seconds. The two new scoring points and one unchanged point replayed from commands alone, with zero penetrations across 4,096 solver states and 15.8 mm minimum clearance. [Binary-only control](online-drive-continuations-binary.json.gz) · [Physical command replays](online-drive-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_online_drive plan
python -m scripts.experimental.tt37_online_drive practice nominal --limit 6
python -m scripts.experimental.tt37_online_drive practice known --limit 6
python -m scripts.experimental.tt37_online_drive practice learn
for mode in known carry; do
  python -m scripts.experimental.tt37_online_drive continuations "$mode"
  python -m scripts.experimental.tt37_online_drive new "$mode" --limit 12
done
python -m scripts.experimental.tt37_online_drive continuations binary --ids 6208
python -m scripts.experimental.tt37_online_drive summary
python -m scripts.experimental.tt37_online_drive audit
```
