# Camera-guided corrections during a recalled swing

The SO-101 changed a previously selected stroke when a late stereo estimate placed the incoming ball elsewhere after the table bounce. On a fresh wide-target schedule, one opening improved from 2 to 22 consecutive legal returns. Across three openings, the same change brought the total from 34 to 54. Both runs began with the same 223 rewarded movements and used the same recorded openings, camera cadence, motor torque curve and opponent seeds. [Paired measurements](motor-correction-held.json)

The isolated model recorded the selected stroke’s predicted impact time and depth. Fresh RGB frames revised when the ball would cross that depth and where it would be laterally and vertically. The controller sampled the remaining joint targets at fractional time offsets and used the paddle’s kinematic Jacobian to translate its face while limiting the change in orientation. It rolled out the original suffix and up to four edited suffixes from current encoders and commanded targets. A correction had to improve the predicted return and pass the existing speed and clearance checks. Legal returns stored the complete issued target sequence for future recall. [Implementation](../../scripts/experimental/tt29_motor_correction.py)

| Schedule | Openings in order | Control returns | Edited returns | Control and edit CPU time |
| --- | --- | --- | --- | --- |
| Original wider opponent, seed offset 62470 | 7271, 7345, 11292, 4187 | 22, 2, 8, 6 | 46, 6, 2, 2 | 80.8 s and 106.7 s |
| Fresh schedule, seed offset 73529 | 7345, 11292, 7637 | 2, 32, 0 | 22, 32, 0 | 52.6 s and 156.8 s |

The fresh run stopped at 32 returns on 11292 in both conditions. The two 7345 command streams agreed through tick 109. The first edit at tick 110 changed the second near-side stroke. After that intervention the two rallies faced different physical returns. The edited point installed five changes before it ended. Its replay reproduced all 116 contact and referee events and found at least 18.6 mm clearance at 7,905 full-CAD states. The fastest stroke joint reached 3.48 rad/s, and all 39,520 signed joint samples satisfied the motor model. [Wide control](envelope-variety.json.gz) · [Wide edited tapes](motor-correction-wide.json.gz) · [Fresh 22-return tape](motor-correction-7345.json.gz) · [Full-CAD replay](motor-correction-7345-audit.json) · [Signed motor audit](motor-correction-7345-motors.json)

An exploratory version stored the original requested target prefix and the edited suffix. Its first wider point reached 22. The tested implementation reconstructs targets from the issued rates for the complete stroke. On the first wider point the two implementations issued identical commands through tick 179, then selected different strokes after the memory update. The fresh paired comparison used the tested command reconstruction in both conditions. Its control performed the same correction checks with edits disabled. Model checks consumed more CPU on the fresh schedule, and two original schedule openings lost returns.

Repeat the fresh test with the pinned feed file and the same checkpoint. Run once with `--mode edit` and once with `--mode observe`.

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 \
python -m scripts.experimental.tt29_motor_correction /tmp/tt29-feeds.json /tmp/tt29-motor-edit.json.gz \
  --checkpoint artifacts/tt29/envelope-carry.json.gz --ids 7345 11292 7637 \
  --seed-offset 73529 --return-target 32 --mode edit
```

An optional `--optical-checkpoint` applies previously measured outgoing error to scoring while keeping those samples fixed. The optical recorder uses the selected edited suffix for the corresponding outgoing RGB trace.
