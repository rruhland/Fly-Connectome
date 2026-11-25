# Can the original arm teach the canted arm an ordinary stroke?

Actual routine conversion declined on the canted body as its low-ball conversion improved. Simply rotating the ball into original arm coordinates does not rotate the table, gravity or the outgoing ball flight. This experiment instead carries the measured world-space paddle center, signed face and angular and linear velocity of an entire legal stock stroke into the canted body's joint geometry. The experiment retains the same six-joint SO-101 hardware, the supported TT84 mount, the standard table and carry, camera observations, and a continuously random reactive native opponent.

## Native teaching works on the ordinary motion it sees

Twelve release-disjoint stock lessons were picked from actually scored ordinary first deliveries before doing any retargeting. All were independently replayed. Canted preparation tried multiple signed in-plane poses, timing offsets and finite servo leads, using the actual same ordinary powered far strike and near bounce. Search used physics only for this private teaching. A mounted movement entered the camera and encoder library only if it physically landed legally, drew a subsequent real answer and passed CAD. Eleven of the twelve source strokes produced an earned canted routine movement. Release 11667 produced none in three tries. All 14 issued calibration tapes held the incoming delivery fixed and had no penetration over 18,651 native solver states.

The retargeted movement did not recreate every encounter exactly. Over the 11 earned counterparts, median actual face deviation was 9.64°, median blade-center separation was 63.49 mm and median difference in patch speed vector was 0.436 m/s. Native evidence supports transferable successful ordinary movements on those known releases, not identical poses or automatic generalization. The 11 successful movements yielded 94 post-bounce camera and encoder clocks. [Preassigned paired forks and lessons](protocol.json) · [Original physical strokes](source-strokes.json.gz) · [Canted physical calibration](transported-practice.json.gz) · [Paid motor library](routine-motions.npz)

## A short local readout did not rescue unseen ordinary misses

On a separate canted course, the live correction only uses previously paid motion, public stereo and own encoders, with small local world-patch shifts and an independent native optical twin. The earlier actor keeps priority whenever its search finds a robust plan. The correction never activates on an estimated low ball.

| Held same-delivery role | Existing actor | With waveform correction |
|---|---:|---:|
| Four previously missed routine deliveries | 0/4 | 0/4 |
| Two previously scored routine deliveries | 2/2 | 2/2 |
| Four low control deliveries | 0/4 | 0/4 |

On 11133 the correction issued one genuinely different legal, answered stroke. It did not improve the score. None of the four low copies changed a command or event, but all were original misses, so these controls establish nonintervention on these lows, not retained positive low ability. Among ten extra exact-state diagnostic trials of *unissued* candidates for the unseen ordinary misses, one actually touched and none scored. The rejection was therefore not merely a conservative camera vote. The transferred proposals usually failed to reach useful contact from the new preparation state. [Matched physical command and no-issue ceiling tapes](held-forks.json.gz)

## Normal-game comparison

The independent eight-release paired schedule was registered before held results. It excludes TT84, TT90, TT98 teacher and matched-fork releases. Both variants play the identical canted installation, standard reactive opponent and 48-point starting carry. Both can earn actual online movement credit. On each point the only optional difference is the ordinary waveform correction. [Frozen normal-game roster](games.protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt98_world_waveform harvest
python -m scripts.experimental.tt98_world_waveform learn
python -m scripts.experimental.tt98_world_waveform fit
python -m scripts.experimental.tt98_world_waveform held
python -m scripts.experimental.tt98_world_waveform normal
python -m scripts.experimental.tt98_world_waveform transport
python -m pytest -q tests/test_tt98_world_waveform.py
```
