# An attained-body direct movement school

The experiment tests whether physically earned alternatives can be mined
from ordinary misses in enough distinct attained bodies to transfer beyond
an individually coached stroke. It neither moves the nominal SO101 base nor
changes the limits, blade, cameras, referee, or continuously randomized
reactive opponent.

The pinned roster contains 40 previously touched, unscored practice shots
on 32 independent measured releases. There are 20 middle, ten left-wing,
and ten right-wing cases. They cover multiple ages and prior rally lengths.
The first ordinary optical search in a fixed public position window defines
each candidate branch. Historical terminal results stratify the docket, not
the live actor. Twelve matched held forks use eight wholly separate release
IDs, excluded both from the school and the 16,081-window donor atlas.
[Protocol and source checksums](protocol.json)

The training coach restores the exact actual native dynamics from the
recorded stock commands. It combines short observed successful paths from
other releases, the actual failed path, time and joint-wise crossovers, and
smooth random perturbations. The native batch grades 128, 128, and 96 paths
and keeps varied useful contacts for refinement, using no backwards
gradients. Batch scores are advisory. A small shortlisted set is executed
against the actual referee and ordinary opponent from the same state. Each
real success must preserve the exact incoming far contact and near bounce.
Export only a scored response with no CAD penetration and a subsequent legal
opponent answer. If the 103-tick branch expires with a response still in
flight, continue the identical tape under the original opponent and longer
quiescent follow-through. Do not confuse a censored reply with no reply.
The coach may use privileged native state or an archived
failed command continuation only inside training. The movement readout
receives only prior stereo and encoder observations and the physically
issued direct control sequence.

The first eight practice cases were predeclared as a feasibility pilot.
Four got their exact original missed shot back and all four got an ordinary
opponent answer. Three were legal in both the native batch and actual trial.
The fourth scored on the actual referee after a net touch that the fast
batch cannot credit. The untouched stock commands failed all eight. This
is mining yield, not transfer. No held action selects on a private replay.
[Progressive practice receipt](practice.json.gz)

The held check compares school paths and original atlas paths with the same
maximum number of options, RGB/encoder-only twin test, and uncertainty
offsets. A supported stock move keeps priority. The additional eight
ordinary held hits are locked independently of any treatment, including
seven cases where stock did not have its robust prediction at the fork.
They distinguish failure to issue a correction from damaging a routinely
scored delivery. [Ordinary positive controls](routine.protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
# Initialize the byte-identical numbered checkpoint alias only when absent.
test -e artifacts/tt52_practice_dose/live-from-48-96.json.gz || ln -s live-from-48.json.gz artifacts/tt52_practice_dose/live-from-48-96.json.gz
python -m scripts.experimental.tt57_motor_school plan
python -m scripts.experimental.tt57_motor_school_transfer plan-routine
python -m scripts.experimental.tt57_motor_school run --n 8
python -m scripts.experimental.tt57_motor_school run --n 40
python -m scripts.experimental.tt57_motor_school_mature
python -m scripts.experimental.tt57_motor_school_transfer fit
python -m scripts.experimental.tt57_motor_school_transfer held
python -m scripts.experimental.tt57_motor_school_transfer routine
python -m scripts.experimental.tt57_motor_school_transfer audit-training --n 8
python -m scripts.experimental.tt57_motor_school_transfer games
python -m scripts.experimental.tt57_motor_school_report --verify
python -m pytest -q tests/test_tt57_motor_school.py
```
