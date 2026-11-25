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

## Physical yield and held qualification

| Comparison | Stock scores | Learned scores |
| --- | ---: | ---: |
| 40 exact practice forks, using training-only native search | 0 | 18 |
| 12 release-disjoint identical terminal offers, RGB and encoders only | 0 | 0 |
| 8 fixed identical ordinary hits, RGB and encoders only | 8 | 8 |

The native teacher evaluated 14,080 paths and proposed one physical check
per fork. All 40 checks preserved the exact original arrival. Fifteen safe
physical practice scores had been marked legal by the batch model. Three
were native false negatives. The 18 safe scores cover 17 openings, with
five left, eleven middle, and two right. Sixteen received a legal reply
after resolving the three censored continuations. The two right scores
were different ordinary offers on the same release and ended naturally
without a legal reply. The prior and rescued contact median radii were
75.0 and 74.5 mm. The search primarily found better strikes near the rim,
not a motion that worked by centering the blade. Exact launch-to-reply
replays independently found zero penetrations for the first middle and
left examples. [Matured exact tapes and public input keys](matured.json.gz)
[Readout with 16 distinct answered releases](readout.npz)

In the held comparison, five otherwise missed receives retained a
stock-protected plan. The school issued on one of the other seven, which
missed. The atlas control did not issue on any held miss and did not rescue
any. The school issued on none of the eight routine hits, while the atlas
issued once and still scored. Give the original successful tape back to
the RGB/encoder model and it recognizes only one of eight as nominally
legal and none as deployable under the motion uncertainty test. Failure
is already present before cross-release borrowing. The new school was not
promoted or tested in full games.
[Held terminal tapes](held-forks.json.gz) · [Routine hit tapes](routine-forks.json.gz) · [Recognition audit](optical-witnesses.json) · [Full report](report.json)

## Train a movement the cameras can recognize

The second teacher co-optimizes exact practice physics and an independently
initialized RGB/encoder shadow, including the same two input stress tests as
the pupil. It considers the first six answered practice examples with native
agreement, selected before inspecting their optical results. Both objectives
rank the same 192 candidates per case. Ranking for exact dynamics alone
finds six safe physical scores that the opponent answers, but only one of
the six chosen moves passes the RGB authorization. The joint objective
finds five simulations with authorization and all five actually score and
are answered without suffix CAD penetration. It does not find a qualifying
movement on the sixth case. [Fixed teaching cases](joint.protocol.json) · [Physical branch tapes](joint-teacher.json.gz)

Replacing these five paths in the 16-example readout preserves the same
keys, releases, and search cost. Both libraries get 32 optical proposals in
each of ten other, exactly matched physical offers excluded from the first
qualification. Neither rescues any of the six archived misses, and both keep
all four routine successes. The one issued routine action comes from a
shared, unreplaced path. Optical recognition and cross-offer transfer are
separate problems. The improved teacher helps the former on these known
origins but does not justify whole-game deployment. [Frozen new offers](joint-transfer.protocol.json) · [Matched transfer tapes](joint-transfer.json.gz) · [Joint readout](joint-readout.npz) · [Independent audit](joint-summary.json)

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
python -m scripts.experimental.tt57_motor_school_report --verify
python -m scripts.experimental.tt57_motor_school_joint_teacher plan
python -m scripts.experimental.tt57_motor_school_joint_teacher run
python -m scripts.experimental.tt57_motor_school_joint_transfer plan
python -m scripts.experimental.tt57_motor_school_joint_transfer fit
python -m scripts.experimental.tt57_motor_school_joint_transfer run
python -m scripts.experimental.tt57_motor_school_joint_transfer audit
python -m pytest -q tests/test_tt57_motor_school.py
```
