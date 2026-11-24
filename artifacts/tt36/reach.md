# Online visual body feedback for reaching the ball

The aligned body correction made **22 opening contacts on 32 new releases**, versus 19 for the raw joint policy and 12 with the same correction sign reversed. It landed 14 legal replies versus 11 for raw and 9 for reversed. The four fresh contacts broadened physical reach on both lateral sides. They did not themselves land legally. [Paired physical results](reach-results.json) · [Controller](../../scripts/experimental/tt36_reach.py)

## What the controller estimates

The feedback layer started with an encoder-to-blade map learned from 1,063 stereo observations of the red blade during scoreless play. At the current measured arm posture, a weighted local regression gave the blade center and a Jacobian for small joint changes. When the live stereo blade agreed with the body model, the controller used it to correct the estimated center.

A separate set of 453 single-frame observations from rewarded teacher strikes recorded where the ball was relative to the blade. At each update, the controller interpolated the ball–blade offset from examples with similar current ball depth, velocity and arm posture. It subtracted that displacement from the current observed ball position and mapped the resulting alignment error into joint angles with a damped local inverse. The correction blended in when the raw pupil had weak sensory support. It was recomputed each tick, using the observed ball at its current position. The examples supplied relative positions and contained no action sequences. The earlier spatial-translation test used this body model to augment demonstration data. Here the learned body Jacobian affects each live command directly. [Reward-free blade observations](../tt34/lag-bank.npz) · [Rewarded observation receipt](reach-observations.json) · [Framewise relative positions](reach-observations.npz)

On 18 separate practice releases, the tested 0.38 rad cap landed seven legal returns, the 0.65 rad cap landed six, and the nominal policy landed six. The 0.38 rad cap was frozen for every held run. Reversal kept the same support gate, sense inputs and bound, and inverted the computed joint correction. [Frozen checkpoint](reach-checkpoint.json) · [0.38 rad practice](reach-practice-0.json.gz) · [0.65 rad practice](reach-practice-1.json.gz)

## Physical play

| Roster and policy | Opening contacts | All near legal returns | Later near legal returns | Alternating legal returns |
| --- | ---: | ---: | ---: | ---: |
| 32 new releases, raw pupil | 19 | 11 | 3 | 54 |
| 32 new releases, aligned reach | 22 | 14 | 3 | 59 |
| 32 new releases, reversed reach | 12 | 9 | 3 | 50 |
| 16 changed continuations, raw pupil | 16 | 12 | 0 | 40 |
| 16 changed continuations, aligned reach | 16 | 13 | 1 | 42 |
| 32 confirmation releases, raw pupil | 16 | 9 | 1 | 49 |
| 32 confirmation releases, aligned reach | 19 | 12 | 4 | 56 |

The aligned controller made first paddle contact on four previously missed openings, two left and two right. It lost one opening contact on a middle ball. The difference in legal returns came from already-contacted balls. Release 3180 went from three to four near legal replies and reached nine alternating returns. The reversed control lost seven opening contacts and gained none. First deliveries matched down to the far contact, near bounce and reference-line sample on all 32 games for both orientations. [New raw commands](paired-new-fixed.json.gz) · [Aligned commands](reach-new-aligned.json.gz) · [Reversed commands](reach-new-reversed.json.gz)

On changed continuations, reach left opening contacts unchanged and added four total paddle contacts. Release 3025 gained one later reply with opponent continuation seed 361746. The feedback layer intervened on 564 ticks on the new-release cohort. It used the live blade observation on 539 of those ticks and reached a peak measured joint speed of 4.82 rad/s. [Changed-continuation proof](reach-replies-aligned.json.gz) · [Paired continuation control](paired-replies-fixed.json.gz)

The frozen result repeated on a second 32-release roster. Reach gained four opening contacts and lost one. Again, the newly contacted balls did not land legally. Opening legal returns tied at eight each, while the additional legal returns came from later replies. Release 8129 went from one near return to four and reached nine alternating returns. First deliveries matched through the reference line on all 32 points. Across both unfamiliar rosters, opening contacts rose from 35 to 41 and legal returns from 20 to 26, including a rise in later legal returns from four to seven. [Confirmation roster](course-confirm.json.gz) · [Confirmation results](reach-confirm-results.json) · [Confirmation commands](reach-confirm-aligned.json.gz) · [Raw control](paired-confirm-fixed.json.gz)

Command-only replay reproduced all referee events on 34 scoring, newly contacting, or later-gain points. Full-CAD replay found 0 penetrations across 53,320 solver steps, with 18.6 mm minimum clearance. [New aligned audit](reach-new-aligned-audit.json) · [Reversed audit](reach-new-reversed-audit.json) · [Later-reply audit](reach-replies-aligned-audit.json) · [Confirmation audit](reach-confirm-aligned-audit.json)

## Reproduce

Use the pinned measured launch file from the [endurance instructions](../tt29/endurance.md). The raw observation export can be rebuilt by replaying its saved demonstrations.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt33_joint capture --raw --output /tmp/tt36-reach-recording.npz
python -m scripts.experimental.tt36_reach prepare /tmp/tt36-reach-recording.npz artifacts/tt36/reach-observations.npz
python -m scripts.experimental.tt36_reach run /tmp/tt25-rallies.json practice 0 1
python -m scripts.experimental.tt36_reach choose
python -m scripts.experimental.tt36_reach run /tmp/tt25-rallies.json new aligned reversed
python -m scripts.experimental.tt36_reach run /tmp/tt25-rallies.json replies aligned
python -m scripts.experimental.tt36_reach run /tmp/tt25-rallies.json confirm aligned
python -m scripts.experimental.tt36_reach report
python -m scripts.experimental.tt36_reach report --confirmation
python -m scripts.experimental.tt36_reach audit aligned
python -m scripts.experimental.tt36_reach audit reversed
python -m scripts.experimental.tt36_reach audit aligned --cohort replies
python -m scripts.experimental.tt36_reach audit aligned --cohort confirm
```
