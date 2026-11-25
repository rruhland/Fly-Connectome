# Few-shot direct-joint repairs from physical misses

On 16 ordinary terminal misses from 16 previously unseen correction-release IDs, context-matched repairs recovered **5 shots**, versus **2** for matched shuffled repair labels and **0** for stock. Four contextual rescues and both shuffled rescues received a legal reply from the ordinary opponent. The fifth contextual return won the point. This is practice efficiency on resettable, same-delivery forks, not a rally-performance result. [Frozen roster](protocol.json) · [Issued tapes and CAD receipts](forks.json.gz)

| Condition | Legal and CAD-clean by attempt 1 | By 2 | By 3 | By 4 | Real attempts spent | Mean attempt to first score, missing coded as 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contextual paired repair | 2 | 4 | 5 | 5 | 53 | 4.00 |
| Same-dose shuffled repair | 1 | 2 | 2 | 2 | 59 | 4.56 |

The contextual first-trial successes were middle 3401 and right 10544. Left 3905 and left 5222 scored on attempt two, and middle 11884 scored on attempt three. Shuffle scored on 3401 and 5222 at the same attempt numbers. Thus there were three contextual-only successes and no shuffle-only successes. The exact two-sided paired sign test is p=.25. The same far-arm contact and near bounce matched on every one of the 112 practiced branches. At the five contextual successes, all 3,767 CAD samples were positive, with minimum clearance 16.48 mm.

## What was learned and what was controlled

The training examples were the 18 TT57 actually issued, safely scored paths on 17 independent releases. Sixteen were answered. Two on one right-wing release ended the point before an answer. For each, subtract the matched physically failed stock reference path, align the difference to a shared optical flight phase, and keep five SVD modes. The modes retain 85.66% of the residual energy. New candidates apply those residuals to three local reference anchors from RGB/encoder feedback or its separately initialized nominal twin. A minority of candidates also blend in a nearby source *failed* path as a common anchor. Neither method receives the held stock continuation or the live simulator's state.

The fixed shuffle changes which release's repair coefficients are attached to each visual/encoder context label. It leaves the full multiset of coefficient vectors intact and preserves nearest-context ordering, baseline anchors, gains, timing grid, nominal proxy, and attempt budget. The resulting *per-case* subsets, scales, and selected command magnitudes can differ. Each arm uses 48 nominal and 24 perturbed twin rollouts. A binary failure after a real trial deprioritizes similar command paths and forbids repeating one. The first attempt uses no reward from that case. Returned candidates were often not endorsed by the proxy. Four of the five actual contextual winners were not nominally legal.

## Transfer beyond these forks

Adding the five new paired differences to the original 18 without refitting the movement basis did **not** improve a second disjoint set of 12 ordinary low/shallow losses. Both arms rescued the same right-wing delivery (6422) on their second attempt with identical actual commands and events, and the ordinary opponent answered. There were no other scored attempts. [Follow-up roster](carry/protocol.json) · [Issued follow-up tapes](carry/forks.json.gz)

A distinct four-assignment prospective continuous check compared stock, contextual, and shuffled versions requiring public-model nominal legality and a perturbation vote before handing over the arm. It completed all 12 one-launch trials and replayed all native referee events with positive CAD clearance. No repair was issued. Each version scored 18 returns across the four assignments, all on the same point. Two assignments used one shared release recording with independently assigned opponent draws. Three points ended before any eligible model branch. The one eligible decision on the fourth point did not satisfy the strict proxy guard. These games confirm the execution path and guard, not an improvement in sustained play. [Preassigned launches](games.protocol.json) · [Full point tapes and replay](games.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt59_fewshot_joints plan
python -m scripts.experimental.tt59_fewshot_joints fit
python -m scripts.experimental.tt59_fewshot_joints forks
python -m scripts.experimental.tt59_fewshot_joint_games plan --feeds /tmp/tt25-rallies.json
python -m scripts.experimental.tt59_fewshot_joint_games run
python -m scripts.experimental.tt59_carry_fewshot plan
python -m scripts.experimental.tt59_carry_fewshot fit
python -m scripts.experimental.tt59_carry_fewshot forks
python -m pytest -q tests/test_tt59_fewshot_joints.py
```

The game protocol embeds the pinned measured initial ball for each point. The `plan` commands for whole points need the original recording export. From checked-in protocols, execution commands do not need that separate export. Derived tapes validate their frozen source hashes when replayed.
