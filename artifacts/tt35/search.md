# Learning a shared joint adjustment from rally outcomes

The learned adjustment extended one practice point from five to nine alternating legal returns. On the eight fresh held releases it improved from three to four near-arm returns, with the gain coming from a later reply. Scores tied on the separate 16-release held roster. [Paired results](search-results.json) · [Implementation](../../scripts/experimental/tt35_search.py)

## Practice

The raw visual controller kept its original stereo inputs, measured joint angles and control rate. Its only learned change was a vector of five smooth, shared joint setpoint biases. The adjustment entered through the existing observation basis at the current control tick. It used the same parameters on every approach and issued one set of joint-rate commands at a time. [Raw policy](../tt33/joint.md) · [Frozen learned adjustment](search-checkpoint.json)

Four sequential updates each compared the incumbent with two opposite perturbations on the same four practice cards. The roster interleaved continuous and periodic deliveries. Each near-arm legal reply earned one unit and each later legal reply earned one additional unit. The learner kept its incumbent on a tie. This fit used 48 physical episodes and 88.29 process CPU seconds. The only accepted update added −0.065, +0.065, −0.065, −0.065 and −0.065 rad to the five playing joints. [All practice variants and commands](search-practice.json.gz)

The acceptance came from release 12330 with opponent seed 315509. The incumbent landed right and left replies. The adjusted policy landed right, left, right and left in one point. A matched replay confirmed the two scores, including the later paddle contacts and legal landings. [Practice replay of both controllers](search-practice-repeat-audit.json)

## Held games

The two held rosters were disjoint from the search practice releases. The selected checkpoint was frozen before the assessments. Every first physical delivery matched the prior exact raw-policy control at the far contact, near bounce and receiving line. After the first reply, the live opponent responded to the ball it received. [Novel roster](../tt34/body-novel-plan.json.gz) · [Fresh roster](../tt32/fresh-plan.json.gz)

| Held roster and policy | Near legal returns | Later near legal returns | Alternating legal returns |
| --- | ---: | ---: | ---: |
| 16 novel releases, frozen raw policy | 6 | 1 | 28 |
| 16 novel releases, learned adjustment | 6 | 1 | 28 |
| 8 fresh releases, frozen raw policy | 3 | 0 | 14 |
| 8 fresh releases, learned adjustment | 4 | 1 | 16 |

The fresh gain came from 8239, where the frozen policy hit the net on the opening reply. The adjusted policy landed two left-side returns and the rally reached five alternating legal returns. It lost the previously successful opening reply on 9532. On the 16-release roster, it gained 11139 and lost 10505. All four affected games had a blade contact on the opening reply with both policies. The outcome changed at the landing rather than at interception. Across the two rosters, the number of openings returned stayed the same. [Learned fresh commands](search-held-fresh.json.gz) · [Raw fresh commands](../tt33/joint-fresh-raw.json.gz) · [Learned novel commands](search-held-novel.json.gz) · [Raw novel commands](../tt34/body-novel-stock.json.gz)

The two held assessments used 46.61 process CPU seconds. The learned policy reached 4.55 rad/s peak measured joint speed. The eight scoring held points and both practice controls reproduced their complete referee streams in command-only replay. Full-CAD scans found zero penetrations across 19,645 physics steps, with 18.6 mm minimum clearance. The observed transfer is one extra held later return, alongside gains and losses on individual openings. [Fresh audit](search-held-fresh-audit.json) · [Novel audit](search-held-novel-audit.json) · [Practice audit](search-practice-repeat-audit.json)

## Reproduce

Use the pinned measured launches from the [endurance reproduction instructions](../tt29/endurance.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt35_search fit /tmp/tt25-rallies.json
for cohort in novel fresh; do
    python -m scripts.experimental.tt35_search assess /tmp/tt25-rallies.json "$cohort"
    python -m scripts.experimental.tt35_search audit "artifacts/tt35/search-held-$cohort.json.gz" "artifacts/tt35/search-held-$cohort-audit.json"
done
python -m scripts.experimental.tt35_search summary
python -m scripts.experimental.tt35_search audit-practice
```
