# Learning how much a table bounce tells a different game

A cart caught **364 of 512 screened rebounds** using a memory of earlier table-to-cart transitions. Reusing the latest table-tennis bounce estimate directly caught 317. Simply averaging that estimate with a generic catch prior caught **361**. The learned map recovered much of the cost of transferring the wrong physical assumption. Its advantage over the cheap fixed blend was three catches.

The learner saw how a ball bounced during table-tennis games, then saw the first eight side-view RGB frames of each cart fall. After each score it could see where the ball crossed the rail. During practice it stored one pair per four-fall cart session. The first item was the current table bounce distribution inferred from stereo. The second was the cart rebound distribution fitted to the four public landing images, starting from a uniform belief. At subsequent handoffs it retrieved cart distributions from sessions with similar optical table estimates. Two generic sessions regularized the average. The cart still learned from terminal images within each session. [Learner and replay](../../scripts/experimental/tt9_transfer_memory.py)

Each training fold contained 64 prior cart sessions from one archived table-tennis visit. Testing switched to the other physical arm, with fresh cart falls. The evaluator kept the most recent material half the time and changed it without a cue otherwise. The map stayed fixed throughout each held course. Material names, referee events and source–target continuity were available only to the evaluator. Reconstructing the practice camera images used the recorded material and applied cart commands. The training and test folds share four materials, one physical cart design and one screen protocol. [Previous chronological benchmark](../tt8/rebound-transfer.md)

| Cart initialization | Gantry source / 256 | Offset source / 256 | Same material / 256 | Material changed / 256 | First fall / 128 |
|---|---:|---:|---:|---:|---:|
| Learned from paired handoffs | 187 | 177 | 178 | 186 | 60 |
| Same training with sources shuffled | 176 | 173 | 174 | 175 | 50 |
| Same cart practice pooled across sources | 179 | 177 | 174 | 182 | 55 |
| Direct table bounce estimate | 158 | 159 | 182 | 135 | 56 |
| Half table estimate, half generic | 186 | 175 | 185 | 176 | 61 |
| Generic hard and soft mixture | 175 | 177 | 170 | 182 | 58 |

The learned pairing gained 15 catches over shuffling and eight over pooling. Most of the 47-catch improvement over direct transfer came when the material changed. A paired bootstrap of target sessions stratified by held source checkpoint and continuity put the overall gain over the fixed blend at 0.59 percentage points (95% interval −1.56 to +2.73). With two source visits, this interval only measures variation across target sessions. Keep the 50/50 blend as the comparison for future games. [Gantry evidence](catch-a.json.gz) · [Offset evidence](catch-b.json.gz) · [Map learned from the offset visit](catch-from-offset.json.gz) · [Map learned from the gantry visit](catch-from-gantry.json.gz)

## Reproduction

In the project Python environment, run the following from the repository root. Each command trains the map, reloads its checkpoint, tests it on the other arm's source stream, and replays all 1,536 recorded cart command sequences against the physical scorer.

```bash
export PYTHONPATH=.:src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt9_transfer_memory /tmp/catch-a.json.gz --train artifacts/tt8/rebound-continuity-b.json.gz --checkpoint /tmp/from-offset.json.gz --visit 0 --seed 401092089
python -m scripts.experimental.tt9_transfer_memory /tmp/catch-b.json.gz --train artifacts/tt8/rebound-continuity-a.json.gz --checkpoint /tmp/from-gantry.json.gz --visit 1 --seed 627830059
```
