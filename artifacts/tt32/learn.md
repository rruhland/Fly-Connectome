# Learning recovery distance from the next return

The 16-drill SO-101 checkpoint and the calibrated left-ready posture were held fixed. After a predicted stroke, the same motor planner used in TT31 checked a smooth return toward the ready pose beginning 65 ms after predicted contact. We varied the distance toward the pose between 100%, 65% and 35%, using the same timing curve. The private physics model had to preserve predicted contact time, landing position and net clearance within the existing 2 mm and 2 ms screen. It could reject a change and leave the stroke as originally planned. [TT31 fixed recovery](../../scripts/experimental/tt31_recovery.py) · [Frozen motion and optical checkpoint](../tt31/ready-staged-checkpoint.json.gz) · [Learning implementation](../../scripts/experimental/tt32_learn.py)

Each distance played the same four alternating games and four games with continuously varying placement across the centerline, using the same release and opponent seed on each card. The four measured practice releases were disjoint from the held releases. The recovery learner used the sign of the camera-based lateral contact prediction for the current stroke. A checked recovery supplied a trial if its commands were actually issued after paddle separation, the stroke landed, and the live opponent then delivered a legal reply. The trial succeeded only when the SO-101 legally returned that next ball. The policy received completed referee outcomes for learning. It received no opponent target, true ball trajectory or label selecting a particular action at play time. [Frozen practice and held schedules](learn-plan.json.gz)

For each predicted side, the learner estimated a next-return probability from the binary outcomes with a uniform beta prior. An alternative needed two physical trials to replace full-distance recovery, and a tie favored the fixed rule. We froze the fitted table and used it alongside the unchanged motion and optical banks in four held alternating games and eight held continuously varied games. The fixed comparison reuses the prior TT31 games with identical caps, releases, seeds, physics and banks. The full-distance implementation reproduced all four archived periodic practice games down to the commands, referee events and placements. [Fixed periodic games](../tt31/recovery-periodic.json.gz) · [Fixed continuously varied games](../tt31/recovery-full.json.gz)

## Practice

| Recovery distance | Next returns after predicted left strike | Next returns after predicted right strike | All legal SO-101 returns |
| --- | ---: | ---: | ---: |
| 100% | 5/8 | 6/7 | 16 |
| 65% | 1/5 | 2/3 | 8 |
| 35% | 3/6 | 4/5 | 12 |

The 34 credited choices favored full-distance recovery after both predicted strike sides. Shorter recovery still changed individual rallies. On alternating release 15343, the arm landed six balls with 35% recovery and four with full recovery. On 7935 it landed seven with full recovery and one at either shorter distance. [Full-distance practice tape](learn-practice-0.json.gz) · [65% practice tape](learn-practice-1.json.gz) · [35% practice tape](learn-practice-2.json.gz) · [Fitted table](learn-checkpoint.json)

Command-only replay of the 35% game on 15343 reproduced thirteen alternating returns and three legal contacts on each side of the arm. Full-CAD replay found zero penetrations across 5,203 solver states and measured a minimum clearance of 18.6 mm. [Independent replay](learn-15343-audit.json)

## Held games

The frozen table reproduced the fixed recovery's commands, physical events and actual deliveries on all twelve held games. Each policy landed six SO-101 returns in the four alternating games, for sixteen total returns, and six in the eight continuously varied games, for twenty total returns. The learned-table runs used 50.9 and 124.4 process CPU seconds, respectively. The archived fixed policy used 49.1 and 122.3. The practice comparison also matched all eight first deliveries across all three distances, through far contact, near bounce and receiving-line measurement. [Paired game scores and input checks](learn-summary.json) · [Alternating tape](learn-held-periodic.json.gz) · [Continuously varied tape](learn-held-full.json.gz)

## Reproduction

Use the pinned measured feeds and dependencies from the [regulation endurance instructions](../tt29/endurance.md). The three practice commands can run in parallel.

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json prepare
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json practice --arm 0
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json practice --arm 1
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json practice --arm 2
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json fit
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json periodic
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json full
python -m scripts.experimental.tt32_learn /tmp/tt25-rallies.json summary
```
