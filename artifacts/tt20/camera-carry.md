# Learning a stroke repertoire through one camera

The same clamped SO-101 played through either one wrist RGB stream or one stationary outside RGB stream. All target play used an unmarked court at 480 × 360 and 30 Hz. Each run exposes only the selected image, its declared lens specification, the service role, clock, and previous commands. The learner also receives the signed point score in practice. A common observer uses ordinary white court lines, the known court and ball dimensions and orange ball appearance. The opponent and referee can see privileged physics. Their hidden measurements stay out of the learner.

A nominal CAD model scored 432 synthetic toss and stroke combinations and supplied four serving candidates and 2,048 dry receiving motions. The receiver carried gantry flight and impact estimates. Neither simulator model used live target joint or ball state. The single-camera receiver tests a short list of physical rollouts against a ball trajectory estimated from its own RGB frames. This is a model-based controller with learned stroke preferences.

## Training and camera changes

| Source camera | Practice tosses | Simulated control time | Legal serves | Points won | Preferred candidate after 4 / 8 / 12 / 16 |
| --- | ---: | ---: | ---: | ---: | --- |
| Wrist | 16 | 33.8 s | 12 | 5 | 1 / 1 / 1 / 1 |
| Outside | 16 | 30.7 s | 11 | 6 | 3 / 2 / 2 / 2 |

Each lens practised the same balanced order on 16 different tosses. The value update used estimated forward ball placement and the signed point score. The factory default was candidate 0. The zero-target transfer condition copied only the chosen stroke into the destination; its sensor began with empty tracking history. The continuous condition carried the four program values and counts, updated each twice from a new set of eight single-camera target serves, and compared the resulting choice with using just those eight target serves. The return planner stayed fixed.

## Held complete games

| Camera used in game | Source of stroke | Opponent | Ranging | Score | Legal own serves | First returns on opponent serve | Legal learner returns | Points with 2+ learner returns |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Outside | Continued checkpoint | Standard | temporal | 6–11 | 7/9 | 5/9 | 6 | 1 |
| Outside | Factory | Standard | temporal | 4–11 | 6/7 | 3/8 | 3 | 0 |
| Outside | Wrist | Standard | temporal | 3–11 | 6/6 | 3/8 | 3 | 0 |
| Outside | Outside | Standard | temporal | 1–11 | 5/6 | 2/6 | 2 | 0 |
| Outside | Wrist | Search | adaptive | 4–11 | 7/7 | 3/8 | 3 | 0 |
| Outside | Wrist | Search | temporal | 4–11 | 7/7 | 3/8 | 3 | 0 |
| Wrist | Factory | Standard | temporal | 4–11 | 6/7 | 6/8 | 6 | 0 |
| Wrist | Wrist | Standard | temporal | 6–11 | 8/8 | 6/9 | 6 | 0 |
| Wrist | Outside | Standard | temporal | 6–11 | 7/8 | 6/9 | 6 | 0 |
| Wrist | Outside | Search | adaptive | 0–11 | 3/5 | 3/6 | 3 | 0 |
| Wrist | Outside | Search | temporal | 0–11 | 3/5 | 3/6 | 3 | 0 |

All rows played to 11, win by two, with service alternation. The standard comparisons used seed 40582930, the search comparisons used 68239503, and the standalone continued checkpoint used its own fresh seed 3044789. Each action tape passed a fresh physical replay. The adaptive optical variant chooses the apparent-size range when estimated error is at most 20 mm and uses the temporal fit otherwise. Both methods use the same single camera. Game score includes later opponent replies.

## Same 24 new own serves

| Camera in play | Source of stroke | Legal serves | Points won | Newly legal / lost vs factory |
| --- | --- | ---: | ---: | ---: |
| Wrist | Factory | 20/24 | 5/24 | 0 / 0 |
| Wrist | Wrist | 22/24 | 4/24 | 4 / 2 |
| Wrist | Outside | 20/24 | 4/24 | 4 / 4 |
| Outside | Factory | 14/24 | 7/24 | 0 / 0 |
| Outside | Outside | 19/24 | 11/24 | 10 / 5 |
| Outside | Wrist | 21/24 | 9/24 | 10 / 3 |

## Eight target-camera updates after the switch

| Direction | Choice from target alone / with source | Legal serves from target alone / with source | Points won from target alone / with source |
| --- | ---: | ---: | ---: |
| Wrist → Outside | 3 / 1 | 24 / 21 | 12 / 9 |
| Outside → Wrist | 1 / 1 | 22 / 22 | 4 / 4 |

The exact paired counts and two-sided binomial tests are in the per-camera receipts. All 24 tosses were disjoint from both stages of practice. For each actual candidate the player finished the point, so legal service and eventually winning the point are separately observable.

## A point-reward comparison

Ranking the same four serving programs from signed point rewards alone selected candidate 1 independently through both cameras. On the same 24 fresh outside-only tosses, the wrist-trained choice raised legal serves from 14 to 21 and point wins from seven to nine. In the reverse direction it raised legal serves from 20 to 22 and changed five wins to four. These counts reuse physically identical command sequences from the paired bench. Experience can help select a legal serve across cameras. Match-winning gains from that selection were inconsistent. [Reward counts and matched receipts](score-only-carry.json.gz)

## Wrist optics in scored play

A compact 126° alternative at 960 × 720 kept the gripper and paddle visible and improved visibility of the legal toss. A receiving policy and a serve trained on the standard wrist eye played the first eight previously scheduled points through this new lens. Each camera served legally on all four own serves. The wide eye produced an early metric ball sighting during each toss, versus none on the standard lens. On the four opposing serves the wider lens returned zero versus three on the original lens, and across eight points won two versus three. Wider coverage alone does not ensure good metric tracking. The 84° external camera remains the primary gameplay lens. [Held RGB and scores](ultrawide-wrist-play.json.gz) · [Visibility study](fixture.md)

## Run locally

From the repository root using the project MuJoCo Python environment, this continues the outside-trained program values on eight new wrist-camera serves and then plays a wrist-only game from that updated checkpoint. Use a new seed for a new continuation.

```bash
python -m scripts.experimental.tt20_stroke_carry wrist --resume artifacts/tt20/camera-carry-left-practice.json.gz --seed 746301
python -m scripts.experimental.tt20_camera_carry wrist --checkpoint artifacts/tt20/stroke-carry-wrist-746301.json.gz
```

[Single-camera training and play](../../scripts/experimental/tt20_camera_carry.py) · [Continuous checkpoint update](../../scripts/experimental/tt20_stroke_carry.py) · [Outside paired serves](camera-carry-left-paired-serves.json.gz) · [Wrist paired serves](camera-carry-wrist-paired-serves.json.gz) · [Reward-only opening pilot](opening-exploration.md)
