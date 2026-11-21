# Native Pong gains on the SO-101 receiver

The Pong gains changed 26 of 36 action tapes. The player made a first legal return on the same six starts as the identity-gain control and scored the same five points. The test found no scoring or first-return gain from the native Pong coefficients on this practiced receiver. [Matched point tapes and results](pong-forward-receive.json.gz)

This test inserts two numbers from native Pong into a practiced SO-101 receiver. Pong supplies the strongest source proposal's lateral displacement gain of 1.044 and motion gain of 1.116. The source ranked four proposals by public native Pong scores, with signed totals of 16, 5, −4 and −4 against 104 scored starts per proposal. Earlier screening and validation used 584 native Pong points. The signed source bias was defined in the source image basis and the SO-101 controller uses its existing lateral origin. [Pong refinement receipt](../tt3/numeric-pong-source-refinement.json.gz) · [Portable source gains](../tt3/numeric-pong-source-refined.json)

The receiver maps the original formula to its existing RGB lateral aim. From the first visible blade position, it scales displacement to the current ball and adds a scaled forecast of lateral movement before the observed ball reaches the blade's depth. The same command bounds, pan calibration, strict legality, stroke archive, wrist and fixed cameras, 15 Hz RGB capture, and 30 Hz command loop apply in every branch. The comparisons set both gains to one or average the four shortlisted native Pong proposals without using their final reward ranks. All branches get the receiving role and the same previously practiced stroke and perception state. [Qualified receiver](stroke-strict-rearm-player.json.gz) · [Arm pan practice](stroke-receive-practice-strict.json.gz)

The 36 held points start on rival service on the compact 0.848 × 1.60 m table. The seeds match across branches and the evaluation order rotates with each start. The test uses zero new practice points and zero calibration commands. The assay measures the effect of the native gains on a shared, already practiced receiver. The archive used to prepare that receiver has earlier SO-101 and gantry stroke experience. [Trial and verifier](../../scripts/experimental/tt19_pong_transfer.py)

| Lateral gains | Starts with a legal return | Total legal returns | Starts with two or more returns | Points scored |
| --- | ---: | ---: | ---: | ---: |
| Identity (1.000, 1.000) | 6/36 | 6 | 0 | 5 |
| Best native Pong reward (1.044, 1.116) | 6/36 | 6 | 0 | 5 |
| Mean native proposals (1.015, 1.218) | 4/36 | 5 | 1 | 4 |

On the 582 frames where the Pong branch adjusted its aim, the requested lateral target moved by a median of 4.8 mm and a maximum of 14.8 mm relative to the identity formula on the same observations. The Pong branch gained two first returns over the score-blind mean, with zero losses in the paired comparison and exact two-sided p = .50. The 36 starts provide limited resolution for small effects or differences on other fixtures. Every branch used the same preparation commands for all starts, and all 108 action tapes reproduced scores and physical events under the strict end-line rule. [Run data](pong-forward-receive.json.gz)

```bash
PYTHONPATH=src:scripts/experimental python -m scripts.experimental.tt19_pong_transfer
```
