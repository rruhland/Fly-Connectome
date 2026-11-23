# Using early outgoing flight to correct an SO-101 return

The outbound-flight learner landed 47 of 368 fresh balls after three probes. The reward-only learner landed 42 and the frozen receiver landed 38. With five probes, both learners landed 50. Early flight feedback sometimes recovered a legal return before practice produced one, with the clearest gains on a blade rotated −0.18 rad. The overall advantage was small and disappeared by the fifth probe. [Held trajectories and command tapes](outgoing-held.json.gz)

| Held launches | Practice balls | Frozen | Reward only | Outgoing flight |
| --- | ---: | ---: | ---: | ---: |
| First 184 | 3 | 20 | 20 | 21 |
| Second 184 | 3 | 18 | 22 | 26 |
| All 368 | 3 | 38 | 42 | 47 |
| First 184 | 5 | 20 | 20 | 21 |
| Second 184 | 5 | 18 | 30 | 29 |
| All 368 | 5 | 38 | 50 | 50 |

After three probes the outgoing branch recovered 14 balls the reward branch missed and lost nine the reward branch landed. After five it gained and lost five each. Resampling by donor gave a 95% interval of −1.4 to +4.9 percentage points for the three-probe advantage over reward only. With shifted blades, the three-probe comparison was 33/240 for outgoing flight and 24/240 for each control. All nine net gains in this subset came at −0.18 rad. The nominal blade scored 14/128 with outgoing flight, 18/128 with reward alone and 14/128 frozen. The selected offset opposed the shift on 18 of the 30 shifted streams after three probes and 21 after five. [Paired physical outcomes](outgoing-held.json.gz)

On donor 14774 with the −0.18 rad blade, three misses produced a +0.18 rad correction that landed four of eight new launches. Both controls landed zero. At the nominal angle on fresh launches from the same donor, that comparison reversed to zero versus four. On donor 9069 with the −0.18 rad blade, three misses still let the outgoing fit land five of eight. After two further probes both learners landed all eight. These results show that an early image-derived flight can supply a usable correction on some new trajectories after a miss. [Per-donor practice and results](outgoing-held.json.gz)

## Procedure

On a regulation table in MuJoCo, the near SO-101 made a short wrist-roll probe on each of five distinct balls. The receiver used ordinary orange-ball pixels in two fixed RGB cameras to find three early positions moving down-table around the reversal. Those points, all before the net, estimated where the ball would first reach table height and how high it would cross the net. The flight fit used gravity and approximate drag. The arm issued ordinary 30 Hz joint-rate commands and used its encoder interface. The far arm stayed at rest after the recorded feed. [Experiment runner](../../scripts/experimental/tt28_outgoing.py)

The course used 23 measured donors for which TT27's 56-goal coverage sweep had recorded at least one paddle contact. The fixture gave every branch the same starting posture by taking the closest opening RGB match among that donor's archived contact goals. The fixture selected this posture using the donor identity and the archived contact bits. The other 15 TT27 donors had no contact in the sweep. This setup evaluates steering after an already supported interception. [Historical contact sweep](../tt27/action-coverage.json.gz)

The same five probes went to both learners in the same order. The reward branch kept the smallest tested offset with a legal landing and used the starting setting when all probes missed. The outgoing branch received the same binary rewards plus the forecasts from its own probes. It fitted the local relationship between wrist offset and forecast by weighted least squares, then chose an offset expected to send the ball past the net and toward the interior of the far half. The fit could choose an intermediate offset even when every tested ball had missed. A frozen branch used the original posture. The learners used only the issued offsets, binary landing rewards and, for the outgoing branch, image-derived forecasts. The experiment supplied the wrist axis, probes, optical calibration and flight approximation. It used no backpropagation.

Two independently drawn sets of variations used the same 23 donors. The evaluator varied the blade angle around its handle among −0.18, 0 and +0.18 radians and rotated the allocation across sets. In each stream, five practice balls were followed by eight fresh variations of the same donor with different launch positions, velocities and spin. All policies were frozen during those eight balls. The first three practice balls also supplied a three-observation comparison on the same held launches. These variants test transfer to a different physical trajectory near one recording. The fit was local to each stream. [Chronological probes and held command tapes](outgoing-held.json.gz)

The detector produced 159 forecasts during 211 practice contacts, usually two control ticks after contact, about 67 ms at 30 Hz. All detected reversals coincided with actual paddle contacts. For the 17 legal practice shots with forecasts, the median error between their projected and actual landing positions was 15 cm. Replaying the 832 tested held feed–offset combinations reproduced their referee events. The 230 practice and 368 held launches were distinct. [Optical and physical records](outgoing-held.json.gz) · [Optical development pilot](outgoing-development.json.gz)

## Reproduction

Use the hash-checked DeepMind measured-launch JSON from the regulation benchmark. The optional development command uses seven previously demonstrated donors disjoint from the held course. [Dataset provenance](../tt25/recorded-benchmarks.md)

```bash
MUJOCO_GL=egl PYTHONPATH=.:src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt28_outgoing /path/to/rallies.json --development
MUJOCO_GL=egl PYTHONPATH=.:src LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt28_outgoing /path/to/rallies.json
```
