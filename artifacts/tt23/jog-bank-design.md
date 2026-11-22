# Three physical jogs on fresh full-rally starts

The four independent streams use launch generator seeds `37418286`, `189532461`, `1486349668` and `308641972`. Each stream interleaves four moderate lanes with four wide lanes and balances left and right within each stratum. The 32 launch IDs are distinct and were checked against the existing TT22 and TT23 receipts. Every start supplies a physical result for each normalized jog `−0.09`, `0` and `+0.09`. The nonzero jogs request a 76.32 mm lateral shift in the arm workspace. The arm sees the same anonymous eight-frame stereo opening throughout each matched set. The body-relative position and velocity context comes from that opening and the nominal body pose. [Physical runner](../../scripts/experimental/tt23_jog_bank.py) · [Jog station](../../scripts/experimental/tt23_jog_station.py)

All alternatives use the frozen conditional opening stroke correction, the same tick-25 handoff to the frozen optical inverse-flight receiver, and the same learned five-axis far coach. The near arm uses RGB, its issued motor commands and its clock. Each point plays through its referee result. If another alternative requests exactly the same issued opening commands through handoff or termination, the physical trace can be shared. The contact, legal return, referee event and duration records are retained for every alternative. Each completed three-option set replays before its verification count advances. [Opening stroke correction](opening-motor.md) · [Learned far coach](coach.md)

The bank supports a separate causal online comparison. Each learner may choose one jog per practice visit, then update from that selected physical outcome. The other alternatives remain available to the evaluator for scoring frozen policies on held starts. The original fixed-opener qualification against a single shuffled policy stayed closed. The action-bank test addresses online selection from three alternatives and frozen priors from independent Pong fits. [Native ordinal fits](direction.md)

## Complete physical bank

All 96 alternatives ran live, and all 96 command tapes reproduced the exact point durations, strict referee events and terminal outcomes. All 32 anonymous opening contexts were available. The final bank preserves the planned launch order after merging two disjoint verified spans. [Stream 1](jog-bank-1.json.gz) · [Stream 2](jog-bank-2.json.gz) · [Stream 3](jog-bank-3.json.gz) · [Stream 4](jog-bank-4.json.gz)

| Physical outcome across 32 starts | Left | Center | Right |
| --- | ---: | ---: | ---: |
| Opening contacts | 16 | 14 | 15 |
| Legal opening returns | 5 | 9 | 11 |
| All near-arm legal returns | 18 | 45 | 49 |
| Points reaching 10 alternating returns | 1 | 4 | 4 |
| Points reaching 20 alternating returns | 0 | 1 | 2 |
| Longest exchange | 10 | 22 | 24 |

Seven distinct starts reached ten with at least one jog. Three reached twenty. The longest result occurred on stream 4 start 2, seed `879560681`. The left, centered and right options reached 0, 19 and 24 alternating returns on that matched release. [Stream 4](jog-bank-4.json.gz)

## Reproduce

Use the MuJoCo environment and native component library from the original opening study. Each stream can resume from its latest saved alternative. [Environment setup](tt23_sequence.md#reproduce)

```sh
python -m scripts.experimental.tt23_jog_bank --stream 1
python -m scripts.experimental.tt23_jog_bank --stream 1 --verify
```
