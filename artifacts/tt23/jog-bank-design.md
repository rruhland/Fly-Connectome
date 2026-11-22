# Three physical jogs on fresh full-rally starts

The four independent streams use launch generator seeds `37418286`, `189532461`, `1486349668` and `308641972`. Each stream interleaves four moderate lanes with four wide lanes and balances left and right within each stratum. The 32 launch IDs are distinct and were checked against the existing TT22 and TT23 receipts. Every start supplies a physical result for each normalized jog `−0.09`, `0` and `+0.09`. The nonzero jogs request a 76.32 mm lateral shift in the arm workspace. The arm sees the same anonymous eight-frame stereo opening throughout each matched set. The body-relative position and velocity context comes from that opening and the nominal body pose. [Physical runner](../../scripts/experimental/tt23_jog_bank.py) · [Jog station](../../scripts/experimental/tt23_jog_station.py)

All alternatives use the frozen conditional opening stroke correction, the same tick-25 handoff to the frozen optical inverse-flight receiver, and the same learned five-axis far coach. The near arm uses RGB, its issued motor commands and its clock. Each point plays through its referee result. If another alternative requests exactly the same issued opening commands through handoff or termination, the physical trace can be shared. The contact, legal return, referee event and duration records are retained for every alternative. Each completed three-option set replays before its verification count advances. [Opening stroke correction](opening-motor.md) · [Learned far coach](coach.md)

The bank supports a separate causal online comparison. Each learner may choose one jog per practice visit, then update from that selected physical outcome. The other alternatives remain available to the evaluator for scoring frozen policies on held starts. The original fixed-opener qualification against a single shuffled policy stayed closed. The action-bank test addresses online selection from three alternatives and frozen priors from independent Pong fits. [Native ordinal fits](direction.md)

## Reproduce

Use the MuJoCo environment and native component library from the original opening study. Each stream can resume from its latest saved alternative. [Environment setup](tt23_sequence.md#reproduce)

```sh
python -m scripts.experimental.tt23_jog_bank --stream 1
python -m scripts.experimental.tt23_jog_bank --stream 1 --verify
```
