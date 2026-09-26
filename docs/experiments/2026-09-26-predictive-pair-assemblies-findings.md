# Delayed local pairing improves the slow motion code, but not its forecast

**Status:** One-shot opt-in comparison under the [registered protocol](2026-09-26-predictive-pair-assemblies-protocol.md). The frozen fast predictor and 64 unlabeled generic training episodes were shared. Aligned pairing, input-only assignment, and time-shuffled future pairing had identical template initialization and update budgets. Future evidence affected winner selection only during training; inference used present fast activity. Production M1A and the measured graph were not changed.

The fixed 3×3 **offline** direction probe found a substantial advantage in the spatial slow state:

| Transfer set | Fast only | Aligned pair | Input only | Shuffled pair |
| --- | ---: | ---: | ---: | ---: |
| Doubled speed | 12/16 | **14/16** | 10/16 | 10/16 |
| Changed shape | 10/16 | **14/16** | 6/16 | 4/16 |
| Separated movers | 14/16 | 16/16 | 16/16 | 16/16 |
| Crossing movers | 2/4 | 4/4 | 4/4 | 4/4 |

The matched controls suggest that delayed *aligned* future evidence can shape a more transferable unit identity. This is stronger evidence for a learned motion code than reconstruction-only or shuffled pair learning, but the probe uses known object centers **after inference**. It does not demonstrate autonomous entity discovery or tracking.

The forecast gate failed. Full-frame next-meaningful-event top-32 recall was:

| Held-out set | Fast only | Aligned pair | Input only | Shuffled pair |
| --- | ---: | ---: | ---: | ---: |
| Generic scenes | .360 | **.433** | .429 | .385 |
| Doubled speed | **.480** | .362 | .339 | .344 |
| Changed shape | **.542** | .473 | .462 | .514 |
| Separated movers | **.312** | .293 | .273 | .286 |
| Crossing movers (2 scenes) | .354 | **.396** | .359 | .349 |

Aligned pairing beat input-only on 11/16 speed cases and 10/16 shape cases. It beat shuffled pairing on only 8/16 speed cases and **4/16** shape cases. It beat fast-only on 2/16 speed and 6/16 shape cases. Top-8 recall likewise remained below fast-only on speed (.141 versus .151) and shape (.167 versus .187). Thus the better slow state has not become a transferable predictor. The small crossing forecast lead is from only two scenes.

Autonomous coherence remains unresolved. On changed-shape scenes, 27.2% of significant aligned-assembly activity fell outside offline object neighborhoods, and there were 1.40 connected slow components per active frame for one mover. On separated scenes there were 3.44 components for two movers, 1.44 excess on average; about 30.4% of activity fell outside the offline neighborhoods. At doubled speed, outside activity reached 48.6%. These are not object detectors or tracking guarantees.

**Decision:** Reject this paired-template system as a complete M1A representation-plus-forecast architecture: it misses the registered speed/shape forecast-preservation gate and the aligned-versus-shuffled shape gate. Preserve the narrower finding that delayed future evidence shapes a transferable *spatial direction code*; do not tune its footprint, fusion gain, sparsity, or learning rate against these held-out sets. The next meaningful experiment should change how a learned state predicts future visual evidence, with a locally learned temporal transition and a way to express both supported and suppressed future activity, rather than add another static readout to the same templates. Require the same fast-only and time-shuffled controls, and continue measuring autonomous coherence.

Reproduction: `python scripts/run_predictive_pair_assemblies.py`; [raw results](2026-09-26-predictive-pair-assemblies-results.json) contain top-8/top-32 recall and precision, per-scene counts, probe scores, and activity/component measures. Elapsed time was about 65 seconds.
