# More reachable sources did not yield a useful forecast

**Status:** Opt-in M1A experiment. Production architecture and measured connectome graph remain unchanged; all experimental updates are local and use no backpropagation.

The [reachability audit](2026-09-25-forecast-reachability-findings.md) showed that an eight-frame causal union of hidden-source locations puts a 5×5 local readout within reach of 95.3% of native four-frame event targets, versus 49.8% from the current state. We tested whether a minimal age-tagged version of that memory can *learn* those targets. Each hidden site retained its most recent activity and elapsed age for four or eight frames. Shared local 5×5 synapses learned signed future event probabilities separately for each hidden unit and age, with four-frame delayed local credit. Persistent camera observation remained separate. Four-frame memory, eight-frame memory, shuffled eight-frame credit, and the current-state competitive head shared the same recurrent model, generic training episodes, and generic-only event-budget calibration.

| Held-out domain | Four-frame memory F1 | Eight-frame memory F1 | Shuffled eight-frame F1 | Current-state F1 |
| --- | ---: | ---: | ---: | ---: |
| Generic fractional multi-entity scenes | .049 | .055 | .008 | .047 |
| Native 120 Hz Pong | .020 | .000 | .003 | .059 |
| Generic independent objects | .002 | .002 | .001 | .007 |
| Generic crossing objects | .001 | .001 | .007 | .015 |

Native top-eight signed recall was .132 for four-frame memory and .089 for eight-frame memory, below .195 for the current-state head. Eight-frame memory made no correct native thresholded predictions and its generic-calibrated output produced 1.68 false event pixels per quiet native frame, versus .85 for the current-state head. Held-out generic improvement over shuffled credit is small and does not transfer.

**Interpretation:** The high geometric reachability of recent sites is a *capacity ceiling*, not evidence that an independent synapse from each old site can identify the right future event. Adding old sources increases ambiguous candidates and false alarms. Unit identity plus age at one site does not express the ordered relation among nearby past events that defines local motion. A broader readout or longer hold is therefore not the next priority.

**Decision:** Do not promote bounded source memory or the age-conditioned head. The next opt-in architecture experiment should let locally plastic, sparse competing hidden units respond to **conjunctions of recent events across nearby sites and times**, then test whether those learned units forecast four-frame events across position and multi-object scenes. Keep fixed event traces as sensory primitives only; the useful motion/world-state representation must be learned. Compare against time-shuffled sequence learning, current-state readout, persistence, and quiet-frame controls. Require native transfer before a production proposal.

Reproduction: `python scripts/run_lagged_local_memory.py`; [raw results](2026-09-25-lagged-local-memory-results.json) include signed counts, F1, quiet alarms, ranks, budgets, and baselines.
