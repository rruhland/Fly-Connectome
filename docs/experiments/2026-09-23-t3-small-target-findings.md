# T3 rest current does not recover a reliable small-target signal

The [bounded opt-in protocol](2026-09-23-t3-small-target-protocol.md) changed only T3 rest current in a frozen copy of the pinned full graph. Every dot trial was paired with a blank trial restored from the same 500-tick warm state. Retinotopic locality was inferred from measured Mi1/Tm1 afferents: all 976 T3 cells had an inferred column, with a median 102 input contacts. The local x=18 region contained 69 T3 cells. The [results](2026-09-23-t3-small-target-results.json) include source and graph checksums.

| T3 rest current | ON excess local spikes | OFF excess local spikes | Blank T3 spikes/neuron/tick |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 0.85 | 0 | 0 | 0 |
| 0.95 | 0 | 0 | 0 |
| 1.05 | -2 | 0 | 0.00067 |
| 1.15 | -2 | +2 | 0.00099 |

The preset requirement was at least five excess spikes for **each** polarity with a blank rate below 0.01. No current passed, despite 72 camera-pixel events in each moving-dot condition and finite neuron state throughout. The higher currents induced a little blank activity but did not turn the measured T3 input into a dependable event response. Per protocol, no direction, second-location, bar, or static tests were run and no value was selected. These data do not support adding LC11 to the present spiking roster on its own: its existing T3 input would remain nearly silent under this probe.

The narrower next question is anatomical: measure the actual T2/T3-to-LC11 edge mass and what fraction of LC11's measured inputs the pinned graph already retains. That read-only audit can distinguish an omitted downstream decoder from a missing upstream signal without another full simulation or weight sweep. No production architecture or learning rule changed here.
