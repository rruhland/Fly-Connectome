# A fixed resting-voltage reference drives false LC11 activity

The [opt-in partial LC11 protocol](2026-09-23-lc11-shadow-readout-protocol.md) replayed the frozen pinned network and passed its T2/T2a/T3 outputs over their **measured** right LC11 contacts into a separate 75-cell adaptive-LIF bank. No production neuron or edge was changed. The [compact results](2026-09-23-lc11-shadow-results.json) include source and raw-weight checksums; full per-tick calibration release, current, and spikes are in the ignored `runs/motion-lc11-shadow-v1/per-tick.npz` and identified by checksum in the report.

Spike-only transmission produced zero dot-evoked or blank LC11 spikes, as expected from the upstream T2/T2a/T3 silence. Bounded voltage-dependent release changed LC11 current but no allowed setting met the preset **both-polarities** spike criterion:

| Maximum release/source/tick | ON excess LC11 spikes | OFF excess | Blank spikes/neuron/tick |
| ---: | ---: | ---: | ---: |
| 0.05 | 0 | 0 | 0.00181 |
| 0.10 | 0 | +1 | 0.01944 |
| 0.25 | -2 | +6 | 0.06611 |

The largest setting also exceeded the 20% instantaneous-spiking cap (22.7%). No fraction was selected, so the direction, second-location, bar, and static readouts were correctly skipped. This experiment does not test the full LC11 circuit: its other measured excitatory and inhibitory inputs and LC11 output loops were intentionally absent.

The failure has a concrete mechanism. The no-event upstream voltage is *not fixed* after the 500-tick warm state. Relative to that one warm snapshot, its mean positive drift during blank replay was 0.00486 and its positive 99th percentile 0.0352, both larger than the 0.003 release scale. Some 2,433 of the connected source cells exceeded that scale during blank, despite **zero source spikes**. At the smallest release setting, mean blank release was 0.01983 per source per tick versus 0.01994 during the ON dot; the dot's local voltage signal was swamped by autonomous state variation across the convergent population. Raising the release gain mostly amplified that background.

The negative result stops this fixed-reference law, not the measured T2/T3→LC11 route. A single bounded follow-up can test a *local moving voltage baseline* with the already-used 250 ms time constant, initialized from the warm state. It would ask whether local adaptation subtracts the slow blank variation while preserving fast dot transients, without changing signs, contacts, delays, or learning. If that fails, further gain sweeps are unlikely to help; the object pathway needs a better temporal or inhibitory representation before learning on LC11 is justified.
