# Raw-event re-anchoring retains partial code capacity, but has two bottlenecks

The [registered frozen-code diagnostic](2026-09-24-raw-reanchoring-capacity-protocol.md) reused the 96 generic training episodes, the 18-channel learned and random dictionaries, and the prior evaluation-only 5×5 linear future-event decoder. The fitted decoder is an upper-bound probe, **not** the online local learner. Production code and the learned network's weights were unchanged.

| Next-event F1 | Learned raw + correlation code | Random raw + correlation code | Prior learned correlation-only code |
| --- | ---: | ---: | ---: |
| Online local readout, training | 0.395 | 0.305 | 0.671 |
| Offline capacity, training | **0.781** | 0.544 | 0.963 |
| Offline capacity, unseen single patterns | **0.644** | 0.320 | 0.918 |
| Offline capacity, five robust families pooled | **0.579** | 0.262 | Not measured in the prior capacity run |

The learned augmented latent remains more informative than its random-dictionary control, and all unseen single-pattern target events lie within the decoder's local spatial reach. The capacity decline from 0.918 to 0.644 is therefore not a coverage failure. It is substantial loss of decodable motion information when raw and coincidence channels compete for the same 24 latent units. The online local readout loses additional performance even on its training episodes. Changing the offline score threshold from 0.5 to 0.3 barely changes unseen F1 (0.644 to 0.643), so threshold choice does not explain the result.

Robust-scene spatial coverage is 0.965. The learned offline readout reaches 0.38–0.66 across the four disappearance subgroups and 0.49–0.71 across the four sensor-noise subgroups. These are useful upper bounds, but the same augmented model's online local F1 was only 0.123 on disappearance and 0.122 on noise in the preceding experiment. This supports a readout/credit problem in addition to code interference. The registered high-capacity and low-capacity gates were neither cleanly met: this is mixed evidence, not proof of one sole cause.

The next bounded architecture experiment should preserve the successful 16-channel coincidence learner as its own population and provide raw first-sighting evidence through a distinct local pathway. It should first ask whether a frozen combined code preserves ordinary motion capacity while exposing reappearance, then whether locally learned persistent state can exploit that evidence across missing frames. Do not merge raw channels into the same competitive dictionary or promote any architecture from the offline decoder alone. This capacity run took 118 seconds.
