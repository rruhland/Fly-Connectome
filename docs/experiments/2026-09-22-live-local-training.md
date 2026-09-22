# Live local training: validation selects the smaller rate

The approved timing/credit rule has now changed the actual neural weights
during a continuous 42-trial run, with eight neural ticks per camera frame.
The two candidate learning rates were fixed before held-out tests. Both runs
started from identical original magnitudes on L3's 12 existing predictive
edges, with all other weights from the saved mixed-training source. Non-target
weights stayed frozen. No topology, transmitter sign or production plasticity
code changed.

| Rate | Training frames / seconds | Training prequential ON/OFF | Validation ON/OFF | Validation quiet alarms | Validation MSE |
|---:|---:|---:|---:|---:|---:|
| .1 | 2453 / 61.05 (40.18 fps) | .320/.037 | .271/.125 | 2.35% | .0997 |
| 1.0 | 2453 / 64.71 (37.91 fps) | .317/.231 | .600/.035 | 2.09% | .1069 |

Rate .1 was selected by the predeclared validation rule: it alone reaches
both polarity anticipation thresholds with quiet alarms under5%. Rate1.0
drives ON more strongly but weakens OFF below the threshold. The rate .1
final incoming weights include 1.720 on L2 and2.714 on L1; the other active
edge remains near .040. All stay in [0,10].

Training prequential scores use each issued prediction before its own
confirmation update. They reveal that OFF anticipation was still weak
*during* the one-pass training stream even though the final snapshot passes
validation. This limits any claim of fast online convergence. The validated
snapshot was tested on a separate nine-trial stream without further learning.

Mean cell firing during training was3.86 spikes/s and the highest individual
rate was30.62 spikes/s at rate .1. Both were finite. Effective throughput was
40.18 camera frames/s including eight simulation ticks, rule updates and
diagnostic capture per frame. The .1 validation run took13.35 seconds for510
frames. These numbers are specific to this 250-neuron motif on the current CPU.

The next stage, already fixed by the
[protocol](2026-09-22-live-local-timing-protocol.md), compares the selected
snapshot with the initial network on untouched dwell2/3/4/6 sequences and
continuous tempo-switch/omission streams. No held-out result influenced the
rate choice. Even a successful motif result will not yet establish Pong M1A.

Artifacts with per-frame issues, gate/reference, target, weight changes and
spikes are saved locally in `runs/live-local-timing-v1/train-eta-*.npz` and
`validation-eta-*.npz`; `train.json` has their hashes and selection record.
The runner refuses to overwrite its saved parity fixture. Training artifacts
are reused with their source identity when resuming this experiment.
