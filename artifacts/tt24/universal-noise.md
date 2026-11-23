# Public residuals and universal feedback

Ten independent stock imaging clips paired the stored wrist observer with public stereo ball tracks and disk silhouettes while the joints ran ball-independent probes. These were unscored imaging clips with ball contact disabled in the imaging bench. No numeric stock scene state or stock game result entered fitting. The qualified stock game used its original contacts and referee.

The public differences estimated causal lag, cross-channel correlated innovations, per-clip bias, and dropout persistence from 162 usable consecutive-frame pairs, including 22 wrist gaps. During source-only continuation, corrupted copies of 4,196 observations from 24 physical donor drops supplied a closed-form consistency fit for the existing quadratic features. Each output came from the frozen generic readout's own preactivation on the paired clean donor observation. The update used ridge regularization around that prior, and 12 separate donor drops selected its strength. [Implementation](../../scripts/experimental/tt24_universal_noise.py)

The full update increased the mean on the donor validation with injected uncertainty from 1.42 to 2.33. On 16 new stock drops it made smaller gains, while the wrist-only mean stayed below stationary control.

| Stock condition | Mean consecutive | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Stationary | 2.56 | 4 | 0/16 |
| Frozen universal, clean state | 5.69 | 15 | 5/16 |
| Consistency update, clean state | 6.44 | 17 | 5/16 |
| Frozen universal, wrist RGB and encoders | 1.63 | 3 | 0/16 |
| Consistency update, wrist RGB and encoders | 2.06 | 8 | 0/16 |

All 80 held action tapes replayed exactly. The original universal checkpoint and negative wrist result were preserved. [Updated checkpoint](universal-noise.npz) · [Held outcomes and action tapes](universal-noise.json.gz)
