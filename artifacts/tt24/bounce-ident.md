# Identifying the paddle pupil

Four 300-step trajectories from the privileged teacher supplied continuous joint-target labels. The generic ridge readout chose one nonlinear pair from all 15 pairs of the six relative object-position and motion channels by leaving one demonstration out at a time. It selected channels 2 and 5, which represent vertical position and velocity. The chosen head converts its fitted joint targets to normalized rates using its own command history. [Implementation](../../scripts/experimental/tt24_bounce_ident.py)

Perception received **privileged training supervision** on the same four trajectories. The calibration fit camera and paddle transforms from encoder angles, mapped orange silhouette position and size to relative scene position, and fit a causal velocity estimate from short image histories. The silhouette estimate used subpixel boundary midpoints and omitted frame-border edges. The deployed checkpoint takes wrist RGB and own joint angles, with fixed fitted weights. The raw difference comparator uses an immediate vertical difference and a longer lateral window.

## Frozen stock-arm held results

The rows are paired on 16 new held seeds, 247201–247216. Counts are longest consecutive referee-verified rebounds per episode.

| Observation and head | Mean | Median | Maximum | At least 10 |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 2.38 | 2 | 3 | 0/16 |
| Clean state, generic head | 11.00 | 14 | 15 | 10/16 |
| Clean positions with the raw difference window, generic head | 10.63 | 13.5 | 15 | 10/16 |
| Clean state, supplied vertical pair and scales | 10.88 | 14 | 15 | 10/16 |
| Clean state, Fourier ridge | 6.25 | 5 | 15 | 4/16 |
| 192×144 wrist, raw differences, generic head | 5.81 | 4 | 14 | 4/16 |
| 192×144 wrist, fitted causal motion, generic head | 6.56 | 4.5 | 14 | 5/16 |
| Blind fitted-motion pupil | 1.56 | 1.5 | 5 | 0/16 |
| 384×288 wrist, same frozen checkpoint | 6.31 | 4 | 13 | 5/16 |

Applying the identical raw difference window to clean positions retained most of the clean-state performance. The loss was larger when those inputs came from wrist reconstruction. The fitted motion map reduced first-axis velocity RMS error from 0.235 to 0.031 m/s on two separate teacher validation trajectories, while vertical velocity RMS error increased from 0.635 to 0.722 m/s. Doubling the rendered dimensions with the same checkpoint produced no improvement in the held mean. [Paired results](/workspace/Fly-Connectome-m1b/artifacts/tt24/bounce-ident.json.gz) · [Resolution comparison](/workspace/Fly-Connectome-m1b/artifacts/tt24/bounce-ident-resolution.json.gz)

Loading the exported pupil and running fresh seed 247202 live reproduced all 300 commands and 14 consecutive rebounds exactly. Run `python -m scripts.experimental.tt24_bounce_ident` to train and evaluate, then add `--compare-resolution` for the frozen resolution comparison. [Exported wrist pupil](/workspace/Fly-Connectome-m1b/artifacts/tt24/bounce-ident-pupil.npz)
