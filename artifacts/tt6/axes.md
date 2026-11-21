# Transfer to held-out physical joint axes

Two new installations use the five-channel offset receiver. Within each installation, four MuJoCo joint directions change by 13° to 19°. The sweep hinge and its passive counter-rotation share the changed axis. Identical five-frame command pulses with matched physics seeds produced 1.8–7.1 cm differences in the paddle center’s displacement. Reflected cached rendering matched the original shaded native stereo images byte for byte on both tilted models in a separate probe test.

Each condition paid eight camera probes per installation and used the same freshly fitted local motor model. The saved player began with gameplay, rebound and latency experience from 42 points of varied hall practice. The mini-only player began with the first-person mini checkpoint. Both updated online. Across the two installations each received the same 14 scored starts, including four combining altered rubber, command delay and noise.

| Installation | Starting experience | Points with a legal return | Legal returns | Points with at least two legal returns | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Tilt A | Varied hall practice | 5/7 | 7 | 1 | 2 |
| Tilt A | Mini only | 3/7 | 3 | 0 | 2 |
| Tilt B | Varied hall practice | 3/7 | 4 | 1 | 0 |
| Tilt B | Mini only | 0/7 | 0 | 0 | 0 |

Across both installations the practiced player returned balls in 8/14 points and mini-only in 3/14. The combined-disturbance results were 3/4 and 2/4. Both players returned zero of their first four starts, counting the first two at each installation, and both finished with two wins overall. [Point and pulse receipt](axes-held.json.gz)

## Equal practice on tilted and standard axes

Two more groups started from the mini checkpoint and played eight matched practice points at two stations. One group saw two small joint-axis changes during practice. The other saw standard offset geometries. The two practice groups each paid 544 probe commands and saw the same rubber, noise and delay conditions. They then faced the same two held-out geometries listed above after identical local RGB calibration. The practice layouts and serve starts were disjoint from evaluation.

| Prior practice | Held-out points with a legal return | Legal returns | Points with at least two legal returns | Wins |
| --- | ---: | ---: | ---: | ---: |
| Tilted axes | 5/14 | 5 | 0 | 4 |
| Standard axes | 3/14 | 4 | 1 | 2 |

The tilted-practice group returned in 4/7 and 1/7 points on Tilt A and Tilt B. The standard-practice group returned in 3/7 and 0/7. The results show a small difference in opening success across these two fixtures. Repeated returns remained rare. [Practice and held-out receipt](axes-practice.json.gz)

```sh
export PYTHONPATH=.:src:scripts/experimental
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt6_axes_trial /tmp/axes-held.json.gz --cache /tmp/tt6-axis-cache
python -m scripts.experimental.tt6_axes_trial /tmp/axes-practice.json.gz --axis-practice --cache /tmp/tt6-axis-cache
python -m scripts.experimental.tt6_axes_trial artifacts/tt6/axes-held.json.gz --verify
python -m scripts.experimental.tt6_axes_trial artifacts/tt6/axes-practice.json.gz --verify
```
