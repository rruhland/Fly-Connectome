# Identifying the SO-101 camera load from one movement

One fixed 24-frame movement identified the camera load in each fixture. The classifier compared native fixed stereo views of the tip, blade and rim marks with separately simulated RGB traces for 0, 40, 80 and 120 g at the known camera mount. It selected the lowest pixel residual when the margin exceeded 0.02 pixels. [Probe inputs, RGB measurements and scores](direct-receive-payload-adapt.json.gz)

| Fixture | Selected accessory mass | Visible marker measurements | Margin to next hypothesis |
| --- | ---: | ---: | ---: |
| Original camera and mount | 0 g | 146 | 0.165 px |
| Camera and mount with ballast | 80 g | 142 | 0.206 px |

The selected mass, center of mass and inertia entered the independent receiving simulator before play. The original dry movement catalogue still supplied the candidate shortlist. The player used the gantry priors, native wrist and fixed RGB, and the same loaded-arm fixture as the frozen comparison. The calibration used 24 target control frames per fixture. Creating the four candidate traces used 96 separate model frames. [Trial](../../scripts/experimental/tt19_payload_adapt.py) · [Frozen comparison](direct-receive-payload.md)

| Loaded receiver | Legal returns on first 12 matched starts | Points won |
| --- | ---: | ---: |
| Frozen nominal arm model | 7/12 | 3/12 |
| Optically identified load | 7/12 | 3/12 |

The inferred load changed three action tapes and all 12 forecast records. The same seven starts produced legal returns, and neither of the two losses caused by the payload was recovered. The separate six-start block was conditional on raising the matched return count and recovering a loss. This result left the fresh block unplayed. The experiment demonstrates identification within the specified slate and leaves the receiving benefit unestablished. [Full matched tapes](direct-receive-payload-adapt.json.gz)

All 12 adapted physical action tapes replayed exactly. The seed check excluded this derived receipt on rerun and still detected a matching seed in an independent source receipt. [Trial and replay](../../scripts/experimental/tt19_payload_adapt.py)
