# Learning monocular ball range from paired RGB

On eight disjoint measured flights, the existing single-left-camera tracker estimated receiving-window position to a median error of 2.71 cm. The learned silhouette correction increased that error to 3.53 cm and the velocity error from 0.22 to 0.34 m/s. It also worsened direct image-based range before temporal fitting. The results did not support advancing this candidate to continuous points. [Per-frame assay](tt29_mono_assay.json.gz)

| Range model | Median single-image range error | Median receiving position error | Median receiving velocity error |
| --- | ---: | ---: | ---: |
| Existing geometry and hard silhouette | 4.22 cm | 2.71 cm | 0.221 m/s |
| One global scale fit from paired RGB | 4.00 cm | 2.92 cm | 0.213 m/s |
| Seven-feature silhouette correction | 5.78 cm | 3.53 cm | 0.338 m/s |

Five-point-six simulated seconds of previous physical stereo play on opening 7345 gave 324 triangulated RGB pairs. The regression updated a small regularized linear system from those optical labels and used the known ball diameter and calibrated camera as a prior. Its features measured hard radius, soft color flux, spread, brightness and pixel phase. The tracker used the same causal drag, gravity and table-rebound fit with a stronger range term calibrated from the training residuals. The scalar fit used the median ratio between the existing range and the stereo-labeled range. The simulation ball state served only to score the disjoint assessment. All three trackers provided estimates for all 196 receiving-window frames. [Range fitter](../../scripts/experimental/tt29_mono_range.py) · [Reproduction](../../scripts/experimental/tt29_mono_assay.py)

On the stereo training labels, the nonlinear image features reduced median single-frame error by about 0.15 cm. That small in-sample change did not carry over. The direct range regression worsened on held frames before the stronger weight affected the temporal tracker. On the same held tape, the global scale slightly improved single-frame range and velocity while worsening receiving position by 0.21 cm. The geometry-only baseline already resolved most of the depth uncertainty relevant to these receiving windows.
