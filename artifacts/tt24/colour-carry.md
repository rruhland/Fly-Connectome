# Repainting the wrist-bounce pupil's scene

The metric wrist pupil normally excerpts an orange silhouette. This test replaces its fixed color mask with an appearance template acquired from observed motion. The pupil's existing subpixel silhouette fit, metric calibration, motion readout, and motor weights are retained. Its source exposure was four teacher trajectories with privileged perception supervision. At test time it receives wrist RGB and own joint angles. [Original wrist pupil](bounce-ident.md)

The simulator paints the moving ball violet and the paddle green in the changed scene. The native palette and the repaint both hide the five purely visual tabletop stripes. The paddle has no fiducial. The repaint changes material RGB on the actual simulated geoms. The physical state is identical at each paired release, and the exact executed rate tapes reproduce all referee events.

For each palette, the observer gets two short unlabelled drops with zero joint-rate commands. Generic connected-region tracks compete by independent motion relative to other scene regions. The selected track's contour pixels provide a circular distribution of opponent chroma. The template freezes after these clips. On held frames the largest connected match supplies its pixel silhouette to the existing ellipse-to-metric readout. Commissioning receives no object IDs, spatial labels, marks, or scored outcomes. It assumes one dominant independently moving, chromatically distinct region. Runtime association assumes that object's matching component remains the largest. A new palette receives its own two clips. The wrist model still uses a calibrated apparent-size relationship and an encoder-derived paddle estimate. [Implementation](../../scripts/experimental/tt24_colour_carry.py)

The paired assessment uses 16 fresh physical starts. A fixed orange-mask baseline sees both palettes, and the same motor and metric weights also receive the learned template in both. A zero-rate control runs on each physical start. Longest consecutive referee-qualified rebounds count as the outcome.

| Palette and silhouette | Mean longest run | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Native, fixed orange mask | 6.375 | 15 | 5/16 |
| Native, anonymous template | 6.063 | 15 | 5/16 |
| Violet ball and green paddle, fixed orange mask | 1.938 | 4 | 0/16 |
| Violet ball and green paddle, anonymous template | 6.375 | 15 | 4/16 |
| Zero-rate control | 2.375 | 4 | 0/16 |

The repaint template found a silhouette in 2,592 of 2,674 frames, and the fixed orange mask found zero in 1,135 frames. On the repaint, the template improved 13 paired drops, tied two and lost one against the fixed mask. Against the native fixed-mask baseline it won eight, tied three and lost five, with the same mean. All 80 action tapes reproduced the referee results and terminal events.

The four commissioning clips contained 104 observed frames in total. Independent motion identified the selected region on 24 frames per clip. Template evaluation averaged 1.33 ms per frame on the repaint at 192 × 144, excluding rendering and control. These results cover a new saturated palette after appearance commissioning. The metric decoder still comes from the original supervised calibration, and the test supplies a single distinct mover during commissioning. [Raw tapes, proof, and per-drop scores](colour-carry.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_colour_carry --output /tmp/colour-carry-reproduction.json.gz
```
