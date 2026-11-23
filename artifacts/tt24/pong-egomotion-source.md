# Camera-motion compensation in native Pong

Full geometric compensation gave the best terminal score on 64 matched native points viewed from a paddle-mounted camera. A separate causal visual fit recovered a gain of exactly 1.000, the value fixed by the supplied camera geometry. The shuffled fit was 0.005. The pilot gate for further reward fitting and wrist execution stayed closed. [Registered pilot](pong-egomotion-source-plan.json) · [Native receipts](pong-egomotion-source-pilot.json.gz)

The camera translates real native rasters vertically by the observed displacement of the controlled paddle. It uses the paddle coordinate as one public encoder at the camera's known pixel scale. Padding preserves the complete view and translation is quantized to whole pixels. All variants recover the same positions and apply the gain only to the velocity contribution of camera motion. The operation is the camera-relative ball displacement plus a signed scalar times the displacement induced by moving the camera back into a fixed frame. This is the one-axis translation case of the wrist operation, which can also undo rotation using its calibrated pose. The visual fit compares consecutive mover detections from synchronized moving and stationary rasters. Its labels consist entirely of paired optical displacement differences. [Camera and source runner](../../scripts/experimental/tt24_pong_egomotion.py) · [Existing wrist calibration](wrist-egomotion.md)

| Camera-motion gain | Net points | Wins | Losses | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| −0.5 | −8 | 3 | 11 | 50 |
| 0.0 | −4 | 3 | 7 | 54 |
| +0.5 | −5 | 3 | 8 | 53 |
| **+1.0** | **+2** | **3** | **1** | **60** |
| +1.5 | −7 | 3 | 10 | 51 |

Full correction improved the native terminal score by 0.094 per point over velocity from the camera-relative displacement alone. The paired 95% percentile interval was +0.016 to +0.188. The full setting reproduced the original native command digest on all 64 moving-camera starts. All gains reproduced identical original commands on 32 separate stationary-camera starts. The paired optical fit used 95,663 usable differences, of which 55,461 contained camera movement. The geometry was supplied at the outset, so recovering unity demonstrates the expected optical relationship. The existing wrist observer already compensates for camera motion through its encoder-derived registration. [Per-start scores, paired estimates and command digests](pong-egomotion-source-pilot.json.gz) · [Wrist observer](../../scripts/experimental/tt24_public_body.py)
