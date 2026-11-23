# Recalibrating a moved wrist camera from the wrist image

Two generic target-arm movements supplied 192 control ticks and **96 wrist RGB frames** over **3.2 simulated seconds**. Four stationary boundaries visible in both environments supported a camera-mount and link calibration. The fitted map reduced projection residual on a wholly withheld movement from **1.17 to 0.29 pixels**. Target acquisition used only RGB and joint encoders. It observed no gameplay rewards and received no external camera frames, fiducials, demonstration commands or pose labels. [Fitting receipt](wrist-egomotion-fit.json.gz) · [Recorded RGB, encoders and commands](wrist-egomotion-clips.npz) · [Frozen map](wrist-egomotion-map.npz)

## What the camera can see

The blue tabletop and the dark background each contribute two natural boundaries. Robust line fitting excludes visible occlusion by the arm, paddle or ball. The table's short border is often occluded, so the fit uses the independently visible edges available in each frame. The scene map came from 144 new wrist frames in three stock movement clips. This stage used the previously acquired stock camera registration and vendor-link model for metric scale. During target adaptation the scene map was held fixed. The same table and background remained visible in the changed-arm environment. [Stock calibration and its earlier stereo acquisition](public-body.md) · [Acquisition and geometry fit](../../scripts/experimental/tt24_wrist_egomotion.py)

The physical-motion budget comprises two 96-tick target fit clips, a third 96-tick target clip held out to test reprojection, and three 96-tick clips for the stock map. The controller in these clips used generic bounded encoder movements independent of the ball. Only two target clips determined the parameters. The camera-only fit allowed a rigid rotation and translation relative to the stock wrist frame. The combined fit also estimated small corrections to stock link lengths and joint axes. Both used the same wrist images and learned 3D lines. Neither received target geometry readings.

| Frozen target map | Fitting reprojection RMS | Withheld motion RMS |
| --- | ---: | ---: |
| Stock, untouched | 1.49 px | 1.17 px |
| Refitted camera mount | 0.61 px | 0.54 px |
| Refitted mount and links | 0.25 px | 0.29 px |

Four more held reaches were scored against the simulator's camera pose after the fit was frozen. The unmodified map had 56–59 mm position RMS and 2.53–3.10° attitude RMS. The mount-only fit reached 11–13 mm and 2.40–2.97°. The combined map reached 19–20 mm and 0.37–0.47°. The mount-only fit was more accurate in absolute position. The combined map tracked attitude more closely as the joints moved. These camera poses were used only in this diagnostic. [Held camera metrology](wrist-egomotion-diagnostic.json.gz)

## Physical wrist-only bounce

The same stock motor fitted earlier from four demonstrations consumed the original feature representation in every active condition. The new geometry changed how the stock wrist decoder turned ball image rays and joint angles into those features. The frozen motor parameters, ball detector and velocity readout stayed fixed. For the mount-and-link fit, the optical correction also changes the inferred blade pose. This test uses a 600-tick cap and 16 new matched alternate-arm drops. It also evaluates the untouched stock policy on 16 new stock drops. [Stock policy provenance and earlier supervision](bounce-ident.md) · [Game and replay driver](../../scripts/experimental/tt24_wrist_egomotion_trial.py)

| Changed arm on 16 paired starts | Total rebounds | Sum of longest streaks | Mean longest | Maximum longest | Reached five | Reached ten |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untouched stock decoder | 90 | 76 | 4.75 | 12 | 5/16 | 2/16 |
| Camera mount only | 107 | 84 | 5.25 | 23 | 6/16 | 1/16 |
| Camera mount and links | 146 | 87 | 5.44 | 10 | 8/16 | 2/16 |
| Zero command | 37 | 37 | 2.31 | 3 | 0/16 | 0/16 |

The mount-only map improved six matched starts, tied four and lost six. The combined map improved six, tied three and lost seven. Pair-resampling 95% percentile intervals for the mean change in streak were −1.94 to +3.13 and −0.94 to +2.44, respectively. This cohort supports camera relocation and individual successful motor transfer. It leaves the average game benefit uncertain. The combined map scored 146 total rebounds, including runs with intervening contacts that failed the consecutive-streak criteria. Two combined-map runs reached the 600-frame cap. [Complete game judges and command tapes](wrist-egomotion-trial.json.gz)

On seed `368112` the mount-only fit scored 28 total physical rebounds with a longest consecutive streak of 23. The untouched stock decoder scored nine consecutive and holding scored two. This is the best mount-only held run. [Matched wrist-camera replay](wrist-egomotion-held.mp4)

## Recall when the original wrist returns

A separate selector compared static-edge reprojection under both carried calibrations using one reset RGB frame and the current encoder packet. It selected the original stock map on 16 of 16 stock starts and the fitted combined map on 16 of 16 changed starts. Labels were used only to score those choices. This requires the visible environment to remain comparable across the two calibrations. [Reset-frame choices](wrist-egomotion-recall.json.gz)

The retained stock model scored 196 total rebounds and averaged 11.06 consecutive rebounds on the 16 new stock drops, with a longest streak of 29 and 7 drops reaching ten. All 80 scored games reproduced the complete physical judge record from their joint command tapes. [Stock and changed-arm qualification](wrist-egomotion-trial.json.gz)
