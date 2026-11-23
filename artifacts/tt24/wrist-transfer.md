# What transferred to and from the wrist

The same five-joint action readout can be used with different sensory calibrations. Four ordinary drops with stereo reference cameras supplied 248 frames (4.13 simulated seconds) of wrist RGB, joint angles and stereo RGB on the physically altered arm. The source skill was taught earlier with four motor-labeled demonstrations and privileged stock calibration. On 16 new changed-arm drops, wrist-only deployment after public target calibration produced a mean longest run of 6.38 versus 4.44 frozen. The updated map improved 11 drops, tied two and lost three. Three of sixteen reached ten. The spectator cameras were used during calibration only. [Public stereo fit, controls and weights](metric-transfer.md) · [Independent visual registration study](metric-stereo.json.gz) · [Calibration and learning plot](transfer-progress.png)

## Fifteen-second horizon

The original five-second cap could conceal long runs. We froze the previously tested policies and extended the maximum to 900 control frames. The rows are separate cohorts for stock and alternate. Every listed episode reproduced all physical scorer records from its commanded joint sequence. Timeouts give a lower bound on potential streak length.

| Body and wrist calibration | Fresh drops | Mean longest run | Best run | Reached 10 | Survived all 15 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 20 | 14.30 | 42 | 9 | 6 |
| Changed arm, uncorrected stock calibration | 16 | 3.50 | 7 | 0 | 0 |
| Changed arm, four demonstration-supervised calibration drops | 16 | 7.81 | 38 | 3 | 1 |

A new online motion filter that selects among six polynomial extrapolators by their next-observation errors helped on development. On the above held cohorts it reduced the stock mean to 10.15 and the uncorrected changed-arm mean to 2.56. Learning a velocity estimator from a native Pong player's optical tracks improved held native tracking RMS from 0.208 to 0.152 normalized units. Its additional stock-arm benefit over shuffled temporal labels was inconclusive (163 vs 159 consecutive rebounds summed across 16 fresh drops, against 110 with the online filter alone). The altered-arm totals were 42, 49 and 44. This comparison supplies the same four-demonstration stock motor head to every condition. [Stock 15 s tapes](forecast-long-stock.json.gz) · [Altered 15 s tapes](forecast-long-alternate.json.gz) · [Recalibrated 15 s tapes](forecast-adapted-alternate.json.gz) · [Native estimator](pong-optics-source.json) · [Stock transfer](pong-optics-stock.json.gz) · [Altered transfer](pong-optics-alternate.json.gz)

## Carrying a shared action readout through repeated body switches

Four reset camera snapshots per learned calibration supplied grayscale/edge reference images. A visual comparison selected the calibration from the current wrist image, refreshed every six frames. Both calibrations fed the identical action readout. In 24 fresh drops alternating between two previously encountered bodies, it picked the corresponding map at the first frame on all 24. Total longest-run counts over twelve stock and twelve changed releases were 97 and 56. Permanently using the stock map scored 97 and 47. Permanently using the last learned map scored 38 and 56. Averaging the two sensory estimates scored 79 and 83, higher in aggregate and lower on restored stock. These checks establish recall across these visually distinct configurations. They do not show recognition of a new body with the same appearance. [Paired per-frame choices, commands and scores](view-recall.json.gz)

Four new, stationary target calibration drops were also tested with the wrist alone. Aligning wrist estimates around the time of scalar rewards reduced 12-drop total longest runs from 63 frozen to 36. Shuffling reward times gave 38. Aligning event averages alone could confuse a change in the release distribution with a change in the camera. [Single-eye event-alignment trial](reward-registration.json.gz)
