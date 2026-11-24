# Learning actuator response from encoder consequences

A direct table-tennis controller recovered one additional legal return after learning a changed actuator response from its own motor probes. A separate encoder probe showed a reduction in joint-displacement prediction error from 12.278 mrad to 0.031 mrad. The change in game score came from a second ball in one of 16 novel games. [Implementation](../../scripts/experimental/tt34_body.py) · [Probe measurements](body-calibration.json.gz) · [Matched game results](body-results.json)

## What the body model learned

The starting pupil issues six normalized joint-rate commands from the current stereo-derived ball position and velocity, measured joint angles and its previous command reference. Its trained sensor-to-joint coefficients were frozen throughout. An evaluation change multiplied the MuJoCo plant’s command-to-position-target increments on the five playing joints by 0.80, 1.20, 0.80, 1.22 and 0.83. The compensation layer had access to its own nominal commands and encoder readings. [Raw pupil and its training](../tt33/joint.md) · [Frozen checkpoint](../tt33/joint-raw.json.gz)

At each drive setting, the arm issued three pairs of small symmetric pulses near its ready pose. Each pulse lasted nine control ticks, followed by 27 ticks to settle. The ball had yet to reach the near arm. Least squares fitted five coefficients from the known command differences and the measured encoder differences. Relative to stock, the fitted coefficients were 0.8000, 1.1998, 0.8002, 1.2200 and 0.8300. The calibration used no legal-return reward, camera ball measurements, teacher outputs or true gain values.

A new signed pulse pair identified which stored response fit the observed arm. A separate signed pair assessed forward prediction independently. Before play, each probe-based condition physically repeated the identification pulse and selected its memory by encoder prediction error. The selected memory rescaled every requested joint command by the ratio of the stock to selected response, within the command bounds. [All calibration and probe traces](body-calibration.json.gz)

| Physical body | Held displacement error with stock memory | Held displacement error with changed memory |
| --- | ---: | ---: |
| Altered drive | 12.278 mrad | 0.031 mrad |
| Restored stock drive | 0.030 mrad | 12.266 mrad |

## Live RGB games

The novel roster was sealed before learning and selected using only neutral first ball deliveries. It contained six left, four middle and six right deliveries on releases disjoint from the raw pupil’s training and its two earlier evaluations. The frozen and adapted pupils received identical first deliveries on all 16 cards, including the far contact, near bounce and receiving-line measurement. All later balls came from the live opponent and could differ after the first physical return. [Frozen 16-release roster](body-novel-plan.json.gz)

| Physical body and controller | Near legal returns | Alternating legal returns |
| --- | ---: | ---: |
| Stock drive, frozen policy | 6 | 28 |
| Altered drive, frozen policy | 5 | 26 |
| Altered drive, learned body response | 6 | 28 |
| Restored stock drive, original eight releases | 3 | 14 |

The paired difference came from release 3025. Both arms landed the first left reply. The adapted arm landed the next left reply and the game reached five legal alternating returns. Across 654 matched active ticks before the first near paddle contact, encoder RMS deviation from stock fell from 71.75 mrad to 0.053 mrad. Under the altered drive, the adapted pupil reached 4.67 rad/s peak measured joint speed versus 4.87 rad/s with the frozen pupil. The adapter rescaled 1,152 command vectors. Its output reached the normalized cap on 121 ticks. [Frozen altered-drive commands](body-novel-changed.json.gz) · [Adapted altered-drive commands](body-novel-adapted.json.gz) · [Stock commands on the same releases](body-novel-stock.json.gz)

After restoring the stock drive, the probe selected the original response. Each of the eight previous stock games reproduced its earlier commands, physical events, placements and score. Command-only replay verified all 56 games and found zero penetrations across 60,811 full-CAD physics steps. Minimum clearance was 18.6 mm. The experiment tests one stationary diagonal change in the actuator interface and deliberately collects small body probes. Its behavioral gain is one later return. Generalization to coupled loads, changing dynamics and learning directly from rally movements remains open. [Restored-drive tapes](body-retained.json.gz) · [Prior tapes](../tt33/joint-fresh-raw.json.gz) · [Adapted-drive audit](body-novel-adapted-audit.json) · [Frozen altered-drive audit](body-novel-changed-audit.json) · [Stock audit](body-novel-stock-audit.json) · [Retention audit](body-retained-audit.json)

## Reproduce

Use the pinned measured launches from the [endurance reproduction instructions](../tt29/endurance.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt34_body plan /tmp/tt25-rallies.json artifacts/tt34/body-novel-plan.json.gz
python -m scripts.experimental.tt34_body calibrate /tmp/tt25-rallies.json artifacts/tt34/body-calibration.json.gz
for mode in stock changed adapted; do
    python -m scripts.experimental.tt34_body trial /tmp/tt25-rallies.json artifacts/tt34/body-novel-plan.json.gz artifacts/tt34/body-calibration.json.gz "artifacts/tt34/body-novel-$mode.json.gz" "$mode"
done
python -m scripts.experimental.tt34_body trial /tmp/tt25-rallies.json artifacts/tt32/fresh-plan.json.gz artifacts/tt34/body-calibration.json.gz artifacts/tt34/body-retained.json.gz restored
python -m scripts.experimental.tt34_body summarize artifacts/tt34/body-results.json
for mode in novel-stock novel-changed novel-adapted retained; do
    python -m scripts.experimental.tt34_body replay "artifacts/tt34/body-$mode.json.gz" "artifacts/tt34/body-$mode-audit.json" --cad
done
```
