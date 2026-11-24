# Preparing for the next return from stereo RGB

The adaptive controller kept the 65 ms post-stroke recovery from TT31. It recognized a new approach near or beyond the net after seeing an earlier ball reach the arm or depart after an attempted stroke. In the gap before it had a verified stroke for the new approach, it forecast the receive position and velocity from recent stereo images. It selected nearby rewarded movements and used their joint targets approximately 100 ms before contact to seed inverse kinematics. It only averaged poses that were close in all five joints. When the first preparation passed its check, the arm began moving three control ticks before trying the next stroke. Fresh captures could update the preparation while the stroke search was pending. Incremental target speeds were capped at 35% of the nominal command rate, and the private arm model checked clearance, motor speed and premature paddle contact over five control ticks before issuing a move. [Controller](../../scripts/experimental/tt32_anticipate.py)

The first comparison uses the eight continuously varied points from TT31. After a pilot showed that requiring a new approach on the far half could omit short replies, we widened the trigger to accept them near the net. We froze the resulting controller for the new openings. Within each comparison both controllers had the same left-ready posture, frozen 238-movement checkpoint, optical error memory, one-time releases and opponent seeds. Stereo runs at 30 Hz and joint commands at 60 Hz. The adaptive controller starts after the arm's first stroke and leaves a verified stroke in control as soon as one is available. [Fixed recovery tapes](../tt31/recovery-full.json.gz) · [Matched live comparison and replay checks](anticipate-full.json.gz)

| Varied placement course | Arm's legal returns, fixed / adaptive | Alternating returns, fixed / adaptive | Points with changed commands |
| --- | ---: | ---: | ---: |
| Earlier openings, eight points | 6 / 8 | 20 / 24 | 2 |
| New openings, eight points | 4 / 4 | 16 / 16 | 3 |

On opening 3855 the second far hit, near-table bounce and receive-line crossing were identical. Adaptive preparation started 127 ms after that far hit. After three control ticks the adaptive paddle center was 9.9 mm closer to the table centerline than the fixed paddle. Both variants touched the ball 2.6625 s into the point with identical incoming velocity. The adaptive contact produced a legal return and the arm landed the subsequent ball as well. The fixed point ended after its first legal arm return. [Paired action and contact tapes](anticipate-full.json.gz)

In a repeat on 3855, the same optical gate delayed planning by three control ticks while issuing zero preparation commands. This also raised the score from one to three arm returns and from three to seven alternating returns. It followed a different command and contact trajectory. The planning delay was sufficient to reproduce the scoring gain on this opening. The paired replay checked 4,470 physics states with zero penetrations. [Delay-only control](anticipate-delay-control.json.gz)

The eight new openings used the same controller settings. Preparation changed three command tapes, with eight accepted motion updates. All eight paired scores tied. All 32 plays across the two schedules reproduced their referee events and physical placement records on action-only replay. The full-CAD check found zero penetrations over 42,766 physics states, with a nearest checked gap of 18.6 mm. [Fresh fixed tapes](fresh-fixed.json.gz) · [Fresh adaptive tapes and CAD evidence](anticipate-fresh.json.gz)

## Reproduction

Use the pinned DeepMind feeds and environment in the [endurance instructions](../tt29/endurance.md).

```sh
python -m scripts.experimental.tt32_anticipate /tmp/tt29-feeds.json
python -m scripts.experimental.tt32_anticipate /tmp/tt29-feeds.json \
  --source artifacts/tt32/fresh-fixed.json.gz \
  --output artifacts/tt32/anticipate-fresh.json.gz
python -m scripts.experimental.tt32_anticipate /tmp/tt29-feeds.json --delay-control 3855
```
