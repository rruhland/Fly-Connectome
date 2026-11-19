# Hidden button remapping during a continuing rally

Twelve paired native Pong trials started independently from the visual button learner’s exact end-normal scout checkpoint. Both arms had normal controls for four simulated seconds before an execution-only cycle of all three requests. Each trial ended at the first score or at 20 seconds. All 12 pairs entered the changed controls with identical 200-camera public prefixes and identical first changed-contact times. The frozen copy kept its acquired effects and had no further probes.

| Arm | First changed return | Later changed returns | Correct ordering in the same game | Correct ordering before first changed contact |
|---|---:|---:|---:|---:|
| Live | 4/12 | 4/7 | 1/12 | 0/12 |
| Acquired frozen | 4/12 | 0/4 | 0/12 | 0/12 |

On the successful seed 1301527667, both arms returned the first changed contact at 0.267 seconds after the switch. The live controller completed all three displacement measurements and had the correct request ordering by 2.258 seconds. It returned the second contact at 3.800 seconds and all four later opportunities on that start. The frozen controller missed the second contact. All four live later returns came from this one start.

The fixed test used 8,303 transition cameras and 24 initial cameras in 35.093 seconds. It establishes that this learner can recalibrate in a continuing game given time, with one success across the twelve ordinary mid-rally switches. There was no tuning or second serve.

The runner is `scripts/experimental/run_pong_mid_rally_remap.py`. The detailed receipts, schedules, and pinned spec are in `/workspace/m1b-experiments/pong-mid-rally-button-remap-20261119-r1`. The spec SHA-256 is `92e6f3696fa413849b195ce5781a29bad96a7b5459744b49de041a9d4c40ffd4`. The summary SHA-256 is `740e9d0bd2491edae1464e8c7d719d7a4f761f078b40e1d8b8da2046bf07b738`.

## Shorter probe development scout

A single follow-up ran a shorter calibration on the same twelve starts. It kept the contradiction trigger and inherited steering, began probing at any directly observed paddle position, and estimated each effect by contrasting movement in the first four and last eight cameras of a twelve-camera hold.

| Live controller | First changed return | Later changed returns | Correct ordering in the same game | Correct ordering before first changed contact |
|---|---:|---:|---:|---:|
| Standard | 4/12 | 4/7 | 1/12 | 0/12 |
| Shorter probes | 6/12 | 10/13 | 3/12 | 2/12 |

All normal prefixes and first changed-contact clocks matched. The two added first returns followed correct ordering at 1.033 and 1.100 seconds, before contacts at 1.583 and 2.108 seconds. The third successful calibration finished at 1.417 seconds after the first changed contact. The shorter controller used 277 post-change probe frames across all starts, compared with 263 for the standard controller. The reused starts make this a development comparison.

The runner is `scripts/experimental/run_pong_fast_mid_rally_scout.py`. Results and pinned inputs are in `/workspace/m1b-experiments/pong-fast-mid-rally-scout-20261119-r1`, with 5,878 transition cameras, 12 initial cameras and 24.884 seconds of wall time. The summary SHA-256 is `7a9968316e281cfa1a0e21500e6956a6cfe0853a7a2f542e1efa5bc5366f9944`.

## Fresh head-to-head comparison

The unchanged standard and shorter controllers then ran 24 new matched starts. All 48 trials switched at four seconds, with exact pairwise normal-prefix and first changed-contact parity. Each trial independently loaded the same end-normal scout checkpoint.

| Controller | First changed return | Later changed returns | Correct ordering in the same game | Correct ordering before first changed contact |
|---|---:|---:|---:|---:|
| Standard | 8/24 | 15/19 | 5/24 | 3/24 |
| Shorter probes | 10/24 | 22/26 | 6/24 | 3/24 |

The shorter controller gained two first returns and lost none. One gain followed an earlier valid calibration. The other occurred without completing a valid calibration. On the three starts where both controllers found the right order, the shorter probes gained 0.717, 0.725 and 1.000 seconds. Across all starts they used 572 probe frames compared with 475 for the standard controller. The fresh comparison supports faster successful recalibration and a modest return improvement. Most starts still ended without finding the correct order.

The runner is `scripts/experimental/run_pong_mid_rally_head_to_head.py`. The detailed results are in `/workspace/m1b-experiments/pong-mid-rally-head-to-head-20261119-r1`. The comparison used 22,263 transition cameras and 48 initial cameras in 50.480 seconds. The spec SHA-256 is `1e1b271c9b802bb0e29e4bcd4fd0d4103e80b49e38490dfe311695df215f3e2e`. The summary SHA-256 is `dd5db6b90ddc7ea4d92ebbc729cfb03f30daf948b903cfe2037b88166eb219a9`.
