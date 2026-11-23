# A broader repertoire for receiving recorded shots

Adding 13 demonstrations increased legal returns from one to four on the nominal mount and from one to five on the changed yaw mount. Each comparison used the same 136 held feeds, and both retained the original legal return. [Paired scores and replay receipts](match-broader.json.gz)

| Mount | Nearby feeds, six to 19 postures | Unrestricted feeds, six to 19 postures | Held contacts, six to 19 postures |
| --- | ---: | ---: | ---: |
| Nominal | 0/72 → 2/72 | 1/64 → 2/64 | 20 → 26 |
| Yaw +0.22 rad with bounce-optical transport | 0/72 → 3/72 | 1/64 → 2/64 | 20 → 26 |

We added legal receiving postures from raw DeepMind launches to the six-example TT25 camera pupil. From a seeded, randomized stream of recorded feeds, we kept the first 64 with at least one geometric posture proposal. The course excluded all 136 frozen assessment IDs, all shot IDs found in the existing TT25 recorded receipts and launch vectors that matched any exclusion to six decimal places. The selection used ball-only physical flight and arm geometry. The training court started the nominal SO-101 at its normal home position and admitted a posture only after a legal landing. Each admitted practice shot replayed with the same referee events while saving its early stereo image signature. [Training roster and physical trials](match-broader.json.gz)

Course construction screened 134 ball-only trajectories. Learning the 13 extra postures took 1,238 simulated arm trials and 13 verification replays. At most 24 ranked postures were tried on each practice feed. The teacher used the simulator's ball state for ranking, and the deployed pupil used four stereo frames. [Recorded probe expense and checkpoint](match-broader.json.gz)

The expanded and original pupils used the same camera features, nearest-neighbor rule and bounded joint commands. On the changed yaw grip, both used the same geometric transport from the optical calibration acquired during solo bouncing. Both banks and the grip condition were fixed before assessment. The held roster contained 72 real launches near the original teaching feeds and 64 real launches drawn without that similarity filter. The expanded bank added legal returns on IDs 14302, 14835 and 5956 at both mounts. It added 6209 on the yaw mount. Every held success replayed with the same referee result within this run. On the yaw mount, the six-posture bank reproduced all historical legal returns, contacts, incoming landings and frame counts. Nineteen historical event checksums differed. [Frozen held roster](fresh-recordings.json.gz) · [Optical bounce calibration](optical-face.json.gz) · [Expanded nominal checkpoint](match-broader.npz)

```bash
MUJOCO_GL=egl PYTHONPATH=src python -m scripts.experimental.tt26_match_broader /path/to/rallies.json
```
