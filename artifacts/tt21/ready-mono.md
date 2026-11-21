# Lateral preparation with a single camera

The incoming-ball fit learned from stereo was frozen before evaluating an outside-only and a wrist-only player. Each camera had its own 12 unselected starts, half serving and half receiving. The comparison used the same strong physical opponent, strict referee, and ordinary point resets as the stereo study. The evaluation draws were disjoint across cameras and from the stereo teaching and held starts. [Frozen fit and stereo comparison](ready-position-incoming-held.json.gz)

Both versions of each player used the same unmarked 480 × 360 native RGB stream, public lens intrinsics, and table-line calibration from that stream. They used monocular temporal ranging with the existing size fallback for ball position. The receiver took the temporal velocity when available and kept its usual position-history estimate otherwise. Neither version fitted the lateral coefficients on the new camera or on target outcomes.

The comparison varied the shoulder pan after the same early incoming cue. The shifted version used the frozen fit, deadband, rotated shortlist and command-driven nominal arm. The centered version used zero pan. If the learned policy issued no pan, both programs had the same actions and the resulting tape was replayed separately. [Experiment](../../scripts/experimental/tt21_ready_mono.py)

| Camera | Starts with a pan command | Legal returns, centered → shifted | Legal answers after a rival return | Point wins, centered → shifted |
|---|---:|---:|---:|---:|
| Outside only | 3/12 | 1 → 1 | 0/4 → 0/4 | 2/12 → 2/12 |
| Wrist only | 1/12 | 4 → 4 | 0/6 → 0/6 | 4/12 → 4/12 |

The legal-return count and point winner agreed in all 24 pairs. The cue triggered on three outside points and two wrist points. One wrist prediction was inside the existing 105 mm deadband. Only four starts changed controls, and all 48 arm-by-point tapes replayed exactly. The sample gives no evidence of improved play from lateral preparation with either single camera.

## Reproduction

Run from the repository root with the MuJoCo environment active.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt21_ready_mono
```

The receipt records the fit hash, paired point seeds, optical positions and velocities, action tapes, point results, scored events and exact replay counts. Positions are metres with height zero at the tabletop. Optical velocities are metres per 30 Hz frame. [Machine receipt](ready-mono.json.gz)
