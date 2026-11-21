# A Cartesian far-court practice robot

The optional far robot uses three linear slides and separate swivel and loft joints. MuJoCo computes motion from bounded position targets and motor forces. The blue paddle uses the native ball-contact geometry, and the fixture retains the SO-101 learner, table, ball, net, cameras and strict referee. The robot parks behind the far baseline until a return. [Robot and controller](../../scripts/experimental/tt22_precision.py)

The far controller alone sees simulation state. It follows the expected incoming flight until a legal far-side bounce, proposes physical swing tapes around an approximate impact solution, and ranks native MuJoCo return trajectories against a near-side receive corridor. A separate strict simulation checks the leading tapes before the chosen joint commands are applied. Its target is centered at x = 0, y = −0.91 m and z = 0.19 m above the tabletop, with 2.2 m/s toward the learner. The learning-side controller receives the two native RGB streams and its own actuator commands. After the initial serve, both robots act through their joint ports throughout one continuous point. [Trial](../../scripts/experimental/tt22_precision_trial.py)

## Closed-loop pilot

On six narrowed opponent serves, the alternating legal return counts were **6, 0, 6, 4, 4, 4**. All 12 far plans that passed the referee check also produced a legal return in live play. The SO-101 made contact on 11 of those feeds and turned seven into legal replies. The remaining feed crossed y = −0.91 m only 26 mm above the table and hit the learner's clamp beyond the table. The longest point stayed under ten returns. [Pilot receipt](precision-pilot.json.gz) · [Native far-side contact](precision-court.png)

In measured live contacts after a legal far return, the median absolute lateral offset was 40 mm, median y was −0.849 m, median height was 0.158 m, and median incoming speed along the court was 2.60 m/s. The learner intervened before the target depth on all 11 contacts. The receipt records the contact and crossing separately. Planning took 65.2 seconds of cumulative far-controller wall time, and the recorded actions and referee events replayed exactly for all six points.

## Ordinary held serves

With the original, full-width TT20 opponent serve distribution, four fresh points reached **6, 2, 6, 2** alternating legal returns. All eight issued far plans produced legal returns. The SO-101 touched all eight feeds and made four legal replies. The median contacted feed had 71 mm absolute lateral offset, y = −0.860 m, height 0.161 m, and incoming speed 2.63 m/s. The outcomes and events replayed exactly on all four starts. A separate fixture check confirmed exact initial ball position, velocity and spin against the stock arena on three ordinary starts, the five actuator force limits, and physical paddle displacement under an issued command. [Held receipt](precision-held.json.gz)

The Cartesian robot provided repeatable legal feeds on both serve distributions. The far robot won all ten points, and the longest optical rally reached six alternating returns. This fixture has no demonstrated ten-return point.

## Reproduction

Run from the repository root with the MuJoCo environment active.

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m scripts.experimental.tt22_precision_trial
```

Add `--held --count 4 --spread 1 --pace 1` for the original serve distribution. The report counts alternating legal returns after the serve. Both recorded actuator tapes can be replayed without running the controllers again. `--verify PATH` checks exact point outcomes and referee events.
