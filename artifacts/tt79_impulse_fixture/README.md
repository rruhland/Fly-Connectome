# Small grip and riser changes did not improve the low-ball impulse

The 24 mm outward seat and 18 mm flat riser did not earn a return on any of the three exact hard developmental flights. Stock also did not score in this finite comparison. The 27 live-plant simulation attempts preserved the original incoming far contact and near bounce. No held or full-game trial was opened. The original benchmark is unchanged.

TT72 and TT77 reached the centered stock blade face on exact difficult balls, but the contact patch did not move in the direction or at the speed of an earned lateral-clamp stroke. TT79 separates two small, physically plausible interventions from the earlier neck-angle, lateral and drop-shelf tests. The tabletop, ball, blade radius and material, standard opponent draw, servo limits and referee are unchanged. This is an altered-body experiment, not a change to the original benchmark. [TT72 contact audit](../tt72_retarget_trajectory/README.md) · [TT77 physical attempts](../tt77_centered_impulse/README.md)

- **Axial reseating** moves the same straight paddle rigidly along its existing handle. The weld translation changes, not the handle angle, blade shape, paddle mass or native principal inertia. CAD measures the length of the real shaft clamped between both existing jaws and the distance from the blade to the wrist and fingers. A new seat must match the stock 5 mm holding span and have at least 1 mm blade clearance. No claim about clamp strength or physical hardware manufacture is made.
- **Flat risers** lift the original SO-101 flange without changing its horizontal position or rotation. The rigid deck bears on the actual clamp plate and has 2 mm underside relief for its protruding original jaw tabs. The comparison checks the CAD footprint, pad and deck seating, and table/tab clearances. No mass, joint travel or rate is removed from the moving arm.

For each supported fixture the same local workspace checks ordinary-height routes and three previously paid hard delivery trajectories. The differential screen evaluates the blade patch velocity as center velocity plus angular velocity crossed with the contact offset. It requires a safe finite path to a broad-face pose and fits the measured successful donor patch motion inside unchanged signed joint-rate bounds. This intentionally optimistic screen ignores loaded rate and torque loss and does not prove real contact or a legal ball trajectory. Finite physical comparisons use the attained native drive and CAD rather than accepting this proxy. [Frozen design](protocol.json) · [Support, geometry and proxy motion](kinematics.json)

| Fixture | Ordinary geometric sites / 30 | Qualified routed donor poses on 9254 / 8831 / 15456 |
|---|---:|---|
| Stock | 25 | 0 / 3 / 4 |
| In 3 mm | Rejected | 6.42 mm wrist penetration |
| Out 8 mm | 25 | 0 / 2 / 1 |
| Out 16 or 24 mm | 24 | 0 / 2 / 1 |
| Flat rise 8 mm | 25 | 0 / 4 / 3 |
| Flat rise 18 mm | 24 | 0 / 3 / 3 |
| Flat rise 30 mm | 23 | 0 / 3 / 1 |

All three outward seats retained a 5 mm bilateral shaft band and gave 4.58–20.58 mm of blade-to-wrist separation. All three risers had at least 5.49 mm of support footprint margin and 1.00 mm of clearance from the original projecting tabs. Isolated fits on 9254 placed the ball about 15–16 mm from the blade center with a face error below 1.7° on all seven mountable configurations. None also passed the finite reach time and straight-line CAD route screen. This is not a proof that all preparatory paths are impossible. On the other two deliveries the optimistic stock patch speed already closely matched the donor. The outward and height changes improved the best patch-speed proxy by only about 0.001 m/s at most. That box-speed fit did not ensure the robot could produce the normal orientation and motion together at runtime.

The best predicted arm in each altered family faced stock at the same finite budget of 288 native forecasts and up to three actual commands per ball. The native search ran 2,592 forecasts and selected 27 physically separate commands. There were no legal returns. Twenty-five tested commands had no CAD penetration. On 8831 one candidate under each altered fixture penetrated in four solver samples and remained an adverse result. At 9254 the outward seat changed the stock rim approach into oblique edge contacts. The raised fixture retained the face but still sent the ball forward at only 1.72–1.86 m/s. On 15456 stock and both altered bodies contacted broad face. The closest stock impact was 9.3 mm from center, but the entire trial family sent the ball downward at 0.37–0.54 m/s. A clean centered impact was not enough to earn an outgoing flight. [Actual candidate tapes and referee evidence](physical.json.gz)

An independent replay reproduced all 27 event histories and checked all 20,892 CAD solver states. It measured the velocity of the *actual struck patch*, not just the motion of the racket center. On 9254 the scored lateral teacher's patch was moving up at 0.86 m/s. The stock contacts in this assay moved up at 0.21–0.26 m/s, while the outward seat and flat riser moved the patch **down** at 1.07–1.81 m/s and 0.14–0.29 m/s respectively. On 15456 the long seat did make the patch advance faster and descend less than the tested stock motions. It still never achieved upward patch velocity and all three balls departed downward. Neither body change was promoted. [Measured patch velocities and verified random draws](native-contact-audit.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt79_impulse_fixture plan
python -m scripts.experimental.tt79_impulse_fixture geometry
python -m scripts.experimental.tt79_impulse_fixture physical
python -m scripts.experimental.tt79_fixture_report
python -m pytest -q tests/test_tt79_impulse_fixture.py
```
