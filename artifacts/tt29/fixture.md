# SO-101 table-edge fixture

The stock-scale SO-101 base sits on a 4 mm plate at the near end of the table. Two C-clamps connect the plate to screw shoes pressed against the underside of the 50 mm tabletop. Each screw has a small T-handle. MuJoCo fixes the clamp and base to the world. On both tables the base origin is 26 mm inboard of the end and at a height of 766.4 mm. The lowest face of the base CAD sits at 764 mm. Relative to the previous installation, the arm moved 366 mm toward play and 254.4 mm up. [Regulation station](fixture-regulation.png) · [Clamp detail](fixture-clamp.png)

The wrist carries a 30 × 28 mm board with a short black lens barrel on a white angled bracket fitted against the gripper housing behind the jaws. The aperture is rigidly mounted at (+6, +54, −35) mm in the stock gripper frame. The native aperture is at the front of the lens, with a 3 mm near plane and a vertical field of view of 124°. The same mount follows the arm during receiving and solo bouncing. The bracket, jaws, paddle, and court appear in the RGB feed when they enter the aperture’s field of view. [Board detail](fixture-wrist-board.png) · [Regulation wrist RGB](fixture-wrist-rgb.png) · [Solo station](fixture-solo.png) · [Solo wrist RGB](fixture-solo-wrist-rgb.png)

The regulation ready pose puts the paddle at approximately (−.001, −1.051, 1.127) m and points the wrist camera across the court. The paddle is canted at rest. Solo bouncing retains its horizontal drop posture and samples the release relative to the blade.

## Side-on grip and clearance

The optional side-on grip turns the original paddle 90° across the jaws. At reset, rays from the handle axis hit the full-CAD finger surfaces 8.500 and 8.516 mm away, matching the handle’s 8.5 mm radius. The pose holds the blade center at approximately (−.009, −1.336, 1.040) m with the face tilted up toward play. The simulated grip uses a compliant weld while the jaws are closed. The camera retains its fixed relation to the wrist. [Side-on grip](fixture-side-on.png)

The clearance check uses signed distances between convex MuJoCo representations of the CAD. It tests nonadjacent links and contacts against the table, clamp, and net, with a 3 mm penetration tolerance. A deep folded pose at joint angles (−.097, 1.745, 1.093, 1.584, −.642) rad puts the blade close to the court but intersects the shoulder and base by up to 33 mm. The side-on ready pose had no checked intersections at reset or in 20 uncommanded control steps. The beveled camera foot and bracket cleared the moving jaw in a sweep across its full joint range. [Clearance check](../../scripts/experimental/tt29_arm_clearance.py)

## Entry points

- Use `ReferenceReceiver(view_mode="wrist+fixed")` from `scripts.experimental.tt25_reference_receiver` for the regulation court and all three RGB streams. `render_wrist()` also exposes the native wrist feed with the fixed-camera stream selection. `camera_calibration()` gives intrinsics and current world pose for each selected stream. The camera-to-world rotation has right, up and backward columns.
- Pass `paddle_grip="side-on"` to the regulation constructor to select the lower ready pose and sideways grasp.
- Use `BounceArena()` from `scripts.experimental.tt24_bounce` for solo RGB and joint readouts. `stock(fixture="table-edge")` from `scripts.experimental.tt24_public_body` equips the solo station with the existing fixed training views.
- Pass `fixture="archived"` to reproduce the previous geometry programmatically. Set `SO101_FIXTURE=archived` on a historical experiment command whose simulator selection was implicit. The original TT20 configurations with explicit outboard, raised, ultrawide or legacy optics still imply their archived fixture.

To render the frames in the MuJoCo environment, run `MUJOCO_GL=egl PYTHONPATH=.:src:scripts/experimental python -m scripts.experimental.tt29_fixture_views` at the repository root. The frames show the CAD models during a seeded physical regulation opening, at the side-on ready pose, and during a seeded solo drop. [Render script](../../scripts/experimental/tt29_fixture_views.py) · [Fixture tests](../../tests/test_tt20_fixture.py)
