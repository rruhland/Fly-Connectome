# Blade rotation around the stock handle did not rescue two actual low balls

A pure handle-axis rotation changes which wrist roll produces a given blade face. The tool axis is parallel to the SO-101 wrist-roll hinge, which has about 320° of travel. The two surfaces of this cylindrical blade are physically equivalent, so an admissible stock wrist angle can represent the tilted blade plane up to sign at every fixed upstream pose. The measured normal agreement after compensating stock wrist roll was within floating-point error. The grip's roughly 8 mm eccentricity still changes the compatible center when the wrist moves. For the measured −45° and +45° equivalents the shift ranged from about 6.05 mm without a face wrap to 14.60 mm for a π-wrapped equivalent. This leaves room for timing and clearance effects but does not create a new pitch freedom.

The diagnostic changed only the blade's local quaternion in isolated model copies, around the existing local handle axis. The blade center, handle, mount, dimensions, entire captured mass and inertia properties, actuator and joint limits, clamp, playing surfaces and opponent stayed identical. The paddle assembly remained .115 kg. Holding the original inertial axes was deliberate to isolate presentation. This was not a new physical mass-property model or a change to the qualification bench. [Fixed same-offer assignments](protocol.json)

Both developmental openings reproduced the actual native far impact, near bounce and incoming crossing under stock, −45° and +45°. All six fixed-command physical realizations missed. All 5,279 full-route CAD samples were nonpenetrating. On 11846 (projected x = −.011 m, h = .979 m), the original glancing contact barely changed its backward motion, −45° changed the glance to a steeper downward miss and +45° produced no blade touch. The finite signed-face coach found no qualifying geometric basis at any angle. On 14000 (x = −.078 m, h = .992 m), the three rotations yielded two, two and one fitted interior contact planes respectively. The −45° fit reduced one face's vertical normal component from about .38 to .28, but its lateral component remained about .52. The coach predicted 108 contacts across 576 native trajectories and no legal landing. No new model-qualified physical candidate existed. [Physical command tapes and shape audit](results.json.gz)

This is evidence against axial rotation as the main missing capability on these two offers, not against every possible changed grip. A new hardware transfer study that explicitly changes the blade pitch relative to the shaft would ask a different question. The current benchmark is untouched.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt57_handle_rotation run --out /tmp/tt57-handle-check.json.gz
python -m pytest -q tests/test_tt57_handle_rotation.py tests/test_tt56_central_training.py
```
