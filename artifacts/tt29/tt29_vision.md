# SO-101 regulation vision

The near policy reads the available RGB streams, their capture timestamps, its measured joint angles, and its issued joint targets. A fixed-window ray fit estimates ball position and velocity. The moving wrist lens pose comes from the serial arm geometry and the encoder sample taken with the image. The near planner writes those estimates into an independent nominal model to evaluate strokes. The two-sided referee and the powered far carriage operate in the live simulator. The near learner retains stroke settings after a scored legal return. [Optics](../../scripts/experimental/tt29_vision.py) · [Near planner](../../scripts/experimental/tt29_vision_twin.py) · [Motor memory](../../scripts/experimental/tt29_action_memory.py)

## Replayed camera-controlled points

| Measured opening | Alternating legal returns | SO-101 returns | Median departure speed | Full-CAD physics states checked | Negative separations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7345 | 14 | 7 | 5.37 m/s | 5,222 | 0 |
| 7271 | 66 | 33 | 5.63 m/s | 21,952 | 0 |

Both points used native fixed stereo at 60 Hz simulated control. The recorded commands reproduced every contact and referee event. The episode for 7271 started with seven earlier rewarded examples and retrieved 28 strokes during play. Its 337 replayed events covered about 45.7 s. The full-CAD checks used the physical state after every 1/480 s solver step and included the initial and terminal states. The smallest checked clearance in either episode was 18.6 mm. The far side was an idealized force-driven carriage. The 100-return target remains open. [7345 verification](tt29_vision_memory_audit.json) · [7271 verification](tt29_vision_7271_audit.json) · [7345 replay](tt29_vision_memory_rally.mp4)

The 7271 result used 28 memory retrievals and six successful fallback plans. Predictive physics search still adds wall-clock delay. The recordings show the simulated clock, and the replay playback preserves that clock. [Planner timing and motion receipt](tt29_vision_7271_audit.json)

## Wrist field of view

At the tested 4:3 aspect ratio the board lens provides a vertical field of view of 124° and a horizontal field of view of 136.5°. At `EDGE_READY` the optical axis points primarily upcourt, with a forward vector of approximately `(0.008, 0.990, −0.142)`. At `lateral` it points mostly sideways at `(0.966, 0.188, −0.176)`. The `side-on` posture points upward and across the court at `(−0.404, 0.534, 0.743)`. Neither lateral opening produced a ball observation in the first 12 frames. An 11-tick shoulder pan toward 1.10 radians recovered 5 of 12 and 7 of 12 early sightings, respectively. Median near-flight position error in the forward `EDGE_READY` wrist view was 1.5 cm on 7345 and 0.7 cm on 7271. Both openings produced a native legal return from wrist RGB in that posture. The two lateral pan trials and the two side-on trials scored zero. [Field-of-view and reception receipt](tt29_vision_gaze.json.gz)

## View and timing changes

On four selected measured openings, the independent planner scored two legal returns with normal stereo, one with a single fixed left camera, three with a 50 ms image delay, and two with a partial and then full short camera dropout. This small set measures sensitivity on matched launches. A separate six-opening assessment of a continuously retargeted analytic controller recorded median position error of 0.12 cm in stereo and 0.52 cm in wrist-only among the available near-flight estimates. Rotating the fixed cameras raised the error to 20.11 cm. Registering ordinary RGB court patches restored it to 0.30 cm. That analytic motor controller scored no legal returns on that assessment. [Physical view comparison](tt29_vision_twin.json.gz) · [Optics and analytic-controller assessment](tt29_vision_assay.json.gz)
