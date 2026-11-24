# Full-size physical rally teacher

A simulated SO-101 at the regulation table and a force-driven carriage opponent completed **100 consecutive alternating legal returns** in 70.33 simulated seconds. The first delivery used launch 10542 from the pinned DeepMind recordings. The referee excluded that delivery from the count. The carriage varied its target across six forehand positions and pace settings. Every subsequent return came from a native racket contact, traveled across the net, and landed on the receiving half. An independent command replay reproduced all 502 events and sampled the arm geometry at all 33,760 solver steps. [Action and contact receipt](rally-physics-100.json.gz) · [30 FPS real-time replay](rally-physics-teacher.mp4)

| Measurement | 10th percentile | Median | 90th percentile |
| --- | ---: | ---: | ---: |
| SO-101 ball departure, m/s | 4.90 | 5.27 | 5.60 |
| Carriage ball departure, m/s | 5.55 | 6.00 | 6.14 |
| Time between contacts, s | 0.620 | 0.677 | 0.830 |
| Pinned DeepMind launch speed, m/s | 4.22 | 5.50 | 7.18 |

Departure speeds are the three-dimensional ball speeds at racket separation. The 100-return coach used exact ball state, numerical inverse kinematics, and a search over isolated copies of the MuJoCo dynamics. It issued signed servo target rates to the live arm. This run took 376 wall-clock seconds including search and control, so it demonstrates play at measured simulation pace with a compute-intensive coach. [DeepMind source and speed reference](../tt25/recorded-secondary-reference.json)

The compiled arm capped target slew at 5.2 rad/s for each driven axis and torque at 3.35 N·m. Actual joint speed peaked at 7.71 rad/s versus the servo's 5.45 rad/s nominal no-load rating. Two replay samples exhibited racket–table overlap above the 3 mm audit tolerance, with a 4.95 mm worst overlap. The run establishes continuous ball play and leaves hardware motion clearance and speed qualification open. A separate reduced-slew run reproduced four consecutive returns with zero reported geometric violations and 5.28 rad/s peak actual speed. [Short motor and clearance qualification](rally-physics-motor-clear.json.gz)

An earlier 41-return test used a tucked posture that intersected the robot's own CAD geometry throughout replay. That test isolated the receiving window. The later forehand setup repositioned the interception laterally and used a clearance screen. [Folded-posture diagnostic](rally-physics-folded-diagnostic.json.gz)
