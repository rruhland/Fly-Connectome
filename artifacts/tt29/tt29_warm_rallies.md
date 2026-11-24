# Bounce calibration in continuous regulation points

The retained four-drop camera registration improved initial wrist-only position estimates substantially. The two wrist-only pairs produced zero legal returns. The mixed-camera pairs produced two and four continuous legal returns under both registrations. [Paired point tapes and replay](tt29_warm_rallies.json.gz)

The policy used live RGB and encoders throughout each point. It ran the table action-memory learner from the same seven exemplars gathered before opening 7271 in the earlier stereo course. Every branch had a fresh flight-estimation history and the same opponent seed, fixture, arm posture, physical lens and planner. The legacy branch used the old wrist extrinsic with the current lens intrinsics. The retained branch used the wrist extrinsic learned during solo bouncing. The dynamics model and tool geometry stayed the same across branches. Each point ran from its recorded launch until the native physical referee ended it. [Continuous controller](../../scripts/experimental/tt29_action_memory.py) · [Comparison runner](../../scripts/experimental/tt29_warm_rallies.py)

| Measured opening | Physical camera input | Learner + partner legal returns, legacy | Learner + partner legal returns, retained |
| --- | --- | ---: | ---: |
| 7271, historical opportunity | Wrist at upcourt ready pose | 0 | 0 |
| 7271, historical opportunity | Fixed left and wrist at lateral ready pose | 2 | 2 |
| 4230, held from starting motor memory | Wrist at upcourt ready pose | 0 | 0 |
| 15489, held from starting motor memory | Fixed left and wrist at lateral ready pose | 4 | 4 |

On the first 12 camera frames, median position error among available wrist-only estimates fell from 2.82 m to 0.15 m on 7271 and from 2.58 m to 0.23 m on 4230. The detector saw ball-like wrist silhouettes in 11 and 12 frames, respectively. At lateral ready the wrist did not detect the ball during the same window for either mixed-camera opening. The fixed left image supplied the opening measurements. The two mixed-camera 7271 returns shared a median outgoing speed of 4.92 m/s. The legacy point ended when the ball hit the camera foot. The retained point ended out. [Opening optical and end-state audit](tt29_warm_rallies.json.gz)

All eight command tapes reproduced the native referee outcomes. A full-CAD shadow found no checked penetrations above 3 mm across 7,930 solver samples. The four-return points shared a median outgoing speed of 5.75 m/s. Their peak measured elbow rates reached 5.88 rad/s with the retained calibration and 5.99 rad/s with the legacy calibration, both above the 5.45 rad/s unloaded motor reference. [Physical and motor replay audit](tt29_warm_rallies.json.gz)

The 66-return result used the same recorded opening with two fixed cameras, in a previous continuous course that could carry optical practice from earlier points. It provides context for the endurance gap. These pairs held the seven starting motor examples fixed and reset optical practice at each point. The camera calibration transfer changes what the learner sees. The one-opening receiving result in the bounce report uses a separate controller and assesses a different task. [Historical stereo course](camera-action-memory.json.gz) · [Bounce receiving assessment](tt29_bounce.md)
