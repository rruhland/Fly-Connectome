# What carried into a new table-tennis game

## The same arm with delayed controls

Two actual wrist-camera bouncing releases with delayed and noisy commands gave the learner a useful piece of information for its next game. From its issued commands and observed encoders, it identified a two-tick transport delay before playing any table tennis. The first two delayed drops each included a qualified rebound. The learner supplied its estimated delay to the regulation-table controller, where two control ticks correspond to 33 ms. The receiving arm used external stereo, the same 40 starting motor examples as the cold condition, and ongoing legal-landing updates.

| Measured opening | Cold, no timing carry | Timing carried from bouncing | Timing plus nominal queued-arm simulation |
| --- | ---: | ---: | ---: |
| 4527 | 10 | 60 | 40 |
| 7345 | 16 | 30 | 16 |

All entries count alternating legal returns in an uninterrupted point. Each column started from the same motor checkpoint and used the same opponent seeds (293594 and 296413), plant, camera schedule and noise stream. Each played 4527 first and carried its own subsequent return experience to 7345. Thus the second opening measures a continuing course. The two warmed rows used the same frozen lag estimate, with zero extra table probes. The more elaborate queued-arm forecast underperformed the simpler encoder-based forecast on both openings. [Cold games](solo-timing-table-cold.json.gz) · [Warmed games](solo-timing-table-warm.json.gz) · [Queued-arm comparison](queued-arm.json.gz)

The estimator learned a nominal response model from earlier solo-bouncing commands and encoders. During later solo play it compared predictions for candidate delays using at most the first 29 control intervals of each release. It correctly distinguished 0, 1, 2 and 3 control ticks on both releases per setting. The two release seeds were shared among settings. The model used an independent game referee only for evaluating rebounds. Eight command replays reproduced every encoder sample and rebound score. The transfer demonstrated here is **motor timing**. The experiments on eye calibration alone gave equal or shorter full-game rallies after warm-up. [Solo timing observations and fitted lag](bounce-timing.json.gz) · [Estimator](../../scripts/experimental/tt29_bounce_timing.py) · [Same-lens wrist comparison](tt29_wrist_only.md)

## Updating the body and the environment model

| Physical or visual change | Evidence the learner received | Matched result |
| --- | --- | --- |
| Heavier paddle, elbow and wrist and more joint damping | 32 ticks of issued joint commands and encoders before the first point | On openings 7271 and 7345, stale dynamics scored 15 and 12. With the brief body fit the scores were 100 and 44. A third adapted opening scored 4. |
| Shorter and narrower table, clamp moves with the end | One stereo exposure of the white court lines, nominal outside-camera calibration and known tabletop height | On 7271 the stock-sized model scored 0. Updating the model from the visible footprint scored 100. |
| Two fixed external cameras shifted by 80 mm and rotated 5° | A visible table, its regulation dimensions and the original survey | The re-registered controller scored 14 and 100 on 11292 and 7271. In an independent four-return-capped test, stale calibration scored 0/0, recovered calibration 4/0 and known camera poses 4/0. |
| Near-side command latency, image delay and noise | A separate 32-tick encoder probe before play | The learner identified two ticks. Its continuing four-point stream scored 2, 16, 24 and 40. A passive lag fit scored 0, 6, 14 and 16 on the same four points. |

The body fit changed effective link response parameters inside the near controller's nominal simulator. Individual mass estimates traded off against one another, and the inferred wrist mass was too low. The improved score demonstrates an effect on control. It does not establish individual link masses from one half-second motion. [Matched loaded plant](camera-shift-loaded_arm.json.gz) · [Adapted course](active-body.json.gz) · [100-return CAD and referee replay](loaded-camera100-audit.json)

The court-size estimate from RGB was 0.70499 m in half-width and 1.34010 m in half-length. The simulator dimensions used for scoring were 0.705 and 1.340 m. Each outside-camera update used the painted court boundary, with no added colored point markers. The full 100-return replays for both changes reproduced every native event and found positive CAD clearance throughout. The footprint test assumed the cameras stayed surveyed. The moved-camera test assumed a known regulation footprint. Jointly inferring both changes remains open. [Changed table without survey](narrow_table.json.gz) · [Table-size replay](survey-camera100-audit.json) · [Moved-camera replay](registered-camera100-audit.json) · [Independent calibration measurements](court-registration.md)

Opposing trajectories and materials remain important. The broader partner had 96 independently sampled combinations of requested lateral position, depth, height and pace. The first three points scored 22, 21 and 54. On one of those points, a net-cord deflection landed legally and bounced twice before the far carriage could return it. A table with two differently damped halves gave 30 and 12 returns with the carried stock model. [Broader opponent](varied-opponent.json.gz) · [Two-surface court](two_surface.json.gz) · [Time-delay probe](active-delay.json.gz)

## Runtime and scope

Each 100-return video shows uninterrupted simulated ball motion at its actual simulated pace. For a 256 × 192 stereo stream at 30 Hz, control at 60 Hz, and native ball physics at 480 Hz, the learner reached 100 and 88 on two successive openings. The 100-return point occupied 69.4 simulated seconds and used 114.8 CPU seconds for the full live loop. The arm retrieved 41 movements and used ten successful fallback plans. Shared-machine wall time depends heavily on concurrent jobs. CPU time includes rendering both cameras, operating both players, physics search and running the live simulator. The static-stereo cache reproduced all 268 held detections; it assumes stationary external cameras. [256 × 192 comparison](efficient-2.json.gz) · [Rendering and inference breakdown](tt29_fast_camera.md) · [Fast physical opponent](tt29_partner_fast.md)

A controller screening the actual joint speed of each proposed movement also reached 100 from stereo. Its peak measured speed was 5.448 rad/s and the closest CAD separation was 18.6 mm. The simulator capped torque at 3.35 N·m independently of speed for that trial. A separate plant reduces available motoring torque as speed rises, using a linear stall-to-no-load curve and retaining braking authority. The physical arm's under-load curve has not been measured in this project. These two plant models are distinct benchmarks. [Speed-screened pupil replay](rated-camera100-audit.json) · [Coupled motor model and sensitivity](rally-physics.md)

The stricter coupled motor model subsequently reached 200 alternating returns with stereo RGB. The new comparison includes complete duration, learning controls, and simultaneous perturbations. [Coupled-motor endurance and limits](coupled-endurance.md)
