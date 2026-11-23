# Feedback learned in randomized 3D source worlds

The frozen policy averaged **19.55 verified rebounds per drop** on 500 fresh nominal worlds, compared with 1.97 for an idle paddle and 1.66 for a frozen random readout. It averaged **15.70** on 500 separate stress worlds, compared with 1.15 idle and 1.02 random. These are results from bounce-like pretraining with a supplied task-space representation. The source-world scores do not measure physical-arm deployment. [Policy and full qualification receipt](world-prior-source.json)

| Fresh source worlds | Episodes per treatment | Learned mean | Idle mean | Frozen random mean | Learned median | Learned best | Learned ≥10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal randomized physics | 500 | 19.546 | 1.972 | 1.656 | 20 | 24 | 498/500 |
| Wider, slower stress physics | 500 | 15.702 | 1.148 | 1.024 | 18 | 26 | 404/500 |

The nominal released policy averaged a 0.071 m rebound rise and 17.11 rebounds per drop with positive disk vertical velocity at contact above 0.055 m/s. All 500 nominal held drops remained in play for the five-second budget. On the stress group the mean longest sequence of consecutive verified bounces was 12.79, compared with 17.83 nominal. Source counts are paired on the same worlds for all treatments. [Per-world scores](world-prior-source.json)

## Simulation and learned policy

A sphere falls under randomized gravity and quadratic drag. Each contact is a crossing of the sphere radius plus disk half-thickness against a finite moving tilted disk. The impulse uses velocity relative to the disk at its actual contact point, normal restitution and tangential loss. Returns qualify after a distinct incoming impact, upward launch, at least 45 mm of rise, at least 18 mm of separation from the disk, and at least 65 ms before measuring the apex. Flight and contact scoring use the simulated geometry. The controller receives measured-style metric observations with random position and velocity errors. [Native simulator](../../scripts/experimental/tt24_world_prior_native.cpp)

The nominal training family varied gravity between 8.6 and 11.1 m/s², restitution between 0.68 and 0.84, disk radius between 69 and 83 mm, and command delay from zero to two ticks at 60 Hz. Ten of each 40 sampled training worlds came from a wider family with gravity 7.4–12.6 m/s², restitution 0.59–0.88 and delay up to four ticks. In both families release positions, lateral velocities, tangential loss, actuator response, initial pose and tilt, and observation noise also varied. The reserved stress family extends to gravity 6.9–13.0 m/s², restitution 0.54–0.78, delay up to six ticks, slower tilt response and larger velocity bias. The disk is a task-space plant with servo-rate limits, not an arm mechanism. [Training code](../../scripts/experimental/tt24_world_prior_train.py)

The readout has 42 coefficients and shares the horizontal response under rotations and reflections. It uses smooth sums of the observed position, velocity and normal vectors and their magnitudes or products. Augmented random search evaluates paired coefficient perturbations on the same sampled worlds and updates from their achieved returns. Selection uses physical rebound count with a smaller consecutive-run term and a duration tie-break, on separately seeded development episodes. The feedback law has no elapsed-time input. The source optimizer receives only observations and scores from its own bounce-world rollouts. [Policy interface](../../scripts/experimental/tt24_world_prior.py)

## Source experience and deployment contract

| Access | Rollouts | Control frames |
| --- | ---: | ---: |
| Narrow preliminary training | 239,616 | 70,237,247 |
| Preliminary development evaluation | 2,160 | 598,368 |
| Final mixed-family training | 732,160 | 215,503,873 |
| Final checkpoint development selection | 10,080 | 2,839,581 |
| Frozen held and stress evaluation | 3,000 | 430,239 |

The final run started from zero coefficients and ran on two native CPU workers for 14.6 seconds of rollout search. The held and stress seeds were disjoint from training and development, and the submitted weights were frozen before these scores were read. The two software checks also ran 40 paired idle smoke-test rollouts, separate from fitting. The preliminary checkpoint and its exposure record are preserved separately. [Preliminary receipt](world-prior-pilot.json) · [Frozen weights](world-prior.npz)

`WorldPrior.load(path).target(state)` accepts the ball xyz offset from nominal blade home in metres, ball xyz velocity in m/s, current blade xyz offset, and upward-normal x/y. It returns an absolute requested blade offset in metres and upward-normal x/y. The output bounds are 69 mm horizontal displacement, ±39 mm vertical displacement and normal x/y magnitude at most 0.181. The receiving arm must supply metric visual state and task-space grounding. The tests verify horizontal equivariance, Python/native action agreement and deterministic physical scoring. [Deployment interface](../../scripts/experimental/tt24_world_prior.py) · [Tests](../../tests/test_tt24_world_prior.py)
